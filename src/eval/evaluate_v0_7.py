"""Held-out teacher-forced and closed-loop V0.7 closure evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from jka_model.config import ProjectConfig, load_config
from jka_model.data import ChannelStandardizer, SplitManifest, select_split
from jka_model.physics import weighted_integral_2d
from jka_model.problems import create_problem_adapter
from jka_model.residual import (
    ResidualKoopmanModel,
    ResidualWindowDataset,
    build_closure,
    closure_metrics,
    corrected_latent_rollout,
    load_residual_cache,
)
from jka_model.residual.checkpoint import load_residual_checkpoint
from train.train_v0_6 import initialize_v0_6_model
from train.train_v0_7 import _batch_to_device, _require_v0_7


def _load_model(
    config: ProjectConfig, payload: dict[str, Any], device: torch.device
) -> ResidualKoopmanModel:
    assert config.residual_closure and config.koopman
    parameter_dim = config.data.parameter_dim if config.residual_closure.include_parameters else 0
    closure = build_closure(
        str(payload["closure_variant"]),
        latent_dim=config.koopman.state_dim,
        history=config.residual_closure.history,
        parameter_dim=parameter_dim,
        hidden_dim=config.residual_closure.hidden_dim,
        depth=config.residual_closure.depth,
    )
    backbone = initialize_v0_6_model(config, device=device)
    model = ResidualKoopmanModel(backbone, closure).to(device)
    model.load_backbone_state_dict(payload["backbone_state"])
    model.residual_head.load_state_dict(payload["closure_state"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def evaluate_v0_7(
    config: ProjectConfig | str | Path,
    *,
    checkpoint: str | Path,
    cache_path: str | Path,
    device: str | torch.device | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    _require_v0_7(resolved)
    assert resolved.residual_closure and resolved.memory_sweep and resolved.v0_7_evaluation
    selected = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
    )
    payload = load_residual_checkpoint(checkpoint)
    if payload["config_hash"] != resolved.stable_hash:
        raise ValueError("V0.7 evaluation config/checkpoint mismatch")
    cache = load_residual_cache(cache_path)
    if payload["cache_fingerprint"] != cache.fingerprint:
        raise ValueError("V0.7 evaluation cache/checkpoint mismatch")
    model = _load_model(resolved, payload, selected)
    parameter_count = sum(parameter.numel() for parameter in model.residual_head.parameters())
    dataset = ResidualWindowDataset(
        cache,
        "test",
        resolved.residual_closure.history,
        shuffle_history=payload["closure_variant"] == "shuffled_history",
        shuffle_seed=resolved.training.seed,
    )
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        batch = {
            key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
            for key, value in sample.items()
        }
        history_z, history_dts, next_dt, parameters, target = _batch_to_device(
            batch, selected, resolved.residual_closure.include_parameters
        )
        predictions.append(model.residual_head(history_z, history_dts, next_dt, parameters).cpu())
        targets.append(target.cpu())
    teacher_forced = closure_metrics(torch.cat(predictions), torch.cat(targets))
    normalizer = ChannelStandardizer(eps=resolved.data.normalization.eps)
    normalizer.load_state_dict(payload["normalizer_state"])
    adapter = create_problem_adapter(resolved)
    records = adapter.build_dataset(seed=resolved.training.seed)
    manifest = SplitManifest.from_dict(payload["split_manifest"])
    test_records = select_split(records, manifest, "test")
    cached_lookup = {item.trajectory_id: item for item in cache.select("test")}
    spec = adapter.build_problem_spec()
    operator = adapter.build_physics_constraints()["operator"]
    rollout: dict[str, dict[str, Any]] = {}
    for horizon in resolved.v0_7_evaluation.rollout_horizons:
        latent_rmse: list[float] = []
        field_rmse: list[float] = []
        relative_l2: list[float] = []
        mass_drift: list[float] = []
        operator_mse: list[float] = []
        burden: list[float] = []
        burden_curves: list[torch.Tensor] = []
        latent_rmse_curves: list[torch.Tensor] = []
        field_rmse_curves: list[torch.Tensor] = []
        usable = 0
        for record in test_records:
            cached = cached_lookup[record.trajectory_id]
            history = resolved.residual_closure.history
            start = history - 1
            if start + horizon > cached.dts.shape[0]:
                continue
            usable += 1
            initial_history = cached.latents[:history].to(selected).unsqueeze(0)
            history_dts = cached.dts[: history - 1].to(selected).unsqueeze(0)
            future_dts = cached.dts[start : start + horizon].to(selected).unsqueeze(0)
            parameters = cached.parameters.to(selected).unsqueeze(0)
            if not resolved.residual_closure.include_parameters:
                parameters = parameters[:, :0]
            predicted_z, base_z, correction = corrected_latent_rollout(
                model, initial_history, history_dts, future_dts, parameters
            )
            truth_z = cached.latents[start : start + horizon + 1].to(selected).unsqueeze(0)
            latent_rmse.append(float((predicted_z[:, 1:] - truth_z[:, 1:]).square().mean().sqrt()))
            latent_rmse_curves.append(
                (predicted_z[:, 1:] - truth_z[:, 1:]).square().mean(dim=(0, 2)).sqrt().cpu()
            )
            predicted_raw = normalizer.inverse_transform(
                model.training_decoder(predicted_z[:, 1:])
            )[0]
            truth_raw = record.states_raw[start + 1 : start + horizon + 1].to(
                selected, torch.float32
            )
            error = predicted_raw - truth_raw
            field_rmse.append(float(error.square().mean().sqrt()))
            field_rmse_curves.append(error.square().mean(dim=(1, 2, 3)).sqrt().cpu())
            relative_l2.append(float(error.norm() / truth_raw.norm().clamp_min(1e-12)))
            weights = record.cell_weights.to(selected, torch.float32)
            initial_raw = record.states_raw[start : start + 1].to(selected, torch.float32)
            initial_mass = weighted_integral_2d(initial_raw, weights.unsqueeze(0))
            masses = weighted_integral_2d(predicted_raw, weights)
            scale = initial_raw.abs().mul(weights).sum(dim=(-2, -1)).clamp_min(1e-12)
            mass_drift.append(float(((masses - initial_mass).abs() / scale).max()))
            terms: list[torch.Tensor] = []
            previous = initial_raw
            metadata = {
                "mu_static": record.mu_static.to(selected, torch.float32).unsqueeze(0),
                "cell_weights": weights.unsqueeze(0),
            }
            for step in range(horizon):
                current = predicted_raw[step : step + 1]
                terms.append(
                    next(
                        iter(
                            operator.loss(
                                current,
                                prev_state_raw=previous,
                                dt=future_dts[:, step],
                                spec=spec,
                                metadata=metadata,
                            ).values()
                        )
                    )
                )
                previous = current
            operator_mse.append(float(torch.stack(terms).mean()))
            base_increment = base_z - predicted_z[:, :-1]
            burden_curve = correction.norm(dim=-1) / (
                correction.norm(dim=-1) + base_increment.norm(dim=-1) + 1e-12
            )
            burden.append(float(burden_curve.mean()))
            burden_curves.append(burden_curve[0].cpu())
        if usable == 0:
            continue
        rollout[str(horizon)] = {
            "trajectory_count": float(usable),
            "latent_rmse": sum(latent_rmse) / usable,
            "field_rmse": sum(field_rmse) / usable,
            "relative_l2": sum(relative_l2) / usable,
            "mass_drift": sum(mass_drift) / usable,
            "operator_mse": sum(operator_mse) / usable,
            "closure_burden": sum(burden) / usable,
            "closure_burden_by_step": torch.stack(burden_curves).mean(dim=0).tolist(),
            "latent_rmse_by_step": torch.stack(latent_rmse_curves).mean(dim=0).tolist(),
            "field_rmse_by_step": torch.stack(field_rmse_curves).mean(dim=0).tolist(),
        }
    history = resolved.residual_closure.history
    physical_spans = [
        float(item.dts[: history - 1].sum()) if history > 1 else 0.0
        for item in cache.select("test")
    ]
    result = {
        "phase": "v0.7",
        "seed": resolved.training.seed,
        "variant": payload["closure_variant"],
        "closure_family": type(model.residual_head).__name__,
        "history_length_steps": history,
        "history_length_physical_time": {
            "mean": sum(physical_spans) / len(physical_spans),
            "min": min(physical_spans),
            "max": max(physical_spans),
        },
        "parameter_count": parameter_count,
        "parameter_matched_control": payload["closure_variant"] == "instantaneous",
        "history_shuffled": payload["closure_variant"] == "shuffled_history",
        "history_shuffle": payload["closure_variant"] == "shuffled_history",
        "teacher_forced": teacher_forced,
        "closed_loop": rollout,
        "provenance": {
            "backbone_checkpoint_sha256": cache.backbone_checkpoint_sha256,
            "backbone_config_hash": cache.backbone_config_hash,
            "cache_fingerprint": cache.fingerprint,
            "data_fingerprint": cache.data_fingerprint,
            "split_fingerprint": cache.split_fingerprint,
            "normalizer_fingerprint": cache.normalizer_fingerprint,
            "evaluation_trajectory_ids": sorted(item.trajectory_id for item in test_records),
        },
        "rollout_uses_predicted_history": True,
        "target_encoder_used": False,
        "physics_used_for_training": False,
        "exact_mori_zwanzig_kernel_claimed": False,
        "scientific_acceptance": "PENDING_MULTI_SEED_GPU_REVIEW",
        "memory_sweep_config": resolved.memory_sweep.to_dict(),
        "v0_7_evaluation_config": resolved.v0_7_evaluation.to_dict(),
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result
