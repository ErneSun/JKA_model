"""Locked V0.9 one-step, teacher-free rollout, operator, and physical evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from jka_model.adaptive import (
    AdaptiveRolloutDataset,
    AdaptiveWindowDataset,
    FactorizedAdaptiveOperator,
    adaptive_latent_rollout,
    causal_observer_features,
    condition_observer_metrics,
    condition_targets,
    latent_prediction_metrics,
    load_adaptive_cache,
    load_adaptive_checkpoint,
    matched_history_pairs,
    observable_error_attribution,
    operator_burden,
    operator_explained_fraction,
    residual_decomposition,
    symmetric_abscissa_proxy,
)
from jka_model.config import ProjectConfig, load_config
from jka_model.context.checkpoint import load_context_checkpoint
from jka_model.data import ChannelStandardizer
from jka_model.evaluation import (
    GateResult,
    GateStatus,
    MetricDirection,
    MetricGateSpec,
    aggregate_gate_results,
    evaluate_metric_gate,
)
from jka_model.problems import create_observable_problem_adapter
from jka_model.residual.cache import file_sha256
from jka_model.utils import load_checkpoint
from train.train_v0_6 import initialize_v0_6_model
from train.train_v0_9 import _build_model


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "NO_RECORDS"}]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _absolute_representation_spec(spec: MetricGateSpec) -> MetricGateSpec:
    """Use declared absolute physics tolerances for reconstruction-floor diagnosis."""
    if spec.threshold is None:
        raise ValueError("representation physical-floor metrics require an absolute threshold")
    return MetricGateSpec(
        spec.name,
        spec.direction,
        threshold=spec.threshold,
        resolution_floor=spec.resolution_floor,
    )


def phase2_required_control_names(mode: str, *, phase2_enabled: bool) -> tuple[str, ...]:
    """Return mechanism controls without coupling the known oracle to Q(history)."""
    if mode not in {"known", "latent_inferred"}:
        raise ValueError("invalid V0.9 condition mode")
    names = ["dynamic_over_static", "history_over_shuffled"]
    if phase2_enabled:
        if mode == "latent_inferred":
            names.append("condition_observer")
        names.append("paired_history_gain")
    return tuple(names)


def _normalized_condition(
    value: torch.Tensor,
    payload: dict[str, Any],
    mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    if mode == "latent_inferred":
        return None
    mean = torch.as_tensor(payload["condition_mean"], device=device).float()
    std = torch.as_tensor(payload["condition_std"], device=device).float()
    return (value.to(device).float() - mean) / std


def _condition_value(
    raw: dict[str, Any],
    resolved: ProjectConfig,
    payload: dict[str, Any],
    mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    phase2 = resolved.v0_9_phase2 is not None and resolved.v0_9_phase2.enabled
    key = "condition_target" if phase2 else "condition"
    return _normalized_condition(raw[key], payload, mode, device)


@torch.no_grad()
def _condition_only_rollout(
    model: torch.nn.Module,
    initial_history: torch.Tensor,
    history_dts: torch.Tensor,
    future_dts: torch.Tensor,
    context_parameters: torch.Tensor,
    conditions: torch.Tensor | None,
) -> torch.Tensor:
    """Teacher-free rollout under A0 plus only the condition branch."""
    adapter = model.operator_adapter
    if not isinstance(adapter, FactorizedAdaptiveOperator):
        raise TypeError("condition-only rollout requires the Phase-2 adapter")
    history = initial_history.clone()
    dt_history = history_dts.clone()
    states: list[torch.Tensor] = []
    for index in range(future_dts.shape[1]):
        dt = future_dts[:, index : index + 1]
        condition = None if conditions is None else conditions[:, index]
        context = model.context_encoder(history, dt_history, dt, context_parameters)
        components = adapter.phase2_components(
            context,
            condition,
            observer_features=causal_observer_features(history, dt_history),
            active_components="static",
        )
        generator = adapter.nominal_generator.unsqueeze(0) + components["static_delta"]
        transition = torch.linalg.matrix_exp(generator.float() * dt.reshape(-1, 1, 1))
        prediction = torch.einsum("bij,bj->bi", transition, history[:, -1].float())
        states.append(prediction)
        if history.shape[1] > 1:
            history = torch.cat((history[:, 1:], prediction.unsqueeze(1)), dim=1)
            dt_history = torch.cat((dt_history[:, 1:], dt), dim=1) if history.shape[1] > 2 else dt
        else:
            history = prediction.unsqueeze(1)
    return torch.stack(states, dim=1)


@torch.no_grad()
def _mean_training_delta(
    model: torch.nn.Module,
    dataset: AdaptiveWindowDataset,
    resolved: ProjectConfig,
    payload: dict[str, Any],
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for raw in DataLoader(dataset, batch_size=512, shuffle=False):
        history_z = raw["history_z"].to(device).float()
        history_dts = raw["history_dts"].to(device).float()
        next_dt = raw["next_dt"].to(device).float()
        parameters = raw["context_parameters"].to(device).float()
        condition = _condition_value(raw, resolved, payload, mode, device)
        _, _, _, delta, _ = model(history_z, history_dts, next_dt, parameters, condition)
        values.append(delta.cpu())
    return torch.cat(values).mean(dim=0)


def _load_backbone(
    config: ProjectConfig, checkpoint: str | Path, device: torch.device
) -> torch.nn.Module:
    saved = load_checkpoint(checkpoint, map_location="cpu")
    if saved.online_model_state is None or saved.target_model_state is None:
        raise ValueError("V0.9 backbone checkpoint lacks JEPA online/target state")
    if saved.config is None:
        raise ValueError("V0.9 backbone checkpoint lacks a resolved config")
    for name in ("koopman", "field_autoencoder", "field_loss", "jepa_loss", "ema"):
        if getattr(saved.config, name).to_dict() != getattr(config, name).to_dict():
            raise ValueError(f"V0.9 backbone inheritance mismatch in {name}")
    model = initialize_v0_6_model(config, device=device)
    model.load_online_state_dict(saved.online_model_state)
    model.target_encoder.load_state_dict(saved.target_model_state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def evaluate_v0_9(
    config: ProjectConfig | str | Path,
    *,
    checkpoint: str | Path,
    context_checkpoint: str | Path,
    adaptive_cache: str | Path,
    backbone_checkpoint: str | Path,
    physical_dataset: str | Path,
    output_dir: str | Path,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    if not all(
        (
            resolved.v0_8_context,
            resolved.v0_9_adaptive,
            resolved.v0_9_evaluation,
        )
    ):
        raise ValueError("evaluate_v0_9 requires the complete V0.9 contract")
    assert resolved.v0_8_context and resolved.v0_9_adaptive
    assert resolved.v0_9_evaluation
    selected = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
    )
    payload = load_adaptive_checkpoint(checkpoint)
    if payload["config_hash"] != resolved.stable_hash:
        raise ValueError("V0.9 evaluation config/checkpoint mismatch")
    cache = load_adaptive_cache(adaptive_cache)
    if payload["adaptive_cache_fingerprint"] != cache.fingerprint:
        raise ValueError("V0.9 evaluation cache/checkpoint mismatch")
    if payload["context_checkpoint_sha256"] != file_sha256(context_checkpoint):
        raise ValueError("V0.9 evaluation context/checkpoint mismatch")
    if payload["backbone_checkpoint_sha256"] != file_sha256(backbone_checkpoint):
        raise ValueError("V0.9 evaluation backbone/checkpoint mismatch")
    context_payload = load_context_checkpoint(context_checkpoint)
    model = _build_model(resolved, cache, context_payload, selected)
    model.operator_adapter.load_state_dict(payload["best_adaptive_state"], strict=True)
    model.requires_grad_(False)
    model.eval()
    mode = resolved.v0_9_adaptive.condition_mode
    history = int(context_payload["history_length_steps"])
    test_dataset = AdaptiveWindowDataset(cache, "test", history)
    train_dataset = AdaptiveWindowDataset(cache, "train", history)
    static_delta = _mean_training_delta(model, train_dataset, resolved, payload, mode, selected).to(
        selected
    )
    nominal_a = cache.nominal_generator.to(selected)
    one_step_rows: list[dict[str, Any]] = []
    all_nominal: list[torch.Tensor] = []
    all_remaining: list[torch.Tensor] = []
    observer_predictions: list[torch.Tensor] = []
    observer_targets: list[torch.Tensor] = []
    phase2_enabled = bool(resolved.v0_9_phase2 and resolved.v0_9_phase2.enabled)
    for raw in DataLoader(test_dataset, batch_size=512, shuffle=False):
        history_z = raw["history_z"].to(selected).float()
        history_dts = raw["history_dts"].to(selected).float()
        next_dt = raw["next_dt"].to(selected).float()
        parameters = raw["context_parameters"].to(selected).float()
        truth = raw["target_next"].to(selected).float()
        condition = _condition_value(raw, resolved, payload, mode, selected)
        prediction, context, eta, delta, adapted_a = model(
            history_z, history_dts, next_dt, parameters, condition
        )
        gate = model.operator_adapter.adaptation_gate(context, condition)
        nominal_transition = torch.linalg.matrix_exp(
            nominal_a.unsqueeze(0) * next_dt.reshape(-1, 1, 1)
        )
        nominal = torch.einsum("bij,bj->bi", nominal_transition, history_z[:, -1])
        global_static_transition = torch.linalg.matrix_exp(
            (nominal_a + static_delta).unsqueeze(0) * next_dt.reshape(-1, 1, 1)
        )
        global_static = torch.einsum("bij,bj->bi", global_static_transition, history_z[:, -1])
        static = global_static
        if phase2_enabled:
            if not isinstance(model.operator_adapter, FactorizedAdaptiveOperator):
                raise RuntimeError("Phase-2 evaluation loaded the wrong adapter")
            observer_features = causal_observer_features(history_z, history_dts)
            components = model.operator_adapter.phase2_components(
                context, condition, observer_features=observer_features
            )
            gate = components["gate"]
            static_components = model.operator_adapter.phase2_components(
                context,
                condition,
                observer_features=observer_features,
                active_components="static",
            )
            condition_generator = nominal_a.unsqueeze(0) + static_components["static_delta"]
            static_transition = torch.linalg.matrix_exp(
                condition_generator.float() * next_dt.reshape(-1, 1, 1)
            )
            static = torch.einsum("bij,bj->bi", static_transition, history_z[:, -1])
            target = (
                raw["condition_target"].to(selected).float()
                - torch.as_tensor(payload["condition_mean"], device=selected).float()
            ) / torch.as_tensor(payload["condition_std"], device=selected).float()
            observer_predictions.append(components["q_hat"].cpu())
            observer_targets.append(target.cpu())
        r0, _, remaining = residual_decomposition(truth, nominal, prediction)
        all_nominal.append(r0.cpu())
        all_remaining.append(remaining.cpu())
        burdens = operator_burden(delta, nominal_a)
        proxy = symmetric_abscissa_proxy(adapted_a)
        for index in range(truth.shape[0]):
            dynamic_metrics = latent_prediction_metrics(
                truth[index : index + 1],
                prediction[index : index + 1],
                nominal[index : index + 1],
            )
            static_rmse = float((static[index] - truth[index]).square().mean().sqrt())
            global_static_rmse = float((global_static[index] - truth[index]).square().mean().sqrt())
            one_step_rows.append(
                {
                    "trajectory_id": raw["trajectory_id"][index],
                    "target_index": int(raw["target_index"][index]),
                    "relative_transition_index": int(raw["relative_transition_index"][index]),
                    "schedule_type": raw["schedule_type"][index],
                    **dynamic_metrics,
                    "static_latent_rmse": static_rmse,
                    "global_static_latent_rmse": global_static_rmse,
                    "condition_only_latent_rmse": static_rmse,
                    "condition_only_relative_gain": 1.0
                    - static_rmse / max(dynamic_metrics["nominal_latent_rmse"], 1e-12),
                    "static_relative_gain": 1.0
                    - dynamic_metrics["latent_rmse"] / max(static_rmse, 1e-12),
                    "dynamic_over_condition_only_gain": 1.0
                    - dynamic_metrics["latent_rmse"] / max(static_rmse, 1e-12),
                    "operator_burden": float(burdens[index]),
                    "symmetric_abscissa_proxy": float(proxy[index]),
                    "eta_norm": float(eta[index].norm()),
                    "trust_gate": float(gate[index, 0]),
                }
            )
    gamma = float(operator_explained_fraction(torch.cat(all_nominal), torch.cat(all_remaining)))

    shuffled_gain: float | None = None
    if history > 1:
        shuffled = AdaptiveWindowDataset(
            cache, "test", history, shuffle_older_history=True, shuffle_seed=7919
        )
        errors: list[float] = []
        real_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        shuffled_loader = DataLoader(shuffled, batch_size=512, shuffle=False)
        for real_raw, raw in zip(real_loader, shuffled_loader, strict=True):
            condition = _condition_value(real_raw, resolved, payload, mode, selected)
            history_z = real_raw["history_z"].to(selected).float()
            history_dts = real_raw["history_dts"].to(selected).float()
            next_dt = real_raw["next_dt"].to(selected).float()
            parameters = real_raw["context_parameters"].to(selected).float()
            if phase2_enabled:
                adapter = model.operator_adapter
                assert isinstance(adapter, FactorizedAdaptiveOperator)
                real_context = model.context_encoder(history_z, history_dts, next_dt, parameters)
                real_observer_features = causal_observer_features(history_z, history_dts)
                shuffled_context = model.context_encoder(
                    raw["history_z"].to(selected).float(),
                    raw["history_dts"].to(selected).float(),
                    next_dt,
                    parameters,
                )
                real_components = adapter.phase2_components(
                    real_context,
                    condition,
                    observer_features=real_observer_features,
                )
                shuffled_components = adapter.phase2_components(
                    real_context,
                    condition,
                    dynamic_context=shuffled_context,
                    observer_features=real_observer_features,
                    condition_override=real_components["q_used"],
                )
                generator = adapter.nominal_generator.unsqueeze(0) + shuffled_components["delta"]
                transition = torch.linalg.matrix_exp(generator.float() * next_dt.reshape(-1, 1, 1))
                prediction = torch.einsum("bij,bj->bi", transition, history_z[:, -1])
            else:
                prediction, *_ = model(
                    raw["history_z"].to(selected).float(),
                    raw["history_dts"].to(selected).float(),
                    next_dt,
                    parameters,
                    condition,
                )
            errors.append(
                float(
                    (prediction - real_raw["target_next"].to(selected).float())
                    .square()
                    .mean()
                    .sqrt()
                )
            )
        real = sum(float(row["latent_rmse"]) for row in one_step_rows) / len(one_step_rows)
        shuffled_rmse = sum(errors) / len(errors)
        shuffled_gain = 1.0 - real / max(shuffled_rmse, 1e-12)

    rollout_rows: list[dict[str, Any]] = []
    diagnostic: dict[str, Any] | None = None
    condition_mean = torch.as_tensor(payload["condition_mean"], device=selected).float()
    condition_std = torch.as_tensor(payload["condition_std"], device=selected).float()
    for trajectory in cache.select("test"):
        for horizon in resolved.v0_9_evaluation.rollout_horizons:
            start = history - 1
            if start + horizon > trajectory.dts.shape[0]:
                continue
            conditions = None
            if mode == "known":
                trajectory_condition = (
                    condition_targets(trajectory.conditions, trajectory.dts)
                    if phase2_enabled
                    else trajectory.conditions
                )
                conditions = (
                    trajectory_condition[start : start + horizon].to(selected) - condition_mean
                ) / condition_std
                conditions = conditions.unsqueeze(0)
            bundle = adaptive_latent_rollout(
                model,
                trajectory.latents[:history].to(selected).unsqueeze(0),
                trajectory.dts[: history - 1].to(selected).unsqueeze(0),
                trajectory.dts[start : start + horizon].to(selected).unsqueeze(0),
                trajectory.context_parameters.to(selected).unsqueeze(0),
                conditions,
            )
            truth = trajectory.latents[start : start + horizon + 1].to(selected).unsqueeze(0)
            metrics = latent_prediction_metrics(
                truth[:, 1:], bundle["adapted"][:, 1:], bundle["nominal"][:, 1:]
            )
            burden_curve = operator_burden(bundle["delta_a"][0], nominal_a)
            rollout_rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "schedule_type": trajectory.schedule_type,
                    "horizon": horizon,
                    **metrics,
                    "operator_burden_mean": float(burden_curve.mean()),
                    "operator_burden_max": float(burden_curve.max()),
                    "trust_gate_mean": float(bundle["gate"].mean()),
                    "trust_gate_max": float(bundle["gate"].max()),
                    "finite": bool(torch.isfinite(bundle["adapted"]).all()),
                }
            )
            if diagnostic is None and horizon == max(resolved.v0_9_evaluation.rollout_horizons):
                diagnostic = {
                    "trajectory_id": trajectory.trajectory_id,
                    "truth": truth.cpu(),
                    **{key: value.cpu() for key, value in bundle.items()},
                }

    backbone = _load_backbone(resolved, backbone_checkpoint, selected)
    normalizer = ChannelStandardizer(eps=resolved.data.normalization.eps)
    normalizer.load_state_dict(cache.normalizer_state)
    problem = create_observable_problem_adapter(resolved)
    configured_source = getattr(getattr(problem, "config", None), "dataset_path", None)
    if configured_source and Path(configured_source).resolve() != Path(physical_dataset).resolve():
        raise ValueError("V0.9 evaluation observable dataset/config source mismatch")
    physical_lookup = {
        record.trajectory_id: record
        for record in problem.build_dataset(seed=resolved.training.seed)
    }
    observable_objective = problem.build_observable_objective(
        training=resolved.v0_9_training,
        evaluation=resolved.v0_9_evaluation,
    )
    physical_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    attribution_levels: list[dict[str, dict[str, float]]] = []
    longest = max(resolved.v0_9_evaluation.rollout_horizons)
    for trajectory in cache.select("test"):
        start = history - 1
        if start + longest > trajectory.dts.shape[0]:
            continue
        conditions = None
        if mode == "known":
            trajectory_condition = (
                condition_targets(trajectory.conditions, trajectory.dts)
                if phase2_enabled
                else trajectory.conditions
            )
            conditions = (
                trajectory_condition[start : start + longest].to(selected) - condition_mean
            ) / condition_std
            conditions = conditions.unsqueeze(0)
        bundle = adaptive_latent_rollout(
            model,
            trajectory.latents[:history].to(selected).unsqueeze(0),
            trajectory.dts[: history - 1].to(selected).unsqueeze(0),
            trajectory.dts[start : start + longest].to(selected).unsqueeze(0),
            trajectory.context_parameters.to(selected).unsqueeze(0),
            conditions,
        )
        record = physical_lookup[trajectory.trajectory_id]
        truth_raw = record.states_raw[start + 1 : start + longest + 1].to(selected)
        attribution_metrics: dict[str, dict[str, float]] = {
            "data": observable_objective.evaluation_metrics(
                truth_raw,
                truth_raw,
                {
                    "valid_mask": record.valid_mask,
                    "record_metadata": record.metadata,
                },
            )
        }
        reconstructed_raw = normalizer.inverse_transform(
            backbone.decode(
                trajectory.latents[start + 1 : start + longest + 1].to(selected).unsqueeze(0)
            )
        )[0]
        attribution_metrics["reconstruction"] = observable_objective.evaluation_metrics(
            reconstructed_raw,
            truth_raw,
            {
                "valid_mask": record.valid_mask,
                "record_metadata": record.metadata,
            },
        )
        for name, latent in (
            ("adaptive", bundle["adapted"][:, 1:]),
            ("nominal", bundle["nominal"][:, 1:]),
        ):
            raw = normalizer.inverse_transform(backbone.decode(latent))[0]
            assert record.valid_mask is not None
            metrics = observable_objective.evaluation_metrics(
                raw,
                truth_raw,
                {
                    "valid_mask": record.valid_mask,
                    "record_metadata": record.metadata,
                },
            )
            attribution_metrics[name] = metrics
            physical_rows.append(
                {
                    "model": name,
                    "trajectory_id": trajectory.trajectory_id,
                    "schedule_type": trajectory.schedule_type,
                    "horizon": longest,
                    **metrics,
                }
            )
        attribution_levels.append(attribution_metrics)
        attribution_rows.extend(
            {
                "trajectory_id": trajectory.trajectory_id,
                "schedule_type": trajectory.schedule_type,
                **row,
            }
            for row in observable_error_attribution(attribution_metrics)
        )

    observer_metrics: dict[str, float] | None = None
    if phase2_enabled:
        observer_metrics = condition_observer_metrics(
            torch.cat(observer_predictions), torch.cat(observer_targets)
        )

    paired_rows: list[dict[str, Any]] = []
    paired_gain: float | None = None
    paired_count = 0
    if phase2_enabled:
        assert resolved.v0_9_phase2 is not None
        paired_dataset = AdaptiveRolloutDataset(
            cache,
            "test",
            history,
            resolved.v0_9_phase2.paired_horizon,
            stride=1,
        )
        pair_conditions: list[torch.Tensor] = []
        pair_latents: list[torch.Tensor] = []
        pair_histories: list[torch.Tensor] = []
        pair_futures: list[torch.Tensor] = []
        full_errors: list[torch.Tensor] = []
        condition_errors: list[torch.Tensor] = []
        pair_ids: list[str] = []
        residual_scale = torch.as_tensor(
            payload["residual_training_scale"], device=selected
        ).float()
        for raw in DataLoader(paired_dataset, batch_size=256, shuffle=False):
            history_z = raw["history_z"].to(selected).float()
            history_dts = raw["history_dts"].to(selected).float()
            future_dts = raw["future_dts"].to(selected).float()
            parameters = raw["context_parameters"].to(selected).float()
            target = raw["target_latents"].to(selected).float()
            target_conditions = raw["future_condition_targets"].to(selected).float()
            normalized_targets = (target_conditions - condition_mean) / condition_std
            supplied = normalized_targets if mode == "known" else None
            full = adaptive_latent_rollout(
                model,
                history_z,
                history_dts,
                future_dts,
                parameters,
                supplied,
            )["adapted"][:, 1:]
            condition_only = _condition_only_rollout(
                model,
                history_z,
                history_dts,
                future_dts,
                parameters,
                supplied,
            )
            full_errors.append(
                ((full[:, -1] - target[:, -1]) / residual_scale).square().mean(dim=-1).sqrt().cpu()
            )
            condition_errors.append(
                ((condition_only[:, -1] - target[:, -1]) / residual_scale)
                .square()
                .mean(dim=-1)
                .sqrt()
                .cpu()
            )
            pair_conditions.append(normalized_targets[:, 0].cpu())
            pair_latents.append((history_z[:, -1] / residual_scale).cpu())
            pair_histories.append((history_z[:, :-1] / residual_scale).cpu())
            pair_futures.append((target / residual_scale).cpu())
            pair_ids.extend(str(value) for value in raw["trajectory_id"])
        selected_pairs = matched_history_pairs(
            torch.cat(pair_conditions),
            torch.cat(pair_latents),
            torch.cat(pair_histories),
            torch.cat(pair_futures),
            condition_tolerance=resolved.v0_9_phase2.matched_condition_tolerance,
            latent_tolerance=resolved.v0_9_phase2.matched_latent_tolerance,
            minimum_history_separation=(resolved.v0_9_phase2.minimum_history_separation),
            minimum_future_separation=(resolved.v0_9_phase2.minimum_future_separation),
            group_ids=pair_ids,
        )
        full_error = torch.cat(full_errors)
        condition_error = torch.cat(condition_errors)
        gains: list[float] = []
        for pair_index, pair in enumerate(selected_pairs):
            indices = torch.tensor((pair.first, pair.second), dtype=torch.long)
            full_rmse = float(full_error.index_select(0, indices).mean())
            condition_rmse = float(condition_error.index_select(0, indices).mean())
            gain = 1.0 - full_rmse / max(condition_rmse, 1.0e-12)
            gains.append(gain)
            paired_rows.append(
                {
                    "pair_index": pair_index,
                    "first_trajectory_id": pair_ids[pair.first],
                    "second_trajectory_id": pair_ids[pair.second],
                    "condition_distance": pair.condition_distance,
                    "latent_distance": pair.latent_distance,
                    "history_distance": pair.history_distance,
                    "future_separation": pair.future_separation,
                    "full_dynamic_rmse": full_rmse,
                    "condition_only_rmse": condition_rmse,
                    "dynamic_gain": gain,
                }
            )
        paired_count = len(gains)
        if paired_count >= resolved.v0_9_phase2.minimum_identifiable_pairs:
            paired_gain = sum(gains) / paired_count

    horizon_summary: dict[str, dict[str, Any]] = {}
    for horizon in resolved.v0_9_evaluation.rollout_horizons:
        rows = [row for row in rollout_rows if int(row["horizon"]) == horizon]
        gains = [float(row["relative_gain"]) for row in rows]
        gammas = [float(row["gamma_operator"]) for row in rows]
        horizon_summary[str(horizon)] = {
            "run_count": len(rows),
            "relative_gain_mean": sum(gains) / len(gains),
            "gamma_operator_mean": sum(gammas) / len(gammas),
            "pass": bool(
                rows
                and sum(gains) / len(gains) >= resolved.v0_9_evaluation.material_relative_gain
                and sum(gammas) / len(gammas)
                >= resolved.v0_9_evaluation.min_operator_explained_fraction
                and all(bool(row["finite"]) for row in rows)
            ),
        }
    all_horizons_pass = all(bool(item["pass"]) for item in horizon_summary.values())
    longest_pass = bool(horizon_summary[str(longest)]["pass"])
    adaptive_physics = [row for row in physical_rows if row["model"] == "adaptive"]
    nominal_physics = [row for row in physical_rows if row["model"] == "nominal"]
    observable_specs = observable_objective.evaluation_gate_specs(
        sequence_length=longest,
        dt=float(next(iter(physical_lookup.values())).dts[0]),
    )
    observable_gate_rows: list[dict[str, Any]] = []
    observable_pair_results: list[GateResult] = []
    for adapted_row, nominal_row in zip(adaptive_physics, nominal_physics, strict=True):
        metric_results = [
            evaluate_metric_gate(
                float(adapted_row[name]),
                spec,
                baseline=float(nominal_row[name]),
            )
            for name, spec in observable_specs.items()
        ]
        pair = aggregate_gate_results(
            f"observables:{adapted_row['trajectory_id']}",
            metric_results,
            required_pass_fraction=1.0,
            minimum_count=len(observable_specs),
        )
        observable_pair_results.append(pair)
        for result in metric_results:
            observable_gate_rows.append(
                {
                    "trajectory_id": adapted_row["trajectory_id"],
                    "schedule_type": adapted_row["schedule_type"],
                    "horizon": longest,
                    **result.to_dict(),
                }
            )
    observable_gate = aggregate_gate_results(
        "observable_noninferiority",
        observable_pair_results,
        required_pass_fraction=resolved.v0_9_evaluation.observable_pair_pass_fraction,
        minimum_count=1,
    )
    physics_pass = observable_gate.passed
    representation_gate_results: list[GateResult] = []
    for index, levels in enumerate(attribution_levels):
        metric_results = [
            evaluate_metric_gate(
                float(levels["reconstruction"][name]),
                _absolute_representation_spec(spec),
            )
            for name, spec in observable_specs.items()
            if name in {"divergence_rms", "boundary_no_slip_mse"}
        ]
        representation_gate_results.append(
            aggregate_gate_results(
                f"representation:{index}",
                metric_results,
                required_pass_fraction=1.0,
                minimum_count=2,
            )
        )
    representation_gate = aggregate_gate_results(
        "representation_physical_floor",
        representation_gate_results,
        required_pass_fraction=1.0,
        minimum_count=1,
    )
    representation_blocked = not representation_gate.passed
    burden_pass = all(
        float(row["operator_burden_max"]) <= resolved.v0_9_evaluation.max_operator_burden
        for row in rollout_rows
    )
    finite_rollout_pass = all(bool(row["finite"]) for row in rollout_rows)
    one_step_gain = sum(float(row["relative_gain"]) for row in one_step_rows) / len(one_step_rows)
    dynamic_over_static = sum(float(row["static_relative_gain"]) for row in one_step_rows) / len(
        one_step_rows
    )
    condition_only_gain = sum(
        float(row["condition_only_relative_gain"]) for row in one_step_rows
    ) / len(one_step_rows)
    one_step_gate = evaluate_metric_gate(
        one_step_gain,
        MetricGateSpec(
            "one_step_prediction",
            MetricDirection.HIGHER_IS_BETTER,
            threshold=resolved.v0_9_evaluation.material_relative_gain,
        ),
    )
    operator_gate = evaluate_metric_gate(
        gamma,
        MetricGateSpec(
            "operator_explained_residual",
            MetricDirection.HIGHER_IS_BETTER,
            threshold=resolved.v0_9_evaluation.min_operator_explained_fraction,
        ),
    )
    dynamic_gate = evaluate_metric_gate(
        dynamic_over_static,
        MetricGateSpec(
            "dynamic_over_static",
            MetricDirection.HIGHER_IS_BETTER,
            threshold=resolved.v0_9_evaluation.min_dynamic_over_static_gain,
        ),
    )
    condition_only_gate = evaluate_metric_gate(
        condition_only_gain,
        MetricGateSpec(
            "condition_only_over_nominal",
            MetricDirection.HIGHER_IS_BETTER,
            threshold=resolved.v0_9_evaluation.material_relative_gain,
        ),
    )
    observer_gate = GateResult(
        "condition_observer",
        GateStatus.PASS,
        None,
        None,
        "Phase 2 is disabled",
    )
    paired_gate = GateResult(
        "paired_history_gain",
        GateStatus.PASS,
        None,
        None,
        "Phase 2 is disabled",
    )
    if phase2_enabled:
        assert resolved.v0_9_phase2 is not None and observer_metrics is not None
        observer_rmse_gate = evaluate_metric_gate(
            observer_metrics["normalized_rmse"],
            MetricGateSpec(
                "condition_observer_rmse",
                MetricDirection.LOWER_IS_BETTER,
                threshold=resolved.v0_9_phase2.max_condition_observer_normalized_rmse,
            ),
        )
        observer_r2_gate = evaluate_metric_gate(
            observer_metrics["minimum_r2"],
            MetricGateSpec(
                "condition_observer_r2",
                MetricDirection.HIGHER_IS_BETTER,
                threshold=resolved.v0_9_phase2.min_condition_observer_r2,
            ),
        )
        observer_gate = aggregate_gate_results(
            "condition_observer",
            [observer_rmse_gate, observer_r2_gate],
            required_pass_fraction=1.0,
            minimum_count=2,
        )
        if paired_gain is None:
            paired_gate = GateResult(
                "paired_history_gain",
                GateStatus.INCONCLUSIVE,
                None,
                resolved.v0_9_phase2.min_paired_dynamic_gain,
                f"only {paired_count} matched pairs; requires "
                f"{resolved.v0_9_phase2.minimum_identifiable_pairs}",
            )
        else:
            paired_gate = evaluate_metric_gate(
                paired_gain,
                MetricGateSpec(
                    "paired_history_gain",
                    MetricDirection.HIGHER_IS_BETTER,
                    threshold=resolved.v0_9_phase2.min_paired_dynamic_gain,
                ),
            )
    if str(context_payload["residual_route"]) == "R3":
        history_gate = evaluate_metric_gate(
            float("nan") if shuffled_gain is None else shuffled_gain,
            MetricGateSpec(
                "history_over_shuffled",
                MetricDirection.HIGHER_IS_BETTER,
                threshold=resolved.v0_9_evaluation.material_relative_gain,
            ),
        )
    else:
        history_gate = GateResult(
            "history_over_shuffled",
            GateStatus.PASS,
            shuffled_gain,
            None,
            "R2 route does not require a history-shuffle control",
        )
    control_lookup = {
        "dynamic_over_static": dynamic_gate,
        "history_over_shuffled": history_gate,
        "condition_observer": observer_gate,
        "paired_history_gain": paired_gate,
    }
    control_items = [
        control_lookup[name]
        for name in phase2_required_control_names(mode, phase2_enabled=phase2_enabled)
    ]
    controls_gate = aggregate_gate_results(
        "dynamic_controls",
        control_items,
        required_pass_fraction=1.0,
        minimum_count=len(control_items),
    )
    controls_pass = controls_gate.passed
    scientific_gates = {
        result.name: result.to_dict()
        for result in (
            one_step_gate,
            operator_gate,
            dynamic_gate,
            condition_only_gate,
            history_gate,
            observer_gate,
            paired_gate,
            controls_gate,
            observable_gate,
            representation_gate,
        )
    }
    supported = bool(
        one_step_gate.passed
        and operator_gate.passed
        and controls_pass
        and all_horizons_pass
        and longest_pass
        and physics_pass
        and burden_pass
    )
    decision = {
        "schema_version": 5,
        "backbone_seed": int(payload["backbone_seed"]),
        "context_init_seed": int(payload["context_init_seed"]),
        "operator_init_seed": int(payload["operator_init_seed"]),
        "problem_name": resolved.data.problem_name,
        "observable_objective": observable_objective.name,
        "condition_mode": mode,
        "rank": int(payload["rank"]),
        "one_step_relative_gain": one_step_gain,
        "operator_explained_fraction": gamma,
        "dynamic_over_static_gain": dynamic_over_static,
        "dynamic_over_condition_only_gain": dynamic_over_static,
        "condition_only_over_nominal_gain": condition_only_gain,
        "paired_history_gain": paired_gain,
        "paired_history_pair_count": paired_count,
        "condition_observer_metrics": observer_metrics,
        "history_over_shuffled_gain": shuffled_gain,
        "operator_explained_status": operator_gate.status.value,
        "dynamic_over_static_status": dynamic_gate.status.value,
        "dynamic_over_condition_only_status": dynamic_gate.status.value,
        "condition_only_status": condition_only_gate.status.value,
        "condition_observer_status": observer_gate.status.value,
        "paired_identifiability_status": paired_gate.status.value,
        "controls_status": "PASS" if controls_pass else "FAIL",
        "closed_loop_by_horizon": horizon_summary,
        "all_horizons_status": "PASS" if all_horizons_pass else "FAIL",
        "longest_horizon_status": "PASS" if longest_pass else "FAIL",
        "operator_burden_status": "PASS" if burden_pass else "FAIL",
        "numerical_stability": ("PASS" if finite_rollout_pass and burden_pass else "FAIL"),
        "long_rollout_skill": "PASS" if all_horizons_pass else "FAIL",
        # Compatibility alias.  From schema 4 onward this field has the literal
        # stability meaning; predictive skill is reported separately above.
        "long_rollout_stability": ("PASS" if finite_rollout_pass and burden_pass else "FAIL"),
        "physics_status": "PASS" if physics_pass else "FAIL",
        "observable_status": observable_gate.status.value,
        "representation_physical_floor_status": representation_gate.status.value,
        "phase1_diagnosis": "REPRESENTATION_BLOCKED"
        if representation_blocked
        else "OPERATOR_OPTIMIZATION_IDENTIFIABLE",
        "scientific_gates": scientific_gates,
        "adaptive_koopman": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "parameterized_koopman": ("SUPPORTED" if condition_only_gate.passed else "NOT_SUPPORTED"),
        "history_adaptation": "SUPPORTED" if controls_pass else "NOT_SUPPORTED",
        "phase2_classification": (
            "DYNAMIC_ADAPTIVE_KOOPMAN_SUPPORTED"
            if controls_pass
            else "PARAMETERIZED_KOOPMAN_SUPPORTED; HISTORY_ADAPTATION_NOT_REQUIRED"
            if condition_only_gate.passed and (mode == "known" or observer_gate.passed)
            else "LATENT_CONDITION_NOT_IDENTIFIABLE"
            if mode == "latent_inferred" and not observer_gate.passed
            else "PHASE2_NOT_SUPPORTED"
        ),
        "scientific_joint_pass": supported,
        "claims": {
            "backbone_frozen": True,
            "context_frozen": True,
            "A0_frozen": True,
            "additive_residual_enabled": False,
            "persistent_z_R_present": False,
            "phase1_error_attribution_complete": bool(attribution_rows),
            "phase2_factorized_operator": phase2_enabled,
            "condition_centered_history_innovation": phase2_enabled,
            "oracle_condition_curriculum_train_only": phase2_enabled,
            "locked_latent_evaluation_is_teacher_free": mode == "latent_inferred",
            "known_oracle_excludes_observer_gate": mode == "known",
            "causal_observer_uses_state_mean_trend": phase2_enabled,
            "dynamic_context_is_condition_residualized": phase2_enabled,
            "long_rollout_stability_is_numerical": True,
            "innovation_variance_floor": False,
        },
    }
    destination = Path(output_dir)
    evaluation = destination / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    _write_csv(evaluation / "one_step_operator_metrics.csv", one_step_rows)
    _write_csv(evaluation / "rollout_metrics.csv", rollout_rows)
    _write_csv(evaluation / "physical_metrics.csv", physical_rows)
    _write_csv(evaluation / "observable_gate_results.csv", observable_gate_rows)
    _write_csv(evaluation / "error_attribution.csv", attribution_rows)
    if phase2_enabled:
        _write_csv(evaluation / "matched_history_pairs.csv", paired_rows)
        (evaluation / "condition_observer_metrics.json").write_text(
            json.dumps(observer_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (evaluation / "matched_history_pairs.json").write_text(
            json.dumps(
                {
                    "pair_count": paired_count,
                    "minimum_required": resolved.v0_9_phase2.minimum_identifiable_pairs,
                    "paired_dynamic_gain": paired_gain,
                    "status": paired_gate.status.value,
                    "pairs": paired_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    attribution_summary = {
        "status": "COMPLETE" if attribution_rows else "INCOMPLETE",
        "representation_blocked": representation_blocked,
        "representation_gate": representation_gate.to_dict(),
        "metrics": {
            name: {
                field: sum(float(row[field]) for row in attribution_rows if row["metric"] == name)
                / max(sum(row["metric"] == name for row in attribution_rows), 1)
                for field in (
                    "data_floor",
                    "reconstruction",
                    "nominal",
                    "adaptive",
                    "representation_increment",
                    "nominal_dynamics_increment",
                    "adaptive_dynamics_increment",
                )
            }
            for name in sorted({str(row["metric"]) for row in attribution_rows})
        },
    }
    (evaluation / "error_attribution.json").write_text(
        json.dumps(attribution_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (evaluation / "v0_9_scientific_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if diagnostic is not None:
        torch.save(diagnostic, evaluation / "diagnostic_series.pt")
    return decision
