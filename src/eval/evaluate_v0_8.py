"""Canonical V0.8 teacher-forced, ablation, rollout, and physical evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from jka_model.config import ProjectConfig, load_config
from jka_model.context import (
    ContextWindowDataset,
    build_dynamic_context_model,
    context_corrected_latent_rollout,
    context_diagnostics,
    context_prediction_metrics,
)
from jka_model.context.checkpoint import load_context_checkpoint
from jka_model.data import (
    ChannelStandardizer,
    SplitManifest,
    cylinder_force_coefficients,
    select_split,
    shedding_frequency,
    velocity_vorticity_divergence,
)
from jka_model.problems import create_problem_adapter
from jka_model.residual import load_residual_cache
from train.train_v0_7 import load_frozen_v0_6_backbone

MIN_ADEQUACY_R2 = 0.0
MIN_ADEQUACY_CORRELATION = 0.5


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "NO_RECORDS"}]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_closed_loop_horizons(
    rows: list[dict[str, Any]],
    horizons: tuple[int, ...],
    *,
    material_relative_gain: float,
) -> dict[str, dict[str, float | bool | int]]:
    """Summarize each rollout horizon without letting short horizons hide long drift."""
    summary: dict[str, dict[str, float | bool | int]] = {}
    for horizon in horizons:
        selected = [float(row["latent_rmse"]) for row in rows if int(row["horizon"]) == horizon]
        baseline = [
            float(row["koopman_only_latent_rmse"]) for row in rows if int(row["horizon"]) == horizon
        ]
        if not selected or len(selected) != len(baseline):
            raise ValueError(f"closed-loop evaluation lacks paired horizon {horizon}")
        selected_mean = sum(selected) / len(selected)
        baseline_mean = sum(baseline) / len(baseline)
        relative_gain = 1.0 - selected_mean / max(baseline_mean, 1e-12)
        summary[str(horizon)] = {
            "trajectory_count": len(selected),
            "context_latent_rmse_mean": selected_mean,
            "koopman_only_latent_rmse_mean": baseline_mean,
            "relative_gain": relative_gain,
            "pass": relative_gain >= material_relative_gain,
        }
    return summary


def assess_context_acceptance(
    *,
    residual_route: str,
    context_gain: float,
    history_gain: float | None,
    context_effective_rank: float,
    context_collapsed: bool,
    adequacy_r2: float,
    adequacy_correlation: float,
    material_relative_gain: float,
    min_context_effective_rank: float,
    min_adequacy_r2: float,
    min_adequacy_correlation: float,
    burden_pass: bool,
) -> dict[str, bool]:
    """Apply the configured rank, R3-history, and adequacy acceptance contracts."""
    rank_pass = bool(not context_collapsed and context_effective_rank >= min_context_effective_rank)
    history_pass = bool(
        residual_route != "R3"
        or (history_gain is not None and history_gain >= material_relative_gain)
    )
    adequacy_pass = bool(
        adequacy_r2 >= min_adequacy_r2 and adequacy_correlation >= min_adequacy_correlation
    )
    context_supported = bool(
        context_gain >= material_relative_gain and rank_pass and history_pass and burden_pass
    )
    return {
        "rank_pass": rank_pass,
        "history_pass": history_pass,
        "adequacy_pass": adequacy_pass,
        "context_supported": context_supported,
    }


def _move(batch: dict[str, Any], device: torch.device, include_parameters: bool):
    parameters = batch["parameters"].to(device=device, dtype=torch.float32)
    if not include_parameters:
        parameters = parameters[:, :0]
    return (
        batch["history_z"].to(device=device, dtype=torch.float32),
        batch["history_dts"].to(device=device, dtype=torch.float32),
        batch["next_dt"].to(device=device, dtype=torch.float32),
        parameters,
        batch["target_residual"].to(device=device, dtype=torch.float32),
        batch["target_adequacy"].to(device=device, dtype=torch.float32),
    )


@torch.no_grad()
def _prediction_bundle(
    model: torch.nn.Module,
    dataset: ContextWindowDataset,
    device: torch.device,
    include_parameters: bool,
    residual_scale: torch.Tensor,
    adequacy_scale: torch.Tensor,
    *,
    ablate_context: bool = False,
) -> tuple[dict[str, float], torch.Tensor, dict[str, torch.Tensor]]:
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    residual_prediction: list[torch.Tensor] = []
    residual_target: list[torch.Tensor] = []
    adequacy_prediction: list[torch.Tensor] = []
    adequacy_target: list[torch.Tensor] = []
    contexts: list[torch.Tensor] = []
    for raw in loader:
        history_z, history_dts, next_dt, parameters, target_r, target_q = _move(
            raw, device, include_parameters
        )
        context, prediction_r, prediction_q = model(
            history_z,
            history_dts,
            next_dt,
            parameters,
            ablate_context=ablate_context,
        )
        contexts.append(context.cpu())
        residual_prediction.append(prediction_r.cpu())
        residual_target.append(target_r.cpu())
        adequacy_prediction.append(prediction_q.cpu())
        adequacy_target.append(target_q.cpu())
    residual_prediction_tensor = torch.cat(residual_prediction)
    residual_target_tensor = torch.cat(residual_target)
    adequacy_prediction_tensor = torch.cat(adequacy_prediction)
    adequacy_target_tensor = torch.cat(adequacy_target)
    contexts_tensor = torch.cat(contexts)
    return (
        context_prediction_metrics(
            residual_prediction_tensor,
            residual_target_tensor,
            adequacy_prediction_tensor,
            adequacy_target_tensor,
            residual_scale,
            adequacy_scale,
        ),
        contexts_tensor,
        {
            "residual_prediction": residual_prediction_tensor,
            "residual_target": residual_target_tensor,
            "adequacy_prediction": adequacy_prediction_tensor,
            "adequacy_target": adequacy_target_tensor,
            "contexts": contexts_tensor,
        },
    )


@torch.no_grad()
def evaluate_v0_8(
    config: ProjectConfig | str | Path,
    *,
    checkpoint: str | Path,
    backbone_checkpoint: str | Path,
    residual_cache: str | Path,
    output_dir: str | Path,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    if resolved.v0_8_context is None or resolved.v0_8_evaluation is None:
        raise ValueError("evaluate_v0_8 requires V0.8 context/evaluation sections")
    if resolved.cylinder_wake_2d is None:
        raise ValueError("evaluate_v0_8 requires the cylinder-wake problem config")
    payload = load_context_checkpoint(checkpoint)
    if payload["config_hash"] != resolved.stable_hash:
        raise ValueError("V0.8 evaluation config/checkpoint mismatch")
    cache = load_residual_cache(residual_cache)
    if payload["residual_cache_fingerprint"] != cache.fingerprint:
        raise ValueError("V0.8 evaluation cache/checkpoint mismatch")
    selected = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
    )
    family = str(payload["context_family"])
    history = int(payload["history_length_steps"])
    parameter_dim = cache.parameter_dim if resolved.v0_8_context.include_parameters else 0
    model = build_dynamic_context_model(
        resolved.v0_8_context,
        family=family,
        latent_dim=cache.latent_dim,
        parameter_dim=parameter_dim,
        history=history,
    ).to(selected)
    model.load_state_dict(payload["context_state"], strict=True)
    model.requires_grad_(False)
    model.eval()
    residual_scale = torch.as_tensor(payload["residual_training_scale"]).float()
    adequacy_scale = torch.as_tensor(payload["adequacy_training_scale"]).float()
    validation_dataset = ContextWindowDataset(cache, "validation", history)
    test_dataset = ContextWindowDataset(cache, "test", history)
    validation_metrics, _, _ = _prediction_bundle(
        model,
        validation_dataset,
        selected,
        resolved.v0_8_context.include_parameters,
        residual_scale,
        adequacy_scale,
    )
    test_metrics, contexts, test_series = _prediction_bundle(
        model,
        test_dataset,
        selected,
        resolved.v0_8_context.include_parameters,
        residual_scale,
        adequacy_scale,
    )
    ablated_metrics, _, _ = _prediction_bundle(
        model,
        test_dataset,
        selected,
        resolved.v0_8_context.include_parameters,
        residual_scale,
        adequacy_scale,
        ablate_context=True,
    )
    shuffled_metrics: dict[str, float] | None = None
    if family in {"attention", "history_mlp"} and history > 1:
        shuffled_metrics, _, _ = _prediction_bundle(
            model,
            ContextWindowDataset(
                cache,
                "test",
                history,
                shuffle_older_history=True,
                shuffle_seed=7919,
            ),
            selected,
            resolved.v0_8_context.include_parameters,
            residual_scale,
            adequacy_scale,
        )
    context_stats = context_diagnostics(contexts)
    shell, _ = load_frozen_v0_6_backbone(backbone_checkpoint, resolved, selected)
    shell.requires_grad_(False)
    shell.eval()
    normalizer = ChannelStandardizer(eps=resolved.data.normalization.eps)
    normalizer.load_state_dict(cache.normalizer_state)
    adapter = create_problem_adapter(resolved)
    records = adapter.build_dataset(seed=resolved.training.seed)
    manifest = SplitManifest.from_dict(cache.split_manifest)
    test_records = select_split(records, manifest, "test")
    record_lookup = {record.trajectory_id: record for record in test_records}
    cache_lookup = {item.trajectory_id: item for item in cache.select("test")}
    closed_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    representative: dict[str, Any] | None = None
    for horizon in resolved.v0_8_evaluation.rollout_horizons:
        for trajectory_id, cached in cache_lookup.items():
            if cached.dts.numel() < history - 1 + horizon:
                continue
            start = history - 1
            parameters = cached.parameters.to(selected).unsqueeze(0)
            if not resolved.v0_8_context.include_parameters:
                parameters = parameters[:, :0]
            prediction, base, correction = context_corrected_latent_rollout(
                model,
                shell.koopman_core,
                cached.latents[:history].to(selected).unsqueeze(0),
                cached.dts[: history - 1].to(selected).unsqueeze(0),
                cached.dts[start : start + horizon].to(selected).unsqueeze(0),
                parameters,
            )
            truth = cached.latents[start : start + horizon + 1].to(selected).unsqueeze(0)
            latent_rmse = float((prediction[:, 1:] - truth[:, 1:]).square().mean().sqrt())
            koopman_only = shell.koopman_core.rollout(
                cached.latents[start : start + 1].to(selected),
                cached.dts[start : start + horizon].to(selected).unsqueeze(0),
            )
            koopman_only_rmse = float((koopman_only[:, 1:] - truth[:, 1:]).square().mean().sqrt())
            increment = base - prediction[:, :-1]
            burden_curve = correction.norm(dim=-1) / (
                correction.norm(dim=-1) + increment.norm(dim=-1) + 1e-12
            )
            closed_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "horizon": horizon,
                    "latent_rmse": latent_rmse,
                    "koopman_only_latent_rmse": koopman_only_rmse,
                    "latent_relative_gain": 1.0 - latent_rmse / max(koopman_only_rmse, 1e-12),
                    "closure_burden_mean": float(burden_curve.mean()),
                    "closure_burden_max": float(burden_curve.max()),
                }
            )
            predicted_raw = normalizer.inverse_transform(shell.decode(prediction[:, 1:]))[0]
            koopman_raw = normalizer.inverse_transform(shell.decode(koopman_only[:, 1:]))[0]
            record = record_lookup[trajectory_id]
            truth_raw = record.states_raw[start + 1 : start + horizon + 1].to(selected)
            true_vorticity, _ = velocity_vorticity_divergence(truth_raw, resolved.cylinder_wake_2d)
            true_drag, true_lift = cylinder_force_coefficients(truth_raw, resolved.cylinder_wake_2d)
            true_frequency = shedding_frequency(true_lift, resolved.cylinder_wake_2d.snapshot_dt)
            for model_name, physical_prediction in (
                ("context_residual", predicted_raw),
                ("koopman_only", koopman_raw),
            ):
                velocity_error = physical_prediction[:, :2] - truth_raw[:, :2]
                predicted_vorticity, predicted_divergence = velocity_vorticity_divergence(
                    physical_prediction, resolved.cylinder_wake_2d
                )
                predicted_drag, predicted_lift = cylinder_force_coefficients(
                    physical_prediction, resolved.cylinder_wake_2d
                )
                assert record.valid_mask is not None
                solid = ~record.valid_mask.to(selected)
                no_slip_mse = float(physical_prediction[:, :2, solid].square().mean())
                physical_rows.append(
                    {
                        "model": model_name,
                        "trajectory_id": trajectory_id,
                        "horizon": horizon,
                        "velocity_relative_l2": float(
                            velocity_error.norm() / truth_raw[:, :2].norm().clamp_min(1e-12)
                        ),
                        "vorticity_relative_l2": float(
                            (predicted_vorticity - true_vorticity).norm()
                            / true_vorticity.norm().clamp_min(1e-12)
                        ),
                        "divergence_rms": float(predicted_divergence.square().mean().sqrt()),
                        "lift_rmse": float((predicted_lift - true_lift).square().mean().sqrt()),
                        "drag_rmse": float((predicted_drag - true_drag).square().mean().sqrt()),
                        "shedding_frequency_true": true_frequency,
                        "shedding_frequency_predicted": shedding_frequency(
                            predicted_lift, resolved.cylinder_wake_2d.snapshot_dt
                        ),
                        "boundary_no_slip_mse": no_slip_mse,
                    }
                )
                if (
                    representative is None
                    and model_name == "context_residual"
                    and horizon == max(resolved.v0_8_evaluation.rollout_horizons)
                ):
                    representative = {
                        "time": torch.arange(horizon) * resolved.cylinder_wake_2d.snapshot_dt,
                        "true_vorticity": true_vorticity.cpu(),
                        "predicted_vorticity": predicted_vorticity.cpu(),
                        "true_lift": true_lift.cpu(),
                        "predicted_lift": predicted_lift.cpu(),
                        "closed_loop_error": (prediction[:, 1:] - truth[:, 1:])
                        .square()
                        .mean(dim=-1)
                        .sqrt()[0]
                        .cpu(),
                        "koopman_only_error": (koopman_only[:, 1:] - truth[:, 1:])
                        .square()
                        .mean(dim=-1)
                        .sqrt()[0]
                        .cpu(),
                        "closure_burden": burden_curve[0].cpu(),
                    }
    context_gain = 1.0 - test_metrics["residual_standardized_mse"] / max(
        ablated_metrics["residual_standardized_mse"], 1e-12
    )
    history_gain = None
    if shuffled_metrics is not None:
        history_gain = 1.0 - test_metrics["residual_standardized_mse"] / max(
            shuffled_metrics["residual_standardized_mse"], 1e-12
        )
    horizon_summary = summarize_closed_loop_horizons(
        closed_rows,
        resolved.v0_8_evaluation.rollout_horizons,
        material_relative_gain=resolved.v0_8_evaluation.material_relative_gain,
    )
    longest_horizon = max(resolved.v0_8_evaluation.rollout_horizons)
    all_horizons_pass = all(bool(item["pass"]) for item in horizon_summary.values())
    longest_horizon_pass = bool(horizon_summary[str(longest_horizon)]["pass"])
    burden_pass = all(
        row["closure_burden_mean"] <= resolved.v0_8_evaluation.max_closure_burden
        for row in closed_rows
    )
    selected_physics = [row for row in physical_rows if row["model"] == "context_residual"]
    koopman_physics = [row for row in physical_rows if row["model"] == "koopman_only"]
    paired_noninferior: list[bool] = []
    long_frequency_noninferior: list[bool] = []
    relative_fields = (
        "velocity_relative_l2",
        "vorticity_relative_l2",
        "lift_rmse",
        "drag_rmse",
    )
    for selected_row, baseline_row in zip(selected_physics, koopman_physics, strict=True):
        paired_noninferior.append(
            all(
                float(selected_row[field])
                <= float(baseline_row[field])
                * (1.0 + resolved.v0_8_evaluation.max_physics_degradation)
                + 1e-8
                for field in relative_fields
            )
            and float(selected_row["divergence_rms"])
            <= resolved.v0_8_evaluation.max_divergence_mse**0.5
            and float(selected_row["boundary_no_slip_mse"])
            <= resolved.v0_8_evaluation.max_boundary_mse
        )
        if int(selected_row["horizon"]) == longest_horizon:
            true_frequency = float(selected_row["shedding_frequency_true"])
            selected_frequency_error = abs(
                float(selected_row["shedding_frequency_predicted"]) - true_frequency
            )
            baseline_frequency_error = abs(
                float(baseline_row["shedding_frequency_predicted"]) - true_frequency
            )
            long_frequency_noninferior.append(
                selected_frequency_error
                <= baseline_frequency_error
                * (1.0 + resolved.v0_8_evaluation.max_physics_degradation)
                + 1e-8
            )
    physics_noninferiority_fraction = sum(paired_noninferior) / max(len(paired_noninferior), 1)
    long_frequency_noninferiority_fraction = sum(long_frequency_noninferior) / max(
        len(long_frequency_noninferior), 1
    )
    physics_pass = bool(
        burden_pass
        and selected_physics
        and physics_noninferiority_fraction >= resolved.v0_8_evaluation.seed_consistency_fraction
        and long_frequency_noninferior
        and long_frequency_noninferiority_fraction
        >= resolved.v0_8_evaluation.seed_consistency_fraction
    )
    positive_rollout_fraction = sum(
        row["latent_relative_gain"] >= resolved.v0_8_evaluation.material_relative_gain
        for row in closed_rows
    ) / max(len(closed_rows), 1)
    context_checks = assess_context_acceptance(
        residual_route=str(payload["residual_route"]),
        context_gain=context_gain,
        history_gain=history_gain,
        context_effective_rank=float(context_stats["effective_rank"]),
        context_collapsed=bool(context_stats["collapsed"]),
        adequacy_r2=test_metrics["adequacy_r2"],
        adequacy_correlation=test_metrics["adequacy_correlation"],
        material_relative_gain=resolved.v0_8_evaluation.material_relative_gain,
        min_context_effective_rank=resolved.v0_8_evaluation.min_context_effective_rank,
        min_adequacy_r2=MIN_ADEQUACY_R2,
        min_adequacy_correlation=MIN_ADEQUACY_CORRELATION,
        burden_pass=burden_pass,
    )
    context_rank_pass = context_checks["rank_pass"]
    adequacy_pass = context_checks["adequacy_pass"]
    context_supported = context_checks["context_supported"]
    decision = {
        "schema_version": 1,
        "backbone_seed": int(payload["backbone_seed"]),
        "context_init_seed": int(payload["context_init_seed"]),
        "physical_problem": "cylinder_wake_2d",
        "backbone_status": "REQUIRES_FORMAL_ACCEPTANCE_REPORT",
        "v0_7_route_on_new_problem": payload["residual_route"],
        "context_family": family.upper(),
        "residual_prediction": "SUPPORTED" if context_supported else "WEAK",
        "history_value": (
            "SUPPORTED"
            if history_gain is not None
            and history_gain >= resolved.v0_8_evaluation.material_relative_gain
            else ("NOT_SUPPORTED" if history_gain is not None else "N/A")
        ),
        "context_rank_status": "PASS" if context_rank_pass else "FAIL",
        "koopman_adequacy": "CALIBRATED" if adequacy_pass else "UNCALIBRATED",
        "dynamic_context": "SUPPORTED" if context_supported else "NOT_SUPPORTED",
        "closed_loop_utility": "POSITIVE" if all_horizons_pass else "NEUTRAL",
        "longest_horizon_utility": "POSITIVE" if longest_horizon_pass else "NEUTRAL",
        "physics_status": "PASS" if physics_pass else "FAIL",
        "v0_9_operator_adaptation_readiness": "INCONCLUSIVE",
        "validation_metrics": validation_metrics,
        "test_locked_confirmation": test_metrics,
        "context_ablation": ablated_metrics,
        "shuffled_history": shuffled_metrics,
        "context_diagnostics": context_stats,
        "context_ablation_gain": context_gain,
        "history_over_shuffled_gain": history_gain,
        "closed_loop_by_horizon": horizon_summary,
        "closure_burden_pass": burden_pass,
        "positive_rollout_fraction": positive_rollout_fraction,
        "physics_noninferiority_fraction": physics_noninferiority_fraction,
        "long_frequency_noninferiority_fraction": long_frequency_noninferiority_fraction,
        "evaluation_thresholds": {
            "material_relative_gain": resolved.v0_8_evaluation.material_relative_gain,
            "min_context_effective_rank": resolved.v0_8_evaluation.min_context_effective_rank,
            "min_adequacy_r2": MIN_ADEQUACY_R2,
            "min_adequacy_correlation": MIN_ADEQUACY_CORRELATION,
            "seed_consistency_fraction": resolved.v0_8_evaluation.seed_consistency_fraction,
        },
    }
    output = Path(output_dir)
    evaluation = output / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    _write_csv(
        evaluation / "context_model_comparison.csv",
        [
            {"model": "selected", **test_metrics},
            {"model": "no_context", **ablated_metrics},
            *(
                []
                if shuffled_metrics is None
                else [{"model": "shuffled_history", **shuffled_metrics}]
            ),
        ],
    )
    _write_csv(
        evaluation / "context_ablation.csv",
        [
            {"ablation": "full", **test_metrics},
            {"ablation": "context_zero", **ablated_metrics},
        ],
    )
    _write_csv(evaluation / "closed_loop_metrics.csv", closed_rows)
    _write_csv(evaluation / "physical_metrics.csv", physical_rows)
    diagnostic_series = {
        **test_series,
        "representative_rollout": representative,
    }
    torch.save(diagnostic_series, evaluation / "diagnostic_series.pt")
    (evaluation / "v0_8_scientific_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return decision
