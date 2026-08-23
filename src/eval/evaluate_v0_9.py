"""Locked V0.9 one-step, teacher-free rollout, operator, and physical evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from jka_model.adaptive import (
    AdaptiveWindowDataset,
    adaptive_latent_rollout,
    latent_prediction_metrics,
    load_adaptive_cache,
    load_adaptive_checkpoint,
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


@torch.no_grad()
def _mean_training_delta(
    model: torch.nn.Module,
    dataset: AdaptiveWindowDataset,
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
        condition = _normalized_condition(raw["condition"], payload, mode, device)
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
    static_delta = _mean_training_delta(model, train_dataset, payload, mode, selected).to(selected)
    nominal_a = cache.nominal_generator.to(selected)
    one_step_rows: list[dict[str, Any]] = []
    all_nominal: list[torch.Tensor] = []
    all_remaining: list[torch.Tensor] = []
    for raw in DataLoader(test_dataset, batch_size=512, shuffle=False):
        history_z = raw["history_z"].to(selected).float()
        history_dts = raw["history_dts"].to(selected).float()
        next_dt = raw["next_dt"].to(selected).float()
        parameters = raw["context_parameters"].to(selected).float()
        truth = raw["target_next"].to(selected).float()
        condition = _normalized_condition(raw["condition"], payload, mode, selected)
        prediction, context, eta, delta, adapted_a = model(
            history_z, history_dts, next_dt, parameters, condition
        )
        gate = model.operator_adapter.adaptation_gate(context, condition)
        nominal_transition = torch.linalg.matrix_exp(
            nominal_a.unsqueeze(0) * next_dt.reshape(-1, 1, 1)
        )
        nominal = torch.einsum("bij,bj->bi", nominal_transition, history_z[:, -1])
        static_transition = torch.linalg.matrix_exp(
            (nominal_a + static_delta).unsqueeze(0) * next_dt.reshape(-1, 1, 1)
        )
        static = torch.einsum("bij,bj->bi", static_transition, history_z[:, -1])
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
            one_step_rows.append(
                {
                    "trajectory_id": raw["trajectory_id"][index],
                    "target_index": int(raw["target_index"][index]),
                    "relative_transition_index": int(raw["relative_transition_index"][index]),
                    "schedule_type": raw["schedule_type"][index],
                    **dynamic_metrics,
                    "static_latent_rmse": static_rmse,
                    "static_relative_gain": 1.0
                    - dynamic_metrics["latent_rmse"] / max(static_rmse, 1e-12),
                    "operator_burden": float(burdens[index]),
                    "symmetric_abscissa_proxy": float(proxy[index]),
                    "eta_norm": float(eta[index].norm()),
                    "trust_gate": float(gate[index, 0]),
                }
            )
    gamma = float(
        operator_explained_fraction(torch.cat(all_nominal), torch.cat(all_remaining))
    )

    shuffled_gain: float | None = None
    if history > 1:
        shuffled = AdaptiveWindowDataset(
            cache, "test", history, shuffle_older_history=True, shuffle_seed=7919
        )
        errors: list[float] = []
        for raw in DataLoader(shuffled, batch_size=512, shuffle=False):
            condition = _normalized_condition(raw["condition"], payload, mode, selected)
            prediction, *_ = model(
                raw["history_z"].to(selected).float(),
                raw["history_dts"].to(selected).float(),
                raw["next_dt"].to(selected).float(),
                raw["context_parameters"].to(selected).float(),
                condition,
            )
            errors.append(
                float(
                    (prediction - raw["target_next"].to(selected).float())
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
                conditions = (
                    trajectory.conditions[start : start + horizon].to(selected) - condition_mean
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
            conditions = (
                trajectory.conditions[start : start + longest].to(selected) - condition_mean
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
                trajectory.latents[start + 1 : start + longest + 1]
                .to(selected)
                .unsqueeze(0)
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
                and sum(gains) / len(gains)
                >= resolved.v0_9_evaluation.material_relative_gain
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
    one_step_gain = sum(float(row["relative_gain"]) for row in one_step_rows) / len(one_step_rows)
    dynamic_over_static = sum(float(row["static_relative_gain"]) for row in one_step_rows) / len(
        one_step_rows
    )
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
    controls_gate = aggregate_gate_results(
        "dynamic_controls",
        [dynamic_gate, history_gate],
        required_pass_fraction=1.0,
        minimum_count=2,
    )
    controls_pass = controls_gate.passed
    scientific_gates = {
        result.name: result.to_dict()
        for result in (
            one_step_gate,
            operator_gate,
            dynamic_gate,
            history_gate,
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
        "schema_version": 2,
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
        "history_over_shuffled_gain": shuffled_gain,
        "operator_explained_status": operator_gate.status.value,
        "dynamic_over_static_status": dynamic_gate.status.value,
        "controls_status": "PASS" if controls_pass else "FAIL",
        "closed_loop_by_horizon": horizon_summary,
        "all_horizons_status": "PASS" if all_horizons_pass else "FAIL",
        "longest_horizon_status": "PASS" if longest_pass else "FAIL",
        "operator_burden_status": "PASS" if burden_pass else "FAIL",
        "long_rollout_stability": "PASS" if all_horizons_pass and burden_pass else "FAIL",
        "physics_status": "PASS" if physics_pass else "FAIL",
        "observable_status": observable_gate.status.value,
        "representation_physical_floor_status": representation_gate.status.value,
        "phase1_diagnosis": "REPRESENTATION_BLOCKED"
        if representation_blocked
        else "OPERATOR_OPTIMIZATION_IDENTIFIABLE",
        "scientific_gates": scientific_gates,
        "adaptive_koopman": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "scientific_joint_pass": supported,
        "claims": {
            "backbone_frozen": True,
            "context_frozen": True,
            "A0_frozen": True,
            "additive_residual_enabled": False,
            "persistent_z_R_present": False,
            "phase1_error_attribution_complete": bool(attribution_rows),
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
