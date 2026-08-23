"""Canonical V0.9 operator-only training over a frozen V0.8 context."""

from __future__ import annotations

import csv
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from jka_model.adaptive import (
    AdaptiveKoopmanModel,
    AdaptiveRolloutDataset,
    AdaptiveWindowDataset,
    FactorizedAdaptiveOperator,
    FrozenDecoderObservables,
    LowRankAdaptiveOperator,
    adaptive_stabilization_objective,
    adaptive_training_scales,
    curriculum_state,
    load_adaptive_cache,
    load_adaptive_checkpoint,
    operator_burden,
    phase2_condition_scales,
    save_adaptive_checkpoint,
    symmetric_abscissa_proxy,
)
from jka_model.config import ProjectConfig, load_config, save_config
from jka_model.constants import ARCHITECTURE_REVISION, CHECKPOINT_SCHEMA_VERSION, PROJECT_VERSION
from jka_model.context import build_dynamic_context_model
from jka_model.context.checkpoint import load_context_checkpoint
from jka_model.observables import RobustObservableScaleState
from jka_model.optimization import (
    InequalityAugmentedLagrangian,
    gradient_cosine_matrix,
    pcgrad_backward,
)
from jka_model.residual.cache import file_sha256
from jka_model.training import (
    TrainStage,
    assert_optimizer_matches_trainable_params,
    configure_train_stage,
)
from jka_model.utils import (
    RNGState,
    capture_rng_state,
    get_git_commit,
    restore_rng_state,
    set_global_seed,
)


@dataclass(frozen=True, slots=True)
class V09TrainingResult:
    run_dir: Path
    latest_checkpoint: Path
    best_checkpoint: Path
    condition_mode: str
    rank: int
    start_epoch: int
    completed_epochs: int
    validation_metrics: dict[str, float]


def _require_v0_9(config: ProjectConfig) -> None:
    required = (
        config.koopman,
        config.v0_8_context,
        config.v0_9_condition,
        config.v0_9_adaptive,
        config.v0_9_training,
        config.v0_9_evaluation,
    )
    if any(section is None for section in required):
        raise ValueError("train_v0_9 requires the complete V0.8 handoff and V0.9 contract")
    if config.training.stage is not TrainStage.ADAPTIVE:
        raise ValueError("V0.9 training.stage must be adaptive")


def _build_model(
    config: ProjectConfig,
    cache: Any,
    context_payload: dict[str, Any],
    device: torch.device,
) -> AdaptiveKoopmanModel:
    assert config.v0_8_context and config.v0_9_adaptive
    family = str(context_payload["context_family"])
    history = int(context_payload["history_length_steps"])
    dynamic = build_dynamic_context_model(
        config.v0_8_context,
        family=family,
        latent_dim=cache.latent_dim,
        parameter_dim=cache.context_parameter_dim,
        history=history,
    )
    dynamic.load_state_dict(context_payload["best_context_state"], strict=True)
    if config.v0_9_phase2 is not None and config.v0_9_phase2.enabled:
        adapter = FactorizedAdaptiveOperator(
            cache.nominal_generator,
            config.v0_8_context.context_dim,
            config.v0_9_adaptive,
            config.v0_9_phase2,
        )
    else:
        adapter = LowRankAdaptiveOperator(
            cache.nominal_generator,
            config.v0_8_context.context_dim,
            config.v0_9_adaptive,
        )
    model = AdaptiveKoopmanModel(dynamic.context_encoder, adapter).to(device)
    configure_train_stage(model, TrainStage.ADAPTIVE)
    return model


def _move(raw: dict[str, Any], device: torch.device) -> dict[str, Any]:
    tensor_names = (
        "history_z",
        "history_dts",
        "next_dt",
        "context_parameters",
        "condition",
        "condition_target",
        "target_next",
        "previous_history_z",
        "previous_history_dts",
        "previous_next_dt",
        "previous_condition",
        "previous_condition_target",
    )
    result = dict(raw)
    for name in tensor_names:
        result[name] = raw[name].to(device=device, dtype=torch.float32)
    result["smoothness_eligible"] = raw["smoothness_eligible"].to(device=device)
    return result


def _move_rollout(raw: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(raw)
    for name in (
        "history_z",
        "history_dts",
        "future_dts",
        "future_conditions",
        "future_condition_targets",
        "target_latents",
        "context_parameters",
    ):
        result[name] = raw[name].to(device=device, dtype=torch.float32)
    return result


def _uses_stabilized_objective(config: ProjectConfig) -> bool:
    assert config.v0_9_adaptive and config.v0_9_training
    training = config.v0_9_training
    adaptive = config.v0_9_adaptive
    return bool(
        (config.v0_9_phase2 is not None and config.v0_9_phase2.enabled)
        or
        training.lambda_rollout > 0
        or training.lambda_propagator_growth > 0
        or training.lambda_physics > 0
        or adaptive.bounded_coordinates
        or adaptive.trust_gate
    )


def _condition(
    value: torch.Tensor,
    *,
    mode: str,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor | None:
    return (value - mean) / std if mode == "known" else None


def _loss_bundle(
    model: AdaptiveKoopmanModel,
    batch: dict[str, Any],
    residual_scale: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    config: ProjectConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert config.v0_9_adaptive and config.v0_9_training
    condition = _condition(
        batch["condition"],
        mode=config.v0_9_adaptive.condition_mode,
        mean=condition_mean,
        std=condition_std,
    )
    prediction, _, eta, delta, adapted = model(
        batch["history_z"],
        batch["history_dts"],
        batch["next_dt"],
        batch["context_parameters"],
        condition,
    )
    forecast = ((prediction - batch["target_next"]) / residual_scale).square().mean()
    burden = operator_burden(delta, model.operator_adapter.nominal_generator).square().mean()
    baseline_proxy = symmetric_abscissa_proxy(model.operator_adapter.nominal_generator).detach()
    stability = torch.relu(symmetric_abscissa_proxy(adapted) - baseline_proxy).square().mean()
    smoothness = prediction.new_zeros(())
    eligible = batch["smoothness_eligible"].bool()
    if bool(eligible.any()):
        previous_condition = _condition(
            batch["previous_condition"],
            mode=config.v0_9_adaptive.condition_mode,
            mean=condition_mean,
            std=condition_std,
        )
        with torch.no_grad():
            previous_context = model.context_encoder(
                batch["previous_history_z"],
                batch["previous_history_dts"],
                batch["previous_next_dt"],
                batch["context_parameters"],
            )
        previous_eta = model.operator_adapter.coordinates(previous_context, previous_condition)
        smoothness = (
            (eta[eligible] - previous_eta[eligible]).square().sum(dim=-1)
            / batch["next_dt"][eligible, 0].clamp_min(1e-12)
        ).mean()
    orthogonality = model.operator_adapter.orthogonality_loss()
    training = config.v0_9_training
    total = (
        forecast
        + training.lambda_operator_burden * (burden + orthogonality)
        + training.lambda_smooth * smoothness
        + training.lambda_stability * stability
    )
    return total, {
        "forecast": forecast,
        "burden": burden,
        "smoothness": smoothness,
        "stability": stability,
        "orthogonality": orthogonality,
    }


def _stabilized_loss_bundle(
    model: AdaptiveKoopmanModel,
    batch: dict[str, Any],
    residual_scale: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    config: ProjectConfig,
    epoch: int,
    physical: FrozenDecoderObservables | None,
    augmented_lagrangian: InequalityAugmentedLagrangian | None = None,
    *,
    validation: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert config.v0_9_adaptive and config.v0_9_training
    training = config.v0_9_training
    state = curriculum_state(training, epoch, validation=validation)
    smooth_mask = torch.tensor(
        ["abrupt" not in str(value) for value in batch["schedule_type"]],
        device=batch["history_z"].device,
        dtype=torch.bool,
    )
    objective = adaptive_stabilization_objective(
        model,
        batch,
        residual_scale,
        condition_mean,
        condition_std,
        training,
        config.v0_9_adaptive.condition_mode,
        state,
        smooth_mask,
        config.v0_9_phase2,
    )
    total = objective.total
    terms = dict(objective.terms)
    phase2 = config.v0_9_phase2
    progress = epoch / max(training.epochs - 1, 1)
    observer_warmup = bool(
        not validation
        and phase2 is not None
        and phase2.enabled
        and progress < phase2.observer_warmup_fraction
    )
    if observer_warmup:
        # First identify the latent physical condition.  The operator heads are
        # exactly zero-initialized, so this stage cannot alter A0 or invent a
        # dynamic innovation before Q(history) becomes meaningful.
        total = phase2.lambda_condition_observer * terms["condition_observer"]
        terms.update(
            {
                "phase2_observer_warmup": total.new_ones(()),
                "physics": total.new_zeros(()),
                "force_window": total.new_zeros(()),
                "observable_noninferiority": total.new_zeros(()),
                "augmented_lagrangian": total.new_zeros(()),
                "physics_scale": total.new_zeros(()),
            }
        )
        return total, terms
    terms["phase2_observer_warmup"] = total.new_zeros(())
    physics = total.new_zeros(())
    observable_noninferiority = total.new_zeros(())
    force_window = total.new_zeros(())
    augmented_penalty = total.new_zeros(())
    if state.physics_scale > 0:
        if physical is None:
            raise RuntimeError("enabled V0.9 physical objective lacks frozen decoder state")
        limit = min(training.physics_batch_size, objective.rollout["adapted"].shape[0])
        weight_sum = state.observable_normalizer
        if weight_sum <= 0:
            raise RuntimeError("enabled V0.9 observables require positive horizon weights")
        primary_observable = total.new_zeros(())
        constraint_values: dict[str, list[torch.Tensor]] = {
            "divergence": [],
            "boundary": [],
        }
        component_weights = dict(
            zip(
                training.observable_names,
                training.observable_component_weights,
                strict=True,
            )
        )
        for horizon, weight in zip(
            state.observable_horizons,
            state.observable_weights,
            strict=True,
        ):
            if weight == 0:
                continue
            target_raw, metadata = physical.target_batch(
                batch["trajectory_id"],
                batch["target_index"],
                horizon,
                limit,
            )
            adapted_result = physical.loss(
                objective.rollout["adapted"][:limit, horizon - 1],
                target_raw,
                metadata,
            )
            with torch.no_grad():
                nominal_result = physical.loss(
                    objective.rollout["nominal"][:limit, horizon - 1],
                    target_raw,
                    metadata,
                )
            physics = physics + weight * adapted_result.total
            component_excesses: list[torch.Tensor] = []
            for name, value in adapted_result.terms.items():
                baseline = nominal_result.terms[name].detach()
                terms[f"{name}_h{horizon}"] = value
                terms[f"{name}_nominal_h{horizon}"] = baseline
                floor = training.observable_noninferiority_floor
                allowed = (
                    baseline * (1.0 + training.observable_noninferiority_margin) + floor
                )
                component_excesses.append(
                    (torch.relu(value - allowed) / (baseline.abs() + floor)).square()
                )
                component = name.removeprefix("observable_")
                if training.phase1_enabled and component in {"divergence", "boundary"}:
                    constraint_values[component].append(
                        (value - allowed) / (baseline.abs() + floor)
                    )
                elif training.phase1_enabled and component not in {"lift", "drag"}:
                    primary_observable = primary_observable + weight * component_weights.get(
                        component, 0.0
                    ) * value
            if component_excesses:
                observable_noninferiority = observable_noninferiority + weight * torch.stack(
                    component_excesses
                ).mean()
            if training.phase1_enabled and (
                component_weights.get("lift", 0.0) > 0
                or component_weights.get("drag", 0.0) > 0
            ):
                steps = tuple(
                    sorted(
                        {
                            *range(
                                training.force_window_stride,
                                horizon + 1,
                                training.force_window_stride,
                            ),
                            horizon,
                        }
                    )
                )
                target_sequence, sequence_metadata = physical.target_sequence(
                    batch["trajectory_id"],
                    batch["target_index"],
                    steps,
                    limit,
                )
                indices = torch.tensor(
                    [step - 1 for step in steps],
                    device=objective.rollout["adapted"].device,
                    dtype=torch.long,
                )
                force_result = physical.force_window_loss(
                    objective.rollout["adapted"][:limit].index_select(1, indices),
                    target_sequence,
                    sequence_metadata,
                )
                force_weight = component_weights.get("lift", 0.0) + component_weights.get(
                    "drag", 0.0
                )
                force_window = force_window + weight * force_weight * force_result.total
                for name, value in force_result.terms.items():
                    terms[f"{name}_h{horizon}"] = value
        physics = physics / weight_sum
        observable_noninferiority = observable_noninferiority / weight_sum
        if training.phase1_enabled:
            primary_observable = (primary_observable + force_window) / weight_sum
            # Burden is an inequality constraint in Phase 1, while basis
            # orthogonality remains a structural regularizer.
            total = total - training.lambda_operator_burden * terms["burden"]
            total = total + training.lambda_physics * state.physics_scale * primary_observable
            constraints = {
                name: (
                    torch.stack(values).max()
                    if values
                    else total.new_zeros(())
                )
                for name, values in constraint_values.items()
            }
            constraints["burden"] = terms["burden_max"] - training.operator_burden_target
            for name, value in constraints.items():
                terms[f"constraint_{name}"] = value
            if augmented_lagrangian is None:
                raise RuntimeError("phase-1 training requires augmented-Lagrangian state")
            augmented_penalty = augmented_lagrangian.penalty(constraints)
            total = total + state.physics_scale * augmented_penalty
            terms["observable_primary"] = primary_observable
            terms["objective_prediction"] = (
                terms["forecast"] + training.lambda_rollout * terms["rollout"]
            )
            if phase2 is not None and phase2.enabled:
                terms["objective_prediction"] = (
                    terms["objective_prediction"]
                    + phase2.lambda_condition_observer
                    * terms["condition_observer"]
                )
            terms["objective_regularization"] = (
                training.lambda_operator_burden * terms["orthogonality"]
                + training.lambda_smooth * terms["smoothness"]
                + training.lambda_stability * terms["stability"]
                + training.lambda_propagator_growth * terms["propagator_growth"]
            )
            if phase2 is not None and phase2.enabled:
                terms["objective_regularization"] = (
                    terms["objective_regularization"]
                    + phase2.lambda_condition_centering
                    * terms["condition_centering"]
                    + phase2.lambda_basis_cross_orthogonality
                    * terms["basis_cross_orthogonality"]
                )
            terms["objective_observable"] = (
                training.lambda_physics * state.physics_scale * primary_observable
            )
            terms["objective_constraints"] = state.physics_scale * augmented_penalty
        else:
            observable_objective = (
                physics
                + training.lambda_observable_noninferiority * observable_noninferiority
            )
            total = total + training.lambda_physics * state.physics_scale * observable_objective
    terms["physics"] = physics
    terms["force_window"] = force_window / max(state.observable_normalizer, 1.0)
    terms["observable_noninferiority"] = observable_noninferiority
    terms["augmented_lagrangian"] = augmented_penalty
    terms["physics_scale"] = total.new_tensor(state.physics_scale)
    return total, terms


@torch.no_grad()
def _evaluate(
    model: AdaptiveKoopmanModel,
    loader: DataLoader,
    residual_scale: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    config: ProjectConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for raw in loader:
        batch = _move(raw, device)
        total, terms = _loss_bundle(
            model, batch, residual_scale, condition_mean, condition_std, config
        )
        batch_count = batch["target_next"].shape[0]
        values = {"total": total, **terms}
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_count
        count += batch_count
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def _evaluate_stabilized(
    model: AdaptiveKoopmanModel,
    loader: DataLoader,
    residual_scale: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    config: ProjectConfig,
    physical: FrozenDecoderObservables | None,
    augmented_lagrangian: InequalityAugmentedLagrangian | None,
    device: torch.device,
) -> dict[str, float]:
    assert config.v0_9_training
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    validation_epoch = config.v0_9_training.epochs - 1
    for raw in loader:
        batch = _move_rollout(raw, device)
        total, terms = _stabilized_loss_bundle(
            model,
            batch,
            residual_scale,
            condition_mean,
            condition_std,
            config,
            validation_epoch,
            physical,
            augmented_lagrangian,
            validation=True,
        )
        batch_count = batch["target_latents"].shape[0]
        for name, value in {"total": total, **terms}.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_count
        count += batch_count
    return {name: value / count for name, value in totals.items()}


def _validation_selection_score(metrics: dict[str, float]) -> float:
    """Multiplier-invariant score used for checkpoint and rank selection."""
    return float(
        metrics["total"]
        - metrics.get("augmented_lagrangian", 0.0)
        + sum(
            max(metrics.get(f"constraint_{name}", 0.0), 0.0) ** 2
            for name in ("divergence", "boundary", "burden")
        )
    )


def train_v0_9(
    config: ProjectConfig | str | Path,
    *,
    context_checkpoint: str | Path,
    adaptive_cache: str | Path,
    run_dir: str | Path,
    backbone_checkpoint: str | Path | None = None,
    physical_dataset: str | Path | None = None,
    device: str | torch.device | None = None,
    resume_from: str | Path | None = None,
) -> V09TrainingResult:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    _require_v0_9(resolved)
    assert resolved.v0_9_adaptive and resolved.v0_9_training and resolved.v0_8_context
    selected = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
    )
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    destination = Path(run_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("config", "metadata", "logs", "checkpoints", "evaluation", "reports"):
        (destination / name).mkdir()
    save_config(resolved, destination / "config" / "resolved_config.yaml")
    print(
        f"[V0.9][train:{resolved.v0_9_adaptive.condition_mode}] START "
        f"device={selected} rank={resolved.v0_9_adaptive.rank} "
        f"epochs={resolved.v0_9_training.epochs}",
        flush=True,
    )
    set_global_seed(
        resolved.v0_9_training.operator_initialization_seed,
        deterministic=resolved.training.deterministic,
    )
    cache = load_adaptive_cache(adaptive_cache)
    context_sha = file_sha256(context_checkpoint)
    if cache.context_checkpoint_sha256 != context_sha:
        raise ValueError("V0.9 cache/context checkpoint fingerprint mismatch")
    context_payload = load_context_checkpoint(context_checkpoint)
    model = _build_model(resolved, cache, context_payload, selected)
    stabilized = _uses_stabilized_objective(resolved)
    physical: FrozenDecoderObservables | None = None
    if resolved.v0_9_training.lambda_physics > 0:
        if backbone_checkpoint is None or physical_dataset is None:
            raise ValueError(
                "enabled V0.9 physical objective requires backbone_checkpoint and physical_dataset"
            )
        physical = FrozenDecoderObservables.from_artifacts(
            resolved,
            backbone_checkpoint=backbone_checkpoint,
            physical_dataset=physical_dataset,
            expected_backbone_sha256=cache.backbone_checkpoint_sha256,
            device=selected,
        )
    observable_scale_state: RobustObservableScaleState | None = None
    augmented_lagrangian: InequalityAugmentedLagrangian | None = None
    if resolved.v0_9_training.phase1_enabled:
        if physical is None:
            raise ValueError("phase-1 V0.9 training requires frozen decoder observables")
        train_trajectory_ids = tuple(item.trajectory_id for item in cache.select("train"))
        observable_scale_state = physical.fit_training_scales(
            train_trajectory_ids,
            split_fingerprint=cache.data_fingerprint,
        )
        augmented_lagrangian = InequalityAugmentedLagrangian(
            ("divergence", "boundary", "burden"),
            initial_penalty=(
                resolved.v0_9_training.augmented_lagrangian_initial_penalty
            ),
            penalty_growth=(
                resolved.v0_9_training.augmented_lagrangian_penalty_growth
            ),
            maximum_penalty=(
                resolved.v0_9_training.augmented_lagrangian_max_penalty
            ),
            improvement_ratio=(
                resolved.v0_9_training.augmented_lagrangian_improvement_ratio
            ),
            dual_step_size=(
                resolved.v0_9_training.augmented_lagrangian_dual_step_size
            ),
            maximum_multiplier=(
                resolved.v0_9_training.augmented_lagrangian_max_multiplier
            ),
        )
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=resolved.v0_9_training.learning_rate,
        weight_decay=resolved.v0_9_training.weight_decay,
    )
    assert_optimizer_matches_trainable_params(model, optimizer)
    scheduler = StepLR(optimizer, step_size=max(1, resolved.v0_9_training.epochs // 2), gamma=0.5)
    amp_enabled = selected.type == "cuda" and resolved.v0_9_training.precision != "fp32"
    amp_dtype = (
        torch.float16
        if resolved.v0_9_training.precision == "amp_fp16"
        else torch.bfloat16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype is torch.float16)
    history = int(context_payload["history_length_steps"])
    if stabilized:
        maximum_horizon = max(
            resolved.v0_9_training.rollout_horizons[-1],
            *resolved.v0_9_training.active_observable_horizons,
        )
        datasets = {
            split: AdaptiveRolloutDataset(
                cache,
                split,
                history,
                maximum_horizon,
                stride=resolved.v0_9_training.rollout_stride,
            )
            for split in ("train", "validation")
        }
    else:
        datasets = {
            split: AdaptiveWindowDataset(cache, split, history)
            for split in ("train", "validation")
        }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=(
                resolved.v0_9_training.rollout_batch_size
                if stabilized
                else resolved.v0_9_training.batch_size
            ),
            shuffle=split == "train",
            num_workers=0,
        )
        for split, dataset in datasets.items()
    }
    residual_scale_cpu, condition_mean_cpu, condition_std_cpu = adaptive_training_scales(cache)
    if resolved.v0_9_phase2 is not None and resolved.v0_9_phase2.enabled:
        condition_mean_cpu, condition_std_cpu = phase2_condition_scales(cache)
    residual_scale = residual_scale_cpu.to(selected)
    condition_mean = condition_mean_cpu.to(selected)
    condition_std = condition_std_cpu.to(selected)
    start_epoch = global_step = optimizer_update_step = 0
    best_score = float("inf")
    best_state: dict[str, Any] | None = None
    best_phase1_state: dict[str, Any] | None = None
    stale = 0
    if resume_from is not None:
        saved = load_adaptive_checkpoint(resume_from)
        if saved["config_hash"] != resolved.stable_hash:
            raise ValueError("V0.9 resume config mismatch")
        if saved["adaptive_cache_fingerprint"] != cache.fingerprint:
            raise ValueError("V0.9 resume cache mismatch")
        model.operator_adapter.load_state_dict(saved["adaptive_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        if saved["amp_scaler_state"] is not None:
            scaler.load_state_dict(saved["amp_scaler_state"])
        restore_rng_state(RNGState.from_checkpoint_dict(saved["rng_state"]))
        start_epoch = int(saved["epoch"])
        global_step = int(saved["global_step"])
        optimizer_update_step = int(saved["optimizer_update_step"])
        best_score = float(saved["best_validation_score"])
        best_state = dict(saved["best_adaptive_state"])
        stale = int(saved["epochs_without_improvement"])
        if resolved.v0_9_training.phase1_enabled:
            phase1_state = saved.get("phase1_state")
            if not isinstance(phase1_state, dict):
                raise ValueError("phase-1 V0.9 resume checkpoint lacks phase1_state")
            saved_scale = RobustObservableScaleState.from_dict(
                phase1_state["observable_scale_state"]
            )
            if observable_scale_state is None or (
                saved_scale.to_dict() != observable_scale_state.to_dict()
            ):
                raise ValueError("phase-1 V0.9 resume observable scales mismatch")
            assert physical is not None and augmented_lagrangian is not None
            physical.set_scale_state(saved_scale)
            augmented_lagrangian.load_state_dict(
                phase1_state["augmented_lagrangian_state"]
            )
            saved_best_phase1_state = phase1_state.get(
                "best_augmented_lagrangian_state",
                phase1_state["augmented_lagrangian_state"],
            )
            if not isinstance(saved_best_phase1_state, dict):
                raise ValueError("phase-1 V0.9 resume best dual state is invalid")
            best_phase1_state = dict(saved_best_phase1_state)
    latest = destination / "checkpoints" / "latest.pt"
    best = destination / "checkpoints" / "best_scientific_gate.pt"

    def checkpoint_payload(epoch: int) -> dict[str, Any]:
        if best_state is None:
            raise RuntimeError("cannot checkpoint before validation selects a state")
        assert resolved.v0_9_adaptive is not None
        assert resolved.v0_9_training is not None
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "architecture_revision": ARCHITECTURE_REVISION,
            "project_version": PROJECT_VERSION,
            "train_stage": TrainStage.ADAPTIVE.value,
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_update_step": optimizer_update_step,
            "condition_mode": resolved.v0_9_adaptive.condition_mode,
            "rank": resolved.v0_9_adaptive.rank,
            "flow_data_seed": resolved.training.seed,
            "backbone_seed": resolved.training.seed,
            "context_init_seed": int(context_payload["context_init_seed"]),
            "operator_init_seed": resolved.v0_9_training.operator_initialization_seed,
            "adaptive_state": model.operator_adapter.state_dict(),
            "best_adaptive_state": best_state,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "amp_scaler_state": scaler.state_dict() if amp_enabled else None,
            "rng_state": capture_rng_state().to_checkpoint_dict(),
            "config": resolved.to_dict(),
            "config_hash": resolved.stable_hash,
            "backbone_checkpoint_sha256": cache.backbone_checkpoint_sha256,
            "context_checkpoint_sha256": context_sha,
            "adaptive_cache_fingerprint": cache.fingerprint,
            "residual_training_scale": residual_scale_cpu,
            "condition_mean": condition_mean_cpu,
            "condition_std": condition_std_cpu,
            "best_validation_score": best_score,
            "epochs_without_improvement": stale,
            "phase1_state": None
            if not resolved.v0_9_training.phase1_enabled
            else {
                "observable_scale_state": observable_scale_state.to_dict()
                if observable_scale_state is not None
                else None,
                "augmented_lagrangian_state": augmented_lagrangian.state_dict()
                if augmented_lagrangian is not None
                else None,
                "best_augmented_lagrangian_state": best_phase1_state,
            },
            "git_commit": get_git_commit(Path.cwd()),
            "runtime": {
                "device": str(selected),
                "precision": resolved.v0_9_training.precision,
                "torch_version": torch.__version__,
            },
        }

    completed = start_epoch
    maturity_fraction = max(resolved.v0_9_training.rollout_start_fractions)
    if resolved.v0_9_training.lambda_physics > 0:
        physics_maturity = (
            resolved.v0_9_training.physics_start_fraction
            + resolved.v0_9_training.physics_ramp_duration_fraction
            if resolved.v0_9_training.physics_ramp_duration_fraction > 0
            else 1.0
        )
        maturity_fraction = max(maturity_fraction, physics_maturity)
    final_curriculum_start_epoch = math.ceil(
        maturity_fraction * max(resolved.v0_9_training.epochs - 1, 1)
    )
    log_path = destination / "logs" / "epoch_metrics.csv"
    gradient_log_path = destination / "logs" / "gradient_geometry.jsonl"
    gradient_records: list[dict[str, Any]] = []
    latest_constraint_diagnostics: dict[str, float] = {
        "constraint_max_violation": 0.0,
        **{
            f"multiplier_{name}": 0.0
            for name in ("divergence", "boundary", "burden")
        },
        **{
            f"penalty_{name}": 0.0
            for name in ("divergence", "boundary", "burden")
        },
    }
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "epoch",
            "active_horizons",
            "physics_scale",
            "train_total",
            "train_forecast",
            "train_rollout",
            "train_physics",
            "train_observable_noninferiority",
            "train_force_window",
            "train_propagator_growth",
            "train_condition_observer",
            "train_condition_centering",
            "train_basis_cross_orthogonality",
            "train_pcgrad_conflicts",
            "train_pcgrad_comparisons",
            "validation_total",
            "validation_selection_score",
            "validation_forecast",
            "validation_rollout",
            "validation_physics",
            "validation_observable_noninferiority",
            "validation_force_window",
            "validation_burden",
            "validation_burden_max",
            "validation_stability",
            "validation_propagator_growth",
            "validation_gate_mean",
            "validation_condition_observer",
            "validation_condition_centering",
            "validation_basis_cross_orthogonality",
            "constraint_max_violation",
            "multiplier_divergence",
            "multiplier_boundary",
            "multiplier_burden",
            "penalty_divergence",
            "penalty_boundary",
            "penalty_burden",
            *[
                f"validation_rollout_gain_h{horizon}"
                for horizon in resolved.v0_9_training.rollout_horizons
            ],
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(start_epoch, resolved.v0_9_training.epochs):
            model.train()
            state = curriculum_state(resolved.v0_9_training, epoch)
            if stabilized and epoch == final_curriculum_start_epoch:
                # Earlier scores use the same full validation contract, but the
                # optimizer has not yet seen the final horizon.  Give that stage
                # its own complete patience budget.
                stale = 0
            sums = {
                "total": 0.0,
                "forecast": 0.0,
                "rollout": 0.0,
                "physics": 0.0,
                "observable_noninferiority": 0.0,
                "force_window": 0.0,
                "propagator_growth": 0.0,
                "condition_observer": 0.0,
                "condition_centering": 0.0,
                "basis_cross_orthogonality": 0.0,
                "constraint_divergence": 0.0,
                "constraint_boundary": 0.0,
                "constraint_burden": 0.0,
                "pcgrad_conflicts": 0.0,
                "pcgrad_comparisons": 0.0,
            }
            count = 0
            constraint_count = 0
            for batch_index, raw in enumerate(loaders["train"]):
                batch = _move_rollout(raw, selected) if stabilized else _move(raw, selected)
                optimizer.zero_grad(set_to_none=True)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast:
                    if stabilized:
                        loss, terms = _stabilized_loss_bundle(
                            model,
                            batch,
                            residual_scale,
                            condition_mean,
                            condition_std,
                            resolved,
                            epoch,
                            physical,
                            augmented_lagrangian,
                            validation=False,
                        )
                    else:
                        loss, terms = _loss_bundle(
                            model,
                            batch,
                            residual_scale,
                            condition_mean,
                            condition_std,
                            resolved,
                        )
                audit_interval = resolved.v0_9_training.gradient_audit_interval
                if (
                    stabilized
                    and audit_interval > 0
                    and batch_index == 0
                    and epoch % audit_interval == 0
                ):
                    gradient_objectives: dict[str, torch.Tensor] = {}
                    for name in ("forecast", "rollout"):
                        if name in terms:
                            gradient_objectives[name] = terms[name]
                    for name in (
                        "condition_observer",
                        "condition_centering",
                        "basis_cross_orthogonality",
                    ):
                        if name in terms and bool(terms[name].requires_grad):
                            gradient_objectives[name] = terms[name]
                    for component in resolved.v0_9_training.observable_names:
                        values = [
                            value
                            for name, value in terms.items()
                            if name.startswith(f"observable_{component}_h")
                            and "nominal" not in name
                        ]
                        if values:
                            gradient_objectives[component] = torch.stack(values).mean()
                    for component in ("force_waveform", "force_correlation", "force_spectrum"):
                        values = [
                            value
                            for name, value in terms.items()
                            if name.startswith(f"observable_{component}_h")
                        ]
                        if values:
                            gradient_objectives[component] = torch.stack(values).mean()
                    if len(gradient_objectives) >= 2:
                        geometry = gradient_cosine_matrix(
                            gradient_objectives,
                            tuple(model.operator_adapter.parameters()),
                        )
                        record = {"epoch": epoch + 1, **geometry.to_dict()}
                        gradient_records.append(record)
                        with gradient_log_path.open("a", encoding="utf-8") as audit_stream:
                            audit_stream.write(json.dumps(record, sort_keys=True) + "\n")
                progress = epoch / max(resolved.v0_9_training.epochs - 1, 1)
                use_pcgrad = bool(
                    resolved.v0_9_training.gradient_conflict_method == "pcgrad"
                    and progress
                    >= resolved.v0_9_training.gradient_conflict_start_fraction
                    and all(
                        name in terms
                        for name in (
                            "objective_prediction",
                            "objective_regularization",
                            "objective_observable",
                            "objective_constraints",
                        )
                    )
                )
                if use_pcgrad:
                    pcgrad = pcgrad_backward(
                        {
                            name: scaler.scale(terms[name])
                            for name in (
                                "objective_prediction",
                                "objective_regularization",
                                "objective_observable",
                                "objective_constraints",
                            )
                        },
                        tuple(model.operator_adapter.parameters()),
                    )
                    sums["pcgrad_conflicts"] += pcgrad.projected_conflicts
                    sums["pcgrad_comparisons"] += pcgrad.compared_pairs
                else:
                    scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), resolved.v0_9_training.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                batch_count = (
                    batch["target_latents"].shape[0]
                    if stabilized
                    else batch["target_next"].shape[0]
                )
                sums["total"] += float(loss.detach()) * batch_count
                sums["forecast"] += float(terms["forecast"].detach()) * batch_count
                for name in (
                    "rollout",
                    "physics",
                    "observable_noninferiority",
                    "force_window",
                    "propagator_growth",
                    "condition_observer",
                    "condition_centering",
                    "basis_cross_orthogonality",
                ):
                    if name in terms:
                        sums[name] += float(terms[name].detach()) * batch_count
                if all(
                    f"constraint_{name}" in terms
                    for name in ("divergence", "boundary", "burden")
                ):
                    for name in ("divergence", "boundary", "burden"):
                        sums[f"constraint_{name}"] += float(
                            terms[f"constraint_{name}"].detach()
                        ) * batch_count
                    constraint_count += batch_count
                count += batch_count
                global_step += 1
                optimizer_update_step += 1
            scheduler.step()
            validation = (
                _evaluate_stabilized(
                    model,
                    loaders["validation"],
                    residual_scale,
                    condition_mean,
                    condition_std,
                    resolved,
                    physical,
                    augmented_lagrangian,
                    selected,
                )
                if stabilized
                else _evaluate(
                    model,
                    loaders["validation"],
                    residual_scale,
                    condition_mean,
                    condition_std,
                    resolved,
                    selected,
                )
            )
            if (
                augmented_lagrangian is not None
                and constraint_count > 0
                and state.physics_scale >= 1.0 - 1.0e-12
                and (epoch + 1)
                % resolved.v0_9_training.augmented_lagrangian_update_interval
                == 0
            ):
                latest_constraint_diagnostics = augmented_lagrangian.update(
                    {
                        # Validation evaluates every locked observable horizon,
                        # whereas training samples one horizon for bounded memory.
                        # Update dual variables from the deterministic worst-horizon
                        # contract instead of a stochastic training-horizon average.
                        name: validation[f"constraint_{name}"]
                        for name in ("divergence", "boundary", "burden")
                    }
                )
            row: dict[str, Any] = {
                "epoch": epoch + 1,
                "active_horizons": ";".join(str(value) for value in state.active_horizons),
                "physics_scale": state.physics_scale,
                "train_total": sums["total"] / count,
                "train_forecast": sums["forecast"] / count,
                "train_rollout": sums["rollout"] / count,
                "train_physics": sums["physics"] / count,
                "train_observable_noninferiority": (
                    sums["observable_noninferiority"] / count
                ),
                "train_force_window": sums["force_window"] / count,
                "train_propagator_growth": sums["propagator_growth"] / count,
                "train_condition_observer": sums["condition_observer"] / count,
                "train_condition_centering": sums["condition_centering"] / count,
                "train_basis_cross_orthogonality": (
                    sums["basis_cross_orthogonality"] / count
                ),
                "train_pcgrad_conflicts": sums["pcgrad_conflicts"],
                "train_pcgrad_comparisons": sums["pcgrad_comparisons"],
                "validation_total": validation["total"],
                "validation_selection_score": _validation_selection_score(validation),
                "validation_forecast": validation["forecast"],
                "validation_rollout": validation.get("rollout", 0.0),
                "validation_physics": validation.get("physics", 0.0),
                "validation_observable_noninferiority": validation.get(
                    "observable_noninferiority", 0.0
                ),
                "validation_force_window": validation.get("force_window", 0.0),
                "validation_burden": validation["burden"],
                "validation_burden_max": validation.get("burden_max", 0.0),
                "validation_stability": validation["stability"],
                "validation_propagator_growth": validation.get(
                    "propagator_growth", 0.0
                ),
                "validation_gate_mean": validation.get("gate_mean", 1.0),
                "validation_condition_observer": validation.get(
                    "condition_observer", 0.0
                ),
                "validation_condition_centering": validation.get(
                    "condition_centering", 0.0
                ),
                "validation_basis_cross_orthogonality": validation.get(
                    "basis_cross_orthogonality", 0.0
                ),
                **{
                    name: latest_constraint_diagnostics.get(name, 0.0)
                    for name in (
                        "constraint_max_violation",
                        "multiplier_divergence",
                        "multiplier_boundary",
                        "multiplier_burden",
                        "penalty_divergence",
                        "penalty_boundary",
                        "penalty_burden",
                    )
                },
            }
            for horizon in resolved.v0_9_training.rollout_horizons:
                row[f"validation_rollout_gain_h{horizon}"] = validation.get(
                    f"rollout_gain_h{horizon}", ""
                )
            writer.writerow(row)
            stream.flush()
            score = float(row["validation_selection_score"])
            if score < best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.operator_adapter.state_dict().items()
                }
                if augmented_lagrangian is not None:
                    best_phase1_state = augmented_lagrangian.state_dict()
                stale = 0
                save_adaptive_checkpoint(checkpoint_payload(epoch + 1), best)
            else:
                stale += 1
            completed = epoch + 1
            save_adaptive_checkpoint(checkpoint_payload(completed), latest)
            if (
                epoch >= final_curriculum_start_epoch
                and stale >= resolved.v0_9_training.patience
            ):
                break
    if best_state is None:
        raise RuntimeError("V0.9 training produced no validation checkpoint")
    model.operator_adapter.load_state_dict(best_state, strict=True)
    if augmented_lagrangian is not None and best_phase1_state is not None:
        augmented_lagrangian.load_state_dict(best_phase1_state)
    validation = (
        _evaluate_stabilized(
            model,
            loaders["validation"],
            residual_scale,
            condition_mean,
            condition_std,
            resolved,
            physical,
            augmented_lagrangian,
            selected,
        )
        if stabilized
        else _evaluate(
            model,
            loaders["validation"],
            residual_scale,
            condition_mean,
            condition_std,
            resolved,
            selected,
        )
    )
    validation["selection_score"] = _validation_selection_score(validation)
    summary = {
        "status": "PASS",
        "condition_mode": resolved.v0_9_adaptive.condition_mode,
        "rank": resolved.v0_9_adaptive.rank,
        "completed_epochs": completed,
        "validation": validation,
        "test_locked_confirmation": "NOT_OPENED_DURING_TRAINING",
        "claims": {
            "backbone_frozen": True,
            "context_frozen": True,
            "A0_frozen": True,
            "additive_residual_enabled": False,
            "persistent_z_R_present": False,
            "closed_loop_curriculum": stabilized,
            "bounded_coordinates": resolved.v0_9_adaptive.bounded_coordinates,
            "trust_gate": resolved.v0_9_adaptive.trust_gate,
            "frozen_decoder_observables": resolved.v0_9_training.lambda_physics > 0,
            "phase1_constrained_optimization": resolved.v0_9_training.phase1_enabled,
            "gradient_conflict_method": (
                resolved.v0_9_training.gradient_conflict_method
            ),
            "worst_horizon_constraints": resolved.v0_9_training.phase1_enabled,
            "phase2_condition_history_factorization": bool(
                resolved.v0_9_phase2 is not None and resolved.v0_9_phase2.enabled
            ),
        },
        "curriculum": {
            "rollout_horizons": list(resolved.v0_9_training.rollout_horizons),
            "rollout_start_fractions": list(
                resolved.v0_9_training.rollout_start_fractions
            ),
            "rollout_stride": resolved.v0_9_training.rollout_stride,
            "physics_start_fraction": resolved.v0_9_training.physics_start_fraction,
            "physics_ramp_duration_fraction": (
                resolved.v0_9_training.physics_ramp_duration_fraction
            ),
            "observable_horizons": list(
                resolved.v0_9_training.active_observable_horizons
            ),
            "observable_names": list(resolved.v0_9_training.observable_names),
            "observable_horizon_probabilities": list(
                resolved.v0_9_training.observable_horizon_probabilities
            ),
            "gradient_conflict_start_fraction": (
                resolved.v0_9_training.gradient_conflict_start_fraction
            ),
            "constraint_aggregation": "worst_horizon"
            if resolved.v0_9_training.phase1_enabled
            else "legacy_weighted",
        },
        "phase1": {
            "observable_scale_state": None
            if observable_scale_state is None
            else observable_scale_state.to_dict(),
            "augmented_lagrangian_state": None
            if augmented_lagrangian is None
            else augmented_lagrangian.state_dict(),
            "gradient_audit_records": len(gradient_records),
            "minimum_gradient_cosine": None
            if not gradient_records
            else min(
                float(record["minimum_off_diagonal_cosine"])
                for record in gradient_records
            ),
        },
        "phase2": None
        if resolved.v0_9_phase2 is None or not resolved.v0_9_phase2.enabled
        else {
            "static_rank": resolved.v0_9_phase2.static_rank,
            "dynamic_rank": resolved.v0_9_phase2.dynamic_rank,
            "condition_target": ["Re", "U_infinity", "dRe_dt"],
            "observer_warmup_fraction": (
                resolved.v0_9_phase2.observer_warmup_fraction
            ),
            "conditional_centering_has_variance_floor": False,
        },
    }
    (destination / "evaluation" / "gradient_geometry_summary.json").write_text(
        json.dumps(summary["phase1"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "evaluation" / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[V0.9][train:{resolved.v0_9_adaptive.condition_mode}] PASS "
        f"validation_forecast={validation['forecast']:.6g} "
        f"validation_total={validation['total']:.6g} test=LOCKED",
        flush=True,
    )
    return V09TrainingResult(
        destination,
        latest,
        best,
        resolved.v0_9_adaptive.condition_mode,
        resolved.v0_9_adaptive.rank,
        start_epoch,
        completed,
        validation,
    )
