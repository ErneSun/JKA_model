"""Modular long-horizon objectives for the frozen-backbone V0.9 revision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from jka_model.adaptive.identifiability import conditional_centering_loss
from jka_model.adaptive.models import (
    AdaptiveKoopmanModel,
    operator_burden,
    symmetric_abscissa_proxy,
)
from jka_model.config import V09Phase2Config, V09TrainingConfig


@dataclass(frozen=True, slots=True)
class CurriculumState:
    active_horizons: tuple[int, ...]
    active_weights: tuple[float, ...]
    physics_scale: float
    observable_horizons: tuple[int, ...]
    observable_weights: tuple[float, ...]
    observable_normalizer: float


@dataclass(slots=True)
class AdaptiveObjectiveResult:
    total: Tensor
    terms: dict[str, Tensor]
    rollout: dict[str, Tensor]


def curriculum_state(
    config: V09TrainingConfig, epoch: int, *, validation: bool = False
) -> CurriculumState:
    if not 0 <= epoch < config.epochs:
        raise ValueError("V0.9 curriculum epoch is outside the configured training range")
    progress = 1.0 if validation else epoch / max(config.epochs - 1, 1)
    active = tuple(
        (horizon, weight)
        for horizon, start, weight in zip(
            config.rollout_horizons,
            config.rollout_start_fractions,
            config.rollout_weights,
            strict=True,
        )
        if progress >= start and weight > 0
    )
    if validation:
        physics_scale = 1.0 if config.lambda_physics > 0 else 0.0
    elif config.lambda_physics == 0 or progress < config.physics_start_fraction:
        physics_scale = 0.0
    else:
        denominator = config.physics_ramp_duration_fraction or max(
            1.0 - config.physics_start_fraction, 1e-12
        )
        physics_scale = min(1.0, (progress - config.physics_start_fraction) / denominator)
    observable_horizons: tuple[int, ...] = ()
    observable_weights: tuple[float, ...] = ()
    observable_normalizer = sum(config.active_observable_horizon_weights)
    if physics_scale > 0 and observable_normalizer > 0:
        if validation or not config.phase1_enabled:
            observable_horizons = config.active_observable_horizons
            observable_weights = config.active_observable_horizon_weights
        else:
            probabilities = config.observable_horizon_probabilities or tuple(
                1.0 / len(config.active_observable_horizons)
                for _ in config.active_observable_horizons
            )
            selected = int(torch.multinomial(torch.tensor(probabilities), 1).item())
            observable_horizons = (config.active_observable_horizons[selected],)
            observable_weights = (
                config.active_observable_horizon_weights[selected] / probabilities[selected],
            )
    return CurriculumState(
        tuple(item[0] for item in active),
        tuple(item[1] for item in active),
        physics_scale,
        observable_horizons,
        observable_weights,
        observable_normalizer,
    )


def differentiable_adaptive_rollout(
    model: AdaptiveKoopmanModel,
    initial_history: Tensor,
    history_dts: Tensor,
    future_dts: Tensor,
    context_parameters: Tensor,
    conditions: Tensor | None,
) -> dict[str, Tensor]:
    """Closed-loop rollout that retains gradients only through the operator adapter."""
    if initial_history.ndim != 3:
        raise ValueError("initial_history must have shape [B,H,d_K]")
    batch, history, latent_dim = initial_history.shape
    if history_dts.shape != (batch, history - 1):
        raise ValueError("history dt alignment mismatch")
    if future_dts.ndim != 2 or future_dts.shape[0] != batch or torch.any(future_dts <= 0):
        raise ValueError("future_dts must be positive [B,T]")
    condition_dim = int(getattr(model.operator_adapter, "condition_dim", 2))
    if conditions is not None and conditions.shape != (
        batch,
        future_dts.shape[1],
        condition_dim,
    ):
        raise ValueError(f"known conditions must have shape [B,T,{condition_dim}]")
    if model.context_encoder.latent_dim != latent_dim or model.context_encoder.history != history:
        raise ValueError("initial history disagrees with the frozen context contract")
    latent_buffer = initial_history
    dt_buffer = history_dts
    nominal_current = initial_history[:, -1]
    adapted_states: list[Tensor] = []
    nominal_states: list[Tensor] = []
    etas: list[Tensor] = []
    gates: list[Tensor] = []
    deltas: list[Tensor] = []
    generators: list[Tensor] = []
    phase2_values: dict[str, list[Tensor]] = {
        name: []
        for name in (
            "q_hat",
            "q_used",
            "static_coordinates",
            "dynamic_coordinates",
            "static_delta",
            "dynamic_delta",
        )
    }
    nominal = model.operator_adapter.nominal_generator
    for index in range(future_dts.shape[1]):
        next_dt = future_dts[:, index : index + 1]
        condition = None if conditions is None else conditions[:, index]
        # The encoder parameters are frozen, but its Jacobian with respect to a
        # predicted history must remain in the graph.  Otherwise multi-step
        # training degenerates into detached one-step corrections.
        context = model.context_encoder(
            latent_buffer, dt_buffer, next_dt, context_parameters
        )
        phase2_provider = getattr(model.operator_adapter, "phase2_components", None)
        if phase2_provider is not None:
            components = phase2_provider(context, condition)
            for name in phase2_values:
                phase2_values[name].append(components[name])
        prediction, eta, gate, delta, adapted = model.operator_adapter.step_with_gate(
            latent_buffer[:, -1], context, next_dt, condition
        )
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(
                f"Phase-2 differentiable rollout became non-finite at step {index + 1}"
            )
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            nominal_transition = torch.linalg.matrix_exp(
                nominal.float().unsqueeze(0) * next_dt.reshape(-1, 1, 1).float()
            )
            nominal_current = torch.einsum(
                "bij,bj->bi", nominal_transition, nominal_current.float()
            )
        adapted_states.append(prediction)
        nominal_states.append(nominal_current)
        etas.append(eta)
        gates.append(gate)
        deltas.append(delta)
        generators.append(adapted)
        if history > 1:
            latent_buffer = torch.cat((latent_buffer[:, 1:], prediction.unsqueeze(1)), dim=1)
            dt_buffer = torch.cat((dt_buffer[:, 1:], next_dt), dim=1) if history > 2 else next_dt
        else:
            latent_buffer = prediction.unsqueeze(1)
    result = {
        "adapted": torch.stack(adapted_states, dim=1),
        "nominal": torch.stack(nominal_states, dim=1),
        "eta": torch.stack(etas, dim=1),
        "gate": torch.stack(gates, dim=1),
        "delta_a": torch.stack(deltas, dim=1),
        "a_t": torch.stack(generators, dim=1),
    }
    result.update(
        {
            name: torch.stack(values, dim=1)
            for name, values in phase2_values.items()
            if values
        }
    )
    return result


def relative_propagator_growth_loss(
    adapted_generators: Tensor,
    nominal_generator: Tensor,
    dts: Tensor,
    *,
    margin: float,
) -> Tensor:
    """Penalize local propagator amplification beyond the frozen nominal law."""
    if adapted_generators.ndim != 4 or dts.shape != adapted_generators.shape[:2]:
        raise ValueError("propagator growth expects A[B,T,d,d] and dt[B,T]")
    batch, steps, dimension, _ = adapted_generators.shape
    with torch.autocast(device_type=adapted_generators.device.type, enabled=False):
        adapted_transition = torch.linalg.matrix_exp(
            adapted_generators.float().reshape(-1, dimension, dimension)
            * dts.float().reshape(-1, 1, 1)
        )
        nominal_transition = torch.linalg.matrix_exp(
            nominal_generator.float().reshape(1, dimension, dimension)
            * dts.float().reshape(-1, 1, 1)
        )
        adapted_norm = torch.linalg.matrix_norm(adapted_transition, ord=2).reshape(batch, steps)
        nominal_norm = torch.linalg.matrix_norm(nominal_transition, ord=2).reshape(batch, steps)
    return torch.relu(adapted_norm - nominal_norm - margin).square().mean()


def adaptive_stabilization_objective(
    model: AdaptiveKoopmanModel,
    batch: dict[str, Tensor],
    residual_scale: Tensor,
    condition_mean: Tensor,
    condition_std: Tensor,
    config: V09TrainingConfig,
    condition_mode: str,
    curriculum: CurriculumState,
    smooth_schedule_mask: Tensor,
    phase2: V09Phase2Config | None = None,
) -> AdaptiveObjectiveResult:
    required_steps = max(
        1,
        *(curriculum.active_horizons or (1,)),
        max(curriculum.observable_horizons) if curriculum.observable_horizons else 1,
    )
    if batch["future_dts"].shape[1] < required_steps:
        raise ValueError("rollout batch is shorter than the active curriculum")
    future_dts = batch["future_dts"][:, :required_steps]
    truth = batch["target_latents"][:, :required_steps]
    condition_source = (
        batch["future_condition_targets"]
        if phase2 is not None and phase2.enabled
        else batch["future_conditions"]
    )
    conditions = (
        (condition_source[:, :required_steps] - condition_mean) / condition_std
        if condition_mode == "known"
        else None
    )
    rollout = differentiable_adaptive_rollout(
        model,
        batch["history_z"],
        batch["history_dts"],
        future_dts,
        batch["context_parameters"],
        conditions,
    )
    one_step = ((rollout["adapted"][:, 0] - truth[:, 0]) / residual_scale).square().mean()
    rollout_loss = one_step.new_zeros(())
    terms: dict[str, Tensor] = {"forecast": one_step}
    for horizon, weight in zip(
        curriculum.active_horizons, curriculum.active_weights, strict=True
    ):
        endpoint = (
            (rollout["adapted"][:, horizon - 1] - truth[:, horizon - 1])
            / residual_scale
        ).square().mean()
        rollout_loss = rollout_loss + weight * endpoint
        terms[f"rollout_h{horizon}"] = endpoint
        adaptive_rmse = (
            rollout["adapted"][:, horizon - 1] - truth[:, horizon - 1]
        ).square().mean().sqrt()
        nominal_rmse = (
            rollout["nominal"][:, horizon - 1] - truth[:, horizon - 1]
        ).square().mean().sqrt()
        terms[f"rollout_gain_h{horizon}"] = 1.0 - adaptive_rmse / nominal_rmse.clamp_min(1e-12)
    burdens = operator_burden(
        rollout["delta_a"].reshape(-1, *rollout["delta_a"].shape[-2:]),
        model.operator_adapter.nominal_generator,
    ).reshape(rollout["delta_a"].shape[:2])
    burden = burdens.square().mean() + torch.relu(
        burdens - config.operator_burden_target
    ).square().mean()
    baseline_proxy = symmetric_abscissa_proxy(model.operator_adapter.nominal_generator).detach()
    stability = torch.relu(
        symmetric_abscissa_proxy(rollout["a_t"]) - baseline_proxy
    ).square().mean()
    growth_horizons = sorted(
        {
            *(curriculum.active_horizons or (1,)),
            *(
                curriculum.observable_horizons
            ),
        }
    )
    growth_indices = torch.tensor(
        [horizon - 1 for horizon in growth_horizons],
        device=rollout["a_t"].device,
        dtype=torch.long,
    )
    growth = relative_propagator_growth_loss(
        rollout["a_t"].index_select(1, growth_indices),
        model.operator_adapter.nominal_generator,
        future_dts.index_select(1, growth_indices),
        margin=config.propagator_growth_margin,
    )
    smoothness = one_step.new_zeros(())
    if rollout["eta"].shape[1] > 1 and bool(smooth_schedule_mask.any()):
        differences = torch.diff(rollout["eta"][smooth_schedule_mask], dim=1)
        dt = future_dts[smooth_schedule_mask, 1:].clamp_min(1e-12)
        smoothness = (differences.square().sum(dim=-1) / dt).mean()
    orthogonality = model.operator_adapter.orthogonality_loss()
    observer_loss = one_step.new_zeros(())
    centering = one_step.new_zeros(())
    cross_orthogonality = one_step.new_zeros(())
    if phase2 is not None and phase2.enabled:
        if "q_hat" not in rollout or "dynamic_coordinates" not in rollout:
            raise RuntimeError("Phase-2 objective requires factorized rollout diagnostics")
        observer_target = (
            condition_source[:, :required_steps] - condition_mean
        ) / condition_std
        observer_loss = F.smooth_l1_loss(
            rollout["q_hat"], observer_target, beta=1.0
        )
        selected_indices = sorted(
            {
                0,
                *(horizon - 1 for horizon in curriculum.active_horizons),
                *(horizon - 1 for horizon in curriculum.observable_horizons),
            }
        )
        index = torch.tensor(
            selected_indices,
            device=rollout["dynamic_coordinates"].device,
            dtype=torch.long,
        )
        innovations = rollout["dynamic_coordinates"].index_select(1, index).flatten(0, 1)
        observed_conditions = rollout["q_used"].index_select(1, index).flatten(0, 1)
        centering = conditional_centering_loss(
            innovations,
            observed_conditions,
            bandwidth=phase2.conditional_centering_bandwidth,
        )
        cross_provider = getattr(
            model.operator_adapter, "cross_basis_orthogonality_loss", None
        )
        if cross_provider is None:
            raise RuntimeError("Phase-2 adapter lacks cross-basis orthogonality")
        cross_orthogonality = cross_provider()
    total = (
        one_step
        + config.lambda_rollout * rollout_loss
        + config.lambda_operator_burden * (burden + orthogonality)
        + config.lambda_smooth * smoothness
        + config.lambda_stability * stability
        + config.lambda_propagator_growth * growth
        + (0.0 if phase2 is None else phase2.lambda_condition_observer) * observer_loss
        + (0.0 if phase2 is None else phase2.lambda_condition_centering) * centering
        + (0.0 if phase2 is None else phase2.lambda_basis_cross_orthogonality)
        * cross_orthogonality
    )
    terms.update(
        {
            "rollout": rollout_loss,
            "burden": burden,
            "burden_mean": burdens.mean(),
            "burden_max": burdens.max(),
            "smoothness": smoothness,
            "stability": stability,
            "propagator_growth": growth,
            "orthogonality": orthogonality,
            "gate_mean": rollout["gate"].mean(),
            "condition_observer": observer_loss,
            "condition_centering": centering,
            "basis_cross_orthogonality": cross_orthogonality,
        }
    )
    return AdaptiveObjectiveResult(total, terms, rollout)
