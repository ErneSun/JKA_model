"""Teacher-free V0.9 rollout with causal predicted latent histories."""

from __future__ import annotations

import torch
from torch import Tensor

from jka_model.adaptive.models import AdaptiveKoopmanModel


@torch.no_grad()
def adaptive_latent_rollout(
    model: AdaptiveKoopmanModel,
    initial_history: Tensor,
    history_dts: Tensor,
    future_dts: Tensor,
    context_parameters: Tensor,
    conditions: Tensor | None,
) -> dict[str, Tensor]:
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
    latent_buffer = initial_history.clone()
    dt_buffer = history_dts.clone()
    adapted_states = [latent_buffer[:, -1]]
    nominal_states = [latent_buffer[:, -1]]
    nominal_current = latent_buffer[:, -1]
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
        current_condition = None if conditions is None else conditions[:, index]
        context = model.context_encoder(
            latent_buffer, dt_buffer, next_dt, context_parameters
        )
        phase2_provider = getattr(model.operator_adapter, "phase2_components", None)
        if phase2_provider is not None:
            components = phase2_provider(context, current_condition)
            for name in phase2_values:
                phase2_values[name].append(components[name])
        prediction, eta, gate, delta, adapted = model.operator_adapter.step_with_gate(
            latent_buffer[:, -1], context, next_dt, current_condition
        )
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(
                f"adaptive rollout became non-finite at step {index + 1}"
            )
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            transition = torch.linalg.matrix_exp(
                nominal.float().unsqueeze(0) * next_dt.reshape(-1, 1, 1).float()
            )
            nominal_current = torch.einsum("bij,bj->bi", transition, nominal_current.float())
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
