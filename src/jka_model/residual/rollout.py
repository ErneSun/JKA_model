"""Teacher-free closed-loop application of the V0.7 residual closure."""

from __future__ import annotations

import torch
from torch import Tensor

from jka_model.residual.closures import ResidualKoopmanModel


@torch.no_grad()
def corrected_latent_rollout(
    model: ResidualKoopmanModel,
    initial_history_z: Tensor,
    history_dts: Tensor,
    future_dts: Tensor,
    parameters: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Roll out using predicted latent history after the initial observed context."""
    if future_dts.ndim != 2 or future_dts.shape[0] != initial_history_z.shape[0]:
        raise ValueError("future_dts must have shape [B,H]")
    model.eval()
    latent_history = initial_history_z.clone()
    dt_history = history_dts.clone()
    states = [latent_history[:, -1]]
    base_steps: list[Tensor] = []
    corrections: list[Tensor] = []
    for index in range(future_dts.shape[1]):
        next_dt = future_dts[:, index : index + 1]
        next_state, base, correction = model.corrected_step(
            latent_history, dt_history, next_dt, parameters
        )
        states.append(next_state)
        base_steps.append(base)
        corrections.append(correction)
        latent_history = torch.cat((latent_history[:, 1:], next_state[:, None]), dim=1)
        if model.residual_head.history > 1:
            dt_history = torch.cat((dt_history[:, 1:], next_dt), dim=1)
    return (
        torch.stack(states, dim=1),
        torch.stack(base_steps, dim=1),
        torch.stack(corrections, dim=1),
    )
