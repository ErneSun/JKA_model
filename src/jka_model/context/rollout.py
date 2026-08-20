"""Teacher-free additive-residual utility rollout for V0.8."""

from __future__ import annotations

import torch
from torch import Tensor

from jka_model.context.models import DynamicContextModel
from jka_model.models import ContinuousKoopmanCore


@torch.no_grad()
def context_corrected_latent_rollout(
    model: DynamicContextModel,
    core: ContinuousKoopmanCore,
    initial_history: Tensor,
    history_dts: Tensor,
    future_dts: Tensor,
    parameters: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """After initialization every history state is predicted; truth is never fed back."""
    if initial_history.ndim != 3:
        raise ValueError("initial_history shape mismatch")
    history, latent_dim = initial_history.shape[1:]
    encoder = getattr(model, "context_encoder", None)
    if encoder is not None and (
        getattr(encoder, "history", history) != history
        or getattr(encoder, "latent_dim", latent_dim) != latent_dim
    ):
        raise ValueError("initial history disagrees with context encoder contract")
    if history_dts.shape != (initial_history.shape[0], history - 1):
        raise ValueError("initial history dt alignment mismatch")
    if future_dts.ndim != 2 or torch.any(future_dts <= 0):
        raise ValueError("future_dts must be positive [B,T]")
    latent_buffer = initial_history.clone()
    dt_buffer = history_dts.clone()
    predicted = [latent_buffer[:, -1]]
    bases: list[Tensor] = []
    corrections: list[Tensor] = []
    for index in range(future_dts.shape[1]):
        next_dt = future_dts[:, index : index + 1]
        _, correction, _ = model(latent_buffer, dt_buffer, next_dt, parameters)
        base = core.step(latent_buffer[:, -1], next_dt[:, 0])
        next_latent = base + correction
        predicted.append(next_latent)
        bases.append(base)
        corrections.append(correction)
        if history > 1:
            latent_buffer = torch.cat((latent_buffer[:, 1:], next_latent.unsqueeze(1)), dim=1)
            dt_buffer = torch.cat((dt_buffer[:, 1:], next_dt), dim=1) if history > 2 else next_dt
        else:
            latent_buffer = next_latent.unsqueeze(1)
    return torch.stack(predicted, dim=1), torch.stack(bases, dim=1), torch.stack(corrections, dim=1)
