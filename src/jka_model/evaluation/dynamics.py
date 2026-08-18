"""Direct-state one-step, rollout, and persistence metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class RolloutMetrics:
    rollout_mse: float
    normalized_rollout_error: float
    persistence_mse: float
    finite: bool


def persistence_rollout(initial_state: Tensor, horizon: int) -> Tensor:
    """Repeat the initial state for ``horizon+1`` states."""
    if horizon < 1:
        raise ValueError("persistence horizon must be positive")
    if initial_state.ndim == 1:
        return initial_state.unsqueeze(0).expand(horizon + 1, -1).clone()
    if initial_state.ndim == 2:
        return initial_state.unsqueeze(1).expand(-1, horizon + 1, -1).clone()
    raise ValueError("initial_state must have shape [d] or [B,d]")


def evaluate_rollout(prediction: Tensor, truth: Tensor) -> RolloutMetrics:
    """Compare closed-loop prediction against truth and persistence."""
    if prediction.shape != truth.shape or prediction.ndim not in {2, 3}:
        raise ValueError("prediction and truth must share [H+1,d] or [B,H+1,d]")
    horizon = prediction.shape[-2] - 1
    if horizon < 1:
        raise ValueError("rollout evaluation requires at least one transition")
    if prediction.ndim == 2:
        rollout_mse = (prediction[1:] - truth[1:]).square().mean()
        initial = truth[0]
        persistence = persistence_rollout(initial, horizon)
        persistence_mse = (persistence[1:] - truth[1:]).square().mean()
        centered = truth[1:]
    else:
        rollout_mse = (prediction[:, 1:] - truth[:, 1:]).square().mean()
        initial = truth[:, 0]
        persistence = persistence_rollout(initial, horizon)
        persistence_mse = (persistence[:, 1:] - truth[:, 1:]).square().mean()
        centered = truth[:, 1:]
    energy = centered.square().mean().clamp_min(torch.finfo(centered.dtype).eps)
    normalized = rollout_mse / energy
    finite = bool(
        torch.isfinite(prediction).all()
        and torch.isfinite(rollout_mse)
        and torch.isfinite(persistence_mse)
    )
    return RolloutMetrics(
        rollout_mse=float(rollout_mse.detach().item()),
        normalized_rollout_error=float(normalized.detach().item()),
        persistence_mse=float(persistence_mse.detach().item()),
        finite=finite,
    )
