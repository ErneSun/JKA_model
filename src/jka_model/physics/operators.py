"""Small differentiable operators for the V0.2 one-dimensional toy problem."""

from __future__ import annotations

import torch
from torch import Tensor


def weighted_integral(
    state: Tensor,
    cell_weights: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Integrate the final spatial axis, preserving leading batch/channel axes."""
    weights = cell_weights
    while weights.ndim < state.ndim:
        weights = weights.unsqueeze(-2)
    if valid_mask is not None:
        mask = valid_mask
        while mask.ndim < state.ndim:
            mask = mask.unsqueeze(-2)
        weights = weights * mask.to(dtype=state.dtype)
    return (state * weights.to(device=state.device, dtype=state.dtype)).sum(dim=-1)


def periodic_first_derivative(state: Tensor, dx: float) -> Tensor:
    """Second-order centered derivative for a grid with a duplicated endpoint."""
    if state.shape[-1] < 4 or dx <= 0:
        raise ValueError("periodic derivative requires at least four points and positive dx")
    unique = state[..., :-1]
    derivative = (torch.roll(unique, -1, dims=-1) - torch.roll(unique, 1, dims=-1)) / (
        2.0 * dx
    )
    return torch.cat((derivative, derivative[..., :1]), dim=-1)


def periodic_second_derivative(state: Tensor, dx: float) -> Tensor:
    """Second-order centered Laplacian for a grid with a duplicated endpoint."""
    if state.shape[-1] < 4 or dx <= 0:
        raise ValueError("periodic derivative requires at least four points and positive dx")
    unique = state[..., :-1]
    derivative = (
        torch.roll(unique, -1, dims=-1) - 2.0 * unique + torch.roll(unique, 1, dims=-1)
    ) / dx**2
    return torch.cat((derivative, derivative[..., :1]), dim=-1)

