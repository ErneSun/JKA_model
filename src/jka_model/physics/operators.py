"""Small differentiable periodic finite-difference and quadrature operators."""

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
    derivative = (torch.roll(unique, -1, dims=-1) - torch.roll(unique, 1, dims=-1)) / (2.0 * dx)
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


def periodic_first_derivative_2d(state: Tensor, spacing: float, axis: int) -> Tensor:
    """Second-order centered derivative on an endpoint-free periodic 2-D grid."""
    if state.ndim < 2 or axis not in {-2, -1}:
        raise ValueError("2-D derivative axis must be -2 (x) or -1 (y)")
    if state.shape[axis] < 4 or spacing <= 0:
        raise ValueError("periodic 2-D derivative requires >=4 points and positive spacing")
    return (torch.roll(state, -1, dims=axis) - torch.roll(state, 1, dims=axis)) / (2.0 * spacing)


def periodic_second_derivative_2d(state: Tensor, spacing: float, axis: int) -> Tensor:
    """Second-order centered second derivative on an endpoint-free periodic grid."""
    if state.ndim < 2 or axis not in {-2, -1}:
        raise ValueError("2-D derivative axis must be -2 (x) or -1 (y)")
    if state.shape[axis] < 4 or spacing <= 0:
        raise ValueError("periodic 2-D derivative requires >=4 points and positive spacing")
    return (
        torch.roll(state, -1, dims=axis) - 2.0 * state + torch.roll(state, 1, dims=axis)
    ) / spacing**2


def periodic_laplacian_2d(state: Tensor, dx: float, dy: float) -> Tensor:
    """Sum of endpoint-free periodic second derivatives in x and y."""
    return periodic_second_derivative_2d(state, dx, -2) + periodic_second_derivative_2d(
        state, dy, -1
    )


def weighted_integral_2d(state: Tensor, cell_weights: Tensor) -> Tensor:
    """Integrate the final two axes while preserving batch/channel axes."""
    if state.ndim < 2 or cell_weights.shape[-2:] != state.shape[-2:]:
        raise ValueError("2-D cell_weights must match the final two state axes")
    weights = cell_weights
    while weights.ndim < state.ndim:
        weights = weights.unsqueeze(-3)
    return (state * weights.to(device=state.device, dtype=state.dtype)).sum(dim=(-2, -1))
