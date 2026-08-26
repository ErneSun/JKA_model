"""Geometry-aware building blocks for the V0.9 Phase-3 decoder study."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


def central_difference_2d(field: Tensor, spacing: float, axis: int) -> Tensor:
    """Second-order interior and one-sided boundary derivative on the last two axes."""
    if field.ndim < 2 or axis not in {-2, -1} or spacing <= 0:
        raise ValueError("central difference requires a field, positive spacing and spatial axis")
    result = torch.zeros_like(field)
    if axis == -2:
        if field.shape[-2] < 3:
            raise ValueError("x derivative requires at least three grid points")
        result[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (2.0 * spacing)
        result[..., 0, :] = (field[..., 1, :] - field[..., 0, :]) / spacing
        result[..., -1, :] = (field[..., -1, :] - field[..., -2, :]) / spacing
    else:
        if field.shape[-1] < 3:
            raise ValueError("y derivative requires at least three grid points")
        result[..., :, 1:-1] = (field[..., :, 2:] - field[..., :, :-2]) / (2.0 * spacing)
        result[..., :, 0] = (field[..., :, 1] - field[..., :, 0]) / spacing
        result[..., :, -1] = (field[..., :, -1] - field[..., :, -2]) / spacing
    return result


class StreamFunctionPhysicalDecoder2D(nn.Module):
    """Decode velocity as a boundary lifting plus the curl of one scalar field.

    The velocity correction is divergence-free in the rectangular-grid interior because
    the same discrete derivative pair is used for both mixed derivatives. Cylinder and
    external boundary values are then imposed explicitly. Pressure remains a separate
    learned channel because incompressibility alone does not determine its gauge.

    This module is a Phase-3 candidate, not a post-hoc claim: it must be trained and
    compared against the frozen decoder under the matched route contract.
    """

    def __init__(
        self,
        latent_dim: int,
        nx: int,
        ny: int,
        *,
        hidden_dim: int = 128,
        dx: float = 1.0,
        dy: float = 1.0,
    ) -> None:
        super().__init__()
        if min(latent_dim, nx, ny, hidden_dim) < 1 or min(dx, dy) <= 0:
            raise ValueError("invalid stream-function decoder dimensions")
        self.latent_dim, self.nx, self.ny = latent_dim, nx, ny
        self.dx, self.dy = float(dx), float(dy)
        self.streamfunction_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, nx * ny),
        )
        self.pressure_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, nx * ny),
        )

    def forward(
        self,
        latent: Tensor,
        *,
        valid_mask: Tensor,
        inlet_velocity: Tensor,
    ) -> Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("stream-function latent must have shape [B,d]")
        batch = latent.shape[0]
        if valid_mask.shape not in {(self.nx, self.ny), (batch, self.nx, self.ny)}:
            raise ValueError("valid_mask must have shape [Nx,Ny] or [B,Nx,Ny]")
        if inlet_velocity.shape not in {(batch,), (batch, 1)}:
            raise ValueError("inlet velocity must have shape [B] or [B,1]")
        mask = valid_mask.to(device=latent.device, dtype=torch.bool)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(batch, -1, -1)
        psi = self.streamfunction_head(latent).reshape(batch, self.nx, self.ny)
        pressure = self.pressure_head(latent).reshape(batch, self.nx, self.ny)
        # Curl(psi): u=dpsi/dy and v=-dpsi/dx.
        u = central_difference_2d(psi, self.dy, -1)
        v = -central_difference_2d(psi, self.dx, -2)
        velocity = inlet_velocity.reshape(batch, 1, 1)
        # Cylinder no-slip is exact on solid cells.
        u = torch.where(mask, u, torch.zeros_like(u))
        v = torch.where(mask, v, torch.zeros_like(v))
        pressure = torch.where(mask, pressure, torch.zeros_like(pressure))
        # The benchmark prescribes uniform velocity on inlet and far-field walls.
        u[:, 0, :] = velocity[:, 0, :]
        v[:, 0, :] = 0.0
        u[:, :, 0] = velocity[:, 0, :]
        v[:, :, 0] = 0.0
        u[:, :, -1] = velocity[:, 0, :]
        v[:, :, -1] = 0.0
        return torch.stack((u, v, pressure), dim=1)


def physical_manifold_metrics(
    state: Tensor,
    *,
    valid_mask: Tensor,
    dx: float,
    dy: float,
    boundary_target: Tensor | None = None,
) -> Mapping[str, Tensor]:
    """Return differentiable divergence, no-slip and optional outer-boundary metrics."""
    if state.ndim != 4 or state.shape[1] < 2:
        raise ValueError("physical manifold state must have shape [B,C,Nx,Ny]")
    batch, _, nx, ny = state.shape
    if valid_mask.shape not in {(nx, ny), (batch, nx, ny)}:
        raise ValueError("physical manifold mask shape mismatch")
    mask = valid_mask.to(device=state.device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).expand(batch, -1, -1)
    divergence = central_difference_2d(state[:, 0], dx, -2) + central_difference_2d(
        state[:, 1], dy, -1
    )
    fluid = mask.to(state.dtype)
    solid = (~mask).to(state.dtype)
    divergence_mse = (divergence.square() * fluid).sum() / fluid.sum().clamp_min(1.0)
    no_slip_mse = (state[:, :2].square() * solid.unsqueeze(1)).sum() / (
        2.0 * solid.sum().clamp_min(1.0)
    )
    result: dict[str, Tensor] = {
        "divergence_rms": divergence_mse.sqrt(),
        "boundary_no_slip_mse": no_slip_mse,
    }
    if boundary_target is not None:
        if boundary_target.shape != state.shape:
            raise ValueError("outer-boundary target shape mismatch")
        outer = torch.zeros((batch, nx, ny), device=state.device, dtype=torch.bool)
        outer[:, 0, :] = True
        outer[:, :, 0] = True
        outer[:, :, -1] = True
        delta = state[:, :2] - boundary_target[:, :2]
        result["outer_boundary_mse"] = delta.square()[outer.unsqueeze(1).expand_as(delta)].mean()
    return result
