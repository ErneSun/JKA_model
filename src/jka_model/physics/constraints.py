"""Concrete raw-state physics constraints and their common protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from jka_model.contracts import ProblemBatch, ProblemSpec
from jka_model.physics.operators import (
    periodic_first_derivative,
    periodic_first_derivative_2d,
    periodic_laplacian_2d,
    periodic_second_derivative,
    weighted_integral,
    weighted_integral_2d,
)


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """One named scalar penalty plus detached scalar diagnostics."""

    name: str
    penalty: Tensor
    diagnostics: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if not self.name.strip() or self.penalty.numel() != 1:
            raise ValueError("constraint result requires a name and scalar penalty")


@runtime_checkable
class PhysicsConstraint(Protocol):
    """Physical law evaluated on raw-unit state tensors only."""

    def loss(
        self,
        pred_state_raw: Tensor,
        *,
        prev_state_raw: Tensor | None = None,
        action: Tensor | None = None,
        dt: Tensor | None = None,
        spec: ProblemSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Tensor]: ...


def _metadata_tensor(metadata: Mapping[str, Any] | None, key: str) -> Tensor:
    value = None if metadata is None else metadata.get(key)
    if not isinstance(value, Tensor):
        raise ValueError(f"constraint requires tensor metadata {key!r}")
    return value


@dataclass(frozen=True, slots=True)
class FiniteValueConstraint:
    name: str = "finite_values"

    def evaluate(self, pred_state_raw: Tensor, **_: Any) -> ConstraintResult:
        invalid = ~torch.isfinite(pred_state_raw)
        penalty = invalid.to(dtype=pred_state_raw.dtype).mean()
        return ConstraintResult(self.name, penalty, {"invalid_fraction": penalty.detach()})

    def loss(self, pred_state_raw: Tensor, **kwargs: Any) -> Mapping[str, Tensor]:
        del kwargs
        result = self.evaluate(pred_state_raw)
        return {result.name: result.penalty}


@dataclass(frozen=True, slots=True)
class StateAdmissibilityConstraint:
    lower: float | None = None
    upper: float | None = None
    name: str = "state_admissibility"

    def __post_init__(self) -> None:
        if self.lower is None and self.upper is None:
            raise ValueError("admissibility requires at least one bound")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("admissibility lower bound must not exceed upper bound")

    def evaluate(self, pred_state_raw: Tensor, **_: Any) -> ConstraintResult:
        penalty = pred_state_raw.new_zeros(())
        if self.lower is not None:
            penalty = penalty + torch.relu(self.lower - pred_state_raw).square().mean()
        if self.upper is not None:
            penalty = penalty + torch.relu(pred_state_raw - self.upper).square().mean()
        return ConstraintResult(self.name, penalty, {"bound_penalty": penalty.detach()})

    def loss(self, pred_state_raw: Tensor, **kwargs: Any) -> Mapping[str, Tensor]:
        del kwargs
        result = self.evaluate(pred_state_raw)
        return {result.name: result.penalty}


@dataclass(frozen=True, slots=True)
class PeriodicBoundaryConstraint:
    name: str = "periodic_boundary"

    def evaluate(self, pred_state_raw: Tensor, **_: Any) -> ConstraintResult:
        mismatch = pred_state_raw[..., -1] - pred_state_raw[..., 0]
        penalty = mismatch.square().mean()
        return ConstraintResult(
            self.name, penalty, {"max_abs_mismatch": mismatch.abs().max().detach()}
        )

    def loss(self, pred_state_raw: Tensor, **kwargs: Any) -> Mapping[str, Tensor]:
        del kwargs
        result = self.evaluate(pred_state_raw)
        return {result.name: result.penalty}


@dataclass(frozen=True, slots=True)
class MassConservationConstraint:
    name: str = "mass_conservation"

    def evaluate(
        self,
        pred_state_raw: Tensor,
        *,
        prev_state_raw: Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> ConstraintResult:
        if prev_state_raw is None:
            raise ValueError("mass conservation requires prev_state_raw")
        weights = _metadata_tensor(metadata, "cell_weights")
        mask_value = None if metadata is None else metadata.get("valid_mask")
        mask = mask_value if isinstance(mask_value, Tensor) else None
        mass_pred = weighted_integral(pred_state_raw, weights, mask)
        mass_prev = weighted_integral(prev_state_raw, weights, mask)
        delta = mass_pred - mass_prev
        penalty = delta.square().mean()
        return ConstraintResult(
            self.name, penalty, {"max_abs_mass_delta": delta.abs().max().detach()}
        )

    def loss(
        self,
        pred_state_raw: Tensor,
        *,
        prev_state_raw: Tensor | None = None,
        action: Tensor | None = None,
        dt: Tensor | None = None,
        spec: ProblemSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Tensor]:
        del action, dt, spec
        result = self.evaluate(pred_state_raw, prev_state_raw=prev_state_raw, metadata=metadata)
        return {result.name: result.penalty}


@dataclass(frozen=True, slots=True)
class DiscretePDEResidualConstraint:
    """Forward-time/centered-space residual of ``u_t+c*u_x=nu*u_xx``."""

    name: str = "discrete_pde_residual"

    def evaluate(
        self,
        pred_state_raw: Tensor,
        *,
        prev_state_raw: Tensor | None = None,
        dt: Tensor | None = None,
        spec: ProblemSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> ConstraintResult:
        if prev_state_raw is None or dt is None or spec is None:
            raise ValueError("PDE residual requires prev_state_raw, dt, and spec")
        if spec.grid.spacing is None or len(spec.grid.spacing) != 1:
            raise ValueError("PDE residual requires one-dimensional grid spacing")
        mu_static = _metadata_tensor(metadata, "mu_static")
        if mu_static.ndim != 2 or mu_static.shape[1] < 2:
            raise ValueError("mu_static must have [c, nu] for each batch item")
        dt_view = dt
        while dt_view.ndim < pred_state_raw.ndim:
            dt_view = dt_view.unsqueeze(-1)
        c = mu_static[:, 0]
        nu = mu_static[:, 1]
        while c.ndim < pred_state_raw.ndim:
            c = c.unsqueeze(-1)
            nu = nu.unsqueeze(-1)
        temporal = (pred_state_raw - prev_state_raw) / dt_view
        gradient = periodic_first_derivative(prev_state_raw, spec.grid.spacing[0])
        laplacian = periodic_second_derivative(prev_state_raw, spec.grid.spacing[0])
        residual = temporal + c * gradient - nu * laplacian
        penalty = residual.square().mean()
        return ConstraintResult(
            self.name,
            penalty,
            {"residual_rms": residual.square().mean().sqrt().detach()},
        )

    def loss(
        self,
        pred_state_raw: Tensor,
        *,
        prev_state_raw: Tensor | None = None,
        action: Tensor | None = None,
        dt: Tensor | None = None,
        spec: ProblemSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Tensor]:
        del action
        result = self.evaluate(
            pred_state_raw,
            prev_state_raw=prev_state_raw,
            dt=dt,
            spec=spec,
            metadata=metadata,
        )
        return {result.name: result.penalty}


@dataclass(frozen=True, slots=True)
class MassConservation2DConstraint:
    """Penalize relative deviation of the cell-weighted two-dimensional integral."""

    name: str = "mass_conservation_2d"

    def loss(
        self,
        pred_state_raw: Tensor,
        *,
        prev_state_raw: Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> Mapping[str, Tensor]:
        if prev_state_raw is None:
            raise ValueError("2-D mass conservation requires prev_state_raw")
        weights = _metadata_tensor(metadata, "cell_weights")
        mass_pred = weighted_integral_2d(pred_state_raw, weights)
        mass_prev = weighted_integral_2d(prev_state_raw, weights)
        scale = weighted_integral_2d(prev_state_raw.abs(), weights).clamp_min(1.0e-12)
        return {self.name: ((mass_pred - mass_prev) / scale).square().mean()}


@dataclass(frozen=True, slots=True)
class AdvectionDiffusionOperatorConstraint2D:
    """Trapezoidal residual for the raw-unit continuous PDE operator."""

    name: str = "operator_consistency_2d"

    @staticmethod
    def _rhs(state: Tensor, mu_static: Tensor, dx: float, dy: float) -> Tensor:
        if mu_static.ndim != 2 or mu_static.shape[1] != 3:
            raise ValueError("2-D PDE requires mu_static=[cx,cy,nu] per batch item")
        parameters = [mu_static[:, index] for index in range(3)]
        for index, value in enumerate(parameters):
            while value.ndim < state.ndim:
                value = value.unsqueeze(-1)
            parameters[index] = value
        cx, cy, nu = parameters
        return (
            -cx * periodic_first_derivative_2d(state, dx, -2)
            - cy * periodic_first_derivative_2d(state, dy, -1)
            + nu * periodic_laplacian_2d(state, dx, dy)
        )

    def loss(
        self,
        pred_state_raw: Tensor,
        *,
        prev_state_raw: Tensor | None = None,
        dt: Tensor | None = None,
        spec: ProblemSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> Mapping[str, Tensor]:
        if prev_state_raw is None or dt is None or spec is None:
            raise ValueError("2-D operator consistency requires prev state, dt, and spec")
        if spec.grid.spacing is None or len(spec.grid.spacing) != 2:
            raise ValueError("2-D operator consistency requires two grid spacings")
        if torch.any(dt <= 0):
            raise ValueError("2-D operator consistency requires positive dt")
        mu_static = _metadata_tensor(metadata, "mu_static")
        dt_view = dt
        while dt_view.ndim < pred_state_raw.ndim:
            dt_view = dt_view.unsqueeze(-1)
        dx, dy = spec.grid.spacing
        residual = (pred_state_raw - prev_state_raw) / dt_view - 0.5 * (
            self._rhs(prev_state_raw, mu_static, dx, dy)
            + self._rhs(pred_state_raw, mu_static, dx, dy)
        )
        return {self.name: residual.square().mean()}


@dataclass(frozen=True, slots=True)
class AdvectionDiffusionSpectralStepConstraint2D:
    """Exact Fourier-step consistency for constant-coefficient periodic advection-diffusion."""

    name: str = "spectral_step_consistency_2d"

    def loss(
        self,
        pred_state_raw: Tensor,
        *,
        prev_state_raw: Tensor | None = None,
        dt: Tensor | None = None,
        spec: ProblemSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> Mapping[str, Tensor]:
        if prev_state_raw is None or dt is None or spec is None:
            raise ValueError("2-D spectral step requires prev state, dt, and spec")
        if spec.grid.spacing is None or len(spec.grid.spacing) != 2:
            raise ValueError("2-D spectral step requires two grid spacings")
        if pred_state_raw.shape != prev_state_raw.shape or pred_state_raw.ndim != 4:
            raise ValueError("2-D spectral step states must share shape [B,C,Nx,Ny]")
        if dt.shape != (pred_state_raw.shape[0],) or torch.any(dt <= 0):
            raise ValueError("2-D spectral step requires positive dt[B]")
        mu_static = _metadata_tensor(metadata, "mu_static")
        if mu_static.shape != (pred_state_raw.shape[0], 3):
            raise ValueError("2-D spectral step requires mu_static=[cx,cy,nu] per batch")

        nx, ny = pred_state_raw.shape[-2:]
        dx, dy = spec.grid.spacing
        real_dtype = prev_state_raw.dtype
        device = prev_state_raw.device
        angular_x = 2.0 * torch.pi * torch.fft.fftfreq(nx, d=dx, device=device, dtype=real_dtype)
        angular_y = 2.0 * torch.pi * torch.fft.fftfreq(ny, d=dy, device=device, dtype=real_dtype)
        wave_x = angular_x.view(1, 1, nx, 1)
        wave_y = angular_y.view(1, 1, 1, ny)
        cx = mu_static[:, 0].to(device=device, dtype=real_dtype).view(-1, 1, 1, 1)
        cy = mu_static[:, 1].to(device=device, dtype=real_dtype).view(-1, 1, 1, 1)
        nu = mu_static[:, 2].to(device=device, dtype=real_dtype).view(-1, 1, 1, 1)
        generator = (-nu * (wave_x.square() + wave_y.square())).to(
            torch.complex128 if real_dtype == torch.float64 else torch.complex64
        ) - 1j * (cx * wave_x + cy * wave_y)
        dt_view = dt.to(device=device, dtype=real_dtype).view(-1, 1, 1, 1)
        previous_spectrum = torch.fft.fftn(prev_state_raw, dim=(-2, -1))
        evolved = torch.fft.ifftn(
            previous_spectrum * torch.exp(generator * dt_view), dim=(-2, -1)
        ).real
        return {self.name: (pred_state_raw - evolved).square().mean()}


def _batched_valid_mask(metadata: Mapping[str, Any] | None, state: Tensor) -> Tensor:
    mask = _metadata_tensor(metadata, "valid_mask").to(device=state.device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).expand(state.shape[0], -1, -1)
    if mask.shape != (state.shape[0], *state.shape[-2:]):
        raise ValueError("cylinder valid_mask must have shape [B,Nx,Ny] or [Nx,Ny]")
    return mask


@dataclass(frozen=True, slots=True)
class CylinderDivergenceConstraint2D:
    """Incompressibility penalty over fluid cells for [u,v,p] states."""

    name: str = "cylinder_divergence_2d"

    def loss(
        self,
        pred_state_raw: Tensor,
        *,
        spec: ProblemSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> Mapping[str, Tensor]:
        if spec is None or spec.grid.spacing is None or pred_state_raw.ndim != 4:
            raise ValueError("cylinder divergence requires [B,C,Nx,Ny] and grid spacing")
        if pred_state_raw.shape[1] < 2:
            raise ValueError("cylinder divergence requires u and v channels")
        dx, dy = spec.grid.spacing
        u, v = pred_state_raw[:, 0], pred_state_raw[:, 1]
        divergence = torch.zeros_like(u)
        divergence[:, 1:-1, 1:-1] = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx) + (
            v[:, 1:-1, 2:] - v[:, 1:-1, :-2]
        ) / (2 * dy)
        mask = _batched_valid_mask(metadata, pred_state_raw)
        interior = mask.clone()
        interior[:, (0, -1), :] = False
        interior[:, :, (0, -1)] = False
        values = divergence[interior]
        return {self.name: values.square().mean() if values.numel() else divergence.new_zeros(())}


@dataclass(frozen=True, slots=True)
class CylinderBoundaryConstraint2D:
    """Fixed inlet/far-field velocity and cylinder no-slip penalty."""

    name: str = "cylinder_boundary_2d"

    def loss(
        self,
        pred_state_raw: Tensor,
        *,
        metadata: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> Mapping[str, Tensor]:
        if pred_state_raw.ndim != 4 or pred_state_raw.shape[1] < 2:
            raise ValueError("cylinder boundary requires [B,C,Nx,Ny] with u and v")
        mask = _batched_valid_mask(metadata, pred_state_raw)
        solid = ~mask
        u, v = pred_state_raw[:, 0], pred_state_raw[:, 1]
        no_slip = (u[solid].square() + v[solid].square()).mean()
        far_u = torch.cat((u[:, 0, :], u[:, :, 0], u[:, :, -1]), dim=1)
        far_v = torch.cat((v[:, 0, :], v[:, :, 0], v[:, :, -1]), dim=1)
        far_field = (far_u - 1.0).square().mean() + far_v.square().mean()
        outlet = (pred_state_raw[:, :2, -1] - pred_state_raw[:, :2, -2]).square().mean()
        return {self.name: no_slip + far_field + outlet}


def evaluate_constraints(
    constraints: Sequence[PhysicsConstraint],
    batch: ProblemBatch,
    spec: ProblemSpec,
    *,
    future_index: int = 0,
) -> dict[str, Tensor]:
    """Route canonical raw fields from a batch into every constraint."""
    if not 0 <= future_index < batch.future_states_raw.shape[1]:
        raise IndexError("future_index is outside the batch horizon")
    prev = (
        batch.context_states_raw[:, -1]
        if future_index == 0
        else batch.future_states_raw[:, future_index - 1]
    )
    action = None if batch.future_actions is None else batch.future_actions[:, future_index]
    metadata = {
        "mu_static": batch.mu_static,
        "coordinates": batch.coordinates,
        "cell_weights": batch.cell_weights,
        "valid_mask": batch.valid_mask,
    }
    terms: dict[str, Tensor] = {}
    for constraint in constraints:
        result = constraint.loss(
            batch.future_states_raw[:, future_index],
            prev_state_raw=prev,
            action=action,
            dt=batch.future_dts[:, future_index],
            spec=spec,
            metadata=metadata,
        )
        duplicate = set(terms) & set(result)
        if duplicate:
            raise ValueError(f"duplicate constraint term(s): {', '.join(sorted(duplicate))}")
        terms.update(result)
    return terms
