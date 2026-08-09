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
    periodic_second_derivative,
    weighted_integral,
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
        result = self.evaluate(
            pred_state_raw, prev_state_raw=prev_state_raw, metadata=metadata
        )
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
    action = (
        None if batch.future_actions is None else batch.future_actions[:, future_index]
    )
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
