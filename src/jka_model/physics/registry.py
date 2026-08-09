"""Explicit registry for concrete physics constraints."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from jka_model.physics.constraints import (
    DiscretePDEResidualConstraint,
    FiniteValueConstraint,
    MassConservationConstraint,
    PeriodicBoundaryConstraint,
    PhysicsConstraint,
    StateAdmissibilityConstraint,
)

ConstraintFactory = Callable[..., PhysicsConstraint]
_REGISTRY: dict[str, ConstraintFactory] = {}


def register_constraint(name: str, factory: ConstraintFactory) -> None:
    if not name.strip():
        raise ValueError("constraint registry name must not be empty")
    if name in _REGISTRY:
        raise ValueError(f"constraint {name!r} is already registered")
    _REGISTRY[name] = factory


def get_constraint_factory(name: str) -> ConstraintFactory:
    try:
        return _REGISTRY[name]
    except KeyError as error:
        raise KeyError(f"unregistered physics constraint: {name!r}") from error


def create_constraint(specification: str | Mapping[str, Any]) -> PhysicsConstraint:
    if isinstance(specification, str):
        return get_constraint_factory(specification)()
    allowed = {"name", "parameters"}
    unknown = set(specification) - allowed
    if unknown:
        raise ValueError(f"unknown constraint specification field(s): {', '.join(unknown)}")
    name = str(specification["name"])
    parameters = specification.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise TypeError("constraint parameters must be a mapping")
    return get_constraint_factory(name)(**dict(parameters))


register_constraint("finite_values", FiniteValueConstraint)
register_constraint("state_admissibility", StateAdmissibilityConstraint)
register_constraint("periodic_boundary", PeriodicBoundaryConstraint)
register_constraint("mass_conservation", MassConservationConstraint)
register_constraint("discrete_pde_residual", DiscretePDEResidualConstraint)
