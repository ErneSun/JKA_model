"""Registry-backed construction of scientific problem adapters."""

from __future__ import annotations

from collections.abc import Callable

from jka_model.config import ProjectConfig
from jka_model.problems.advection_diffusion_2d import AdvectionDiffusion2DProblemAdapter
from jka_model.problems.base import ObservableProblemAdapter, ProblemAdapter
from jka_model.problems.cylinder_wake_2d import CylinderWake2DProblemAdapter

ProblemAdapterFactory = Callable[[ProjectConfig], ProblemAdapter]


def _advection_diffusion_2d(config: ProjectConfig) -> ProblemAdapter:
    if config.advection_diffusion_2d is None:
        raise ValueError("periodic_advection_diffusion_2d requires its problem config")
    return AdvectionDiffusion2DProblemAdapter(config.advection_diffusion_2d)


def _cylinder_wake_2d(config: ProjectConfig) -> ProblemAdapter:
    if config.cylinder_wake_2d is None:
        raise ValueError("cylinder_wake_2d requires its problem config")
    return CylinderWake2DProblemAdapter(config.cylinder_wake_2d)


_FACTORIES: dict[str, ProblemAdapterFactory] = {
    "periodic_advection_diffusion_2d": _advection_diffusion_2d,
    "cylinder_wake_2d": _cylinder_wake_2d,
}


def register_problem_adapter(name: str, factory: ProblemAdapterFactory) -> None:
    """Register one adapter factory without changing the canonical trainer."""
    if not name.strip():
        raise ValueError("problem adapter name must not be empty")
    if name in _FACTORIES:
        raise ValueError(f"problem adapter {name!r} is already registered")
    _FACTORIES[name] = factory


def create_problem_adapter(config: ProjectConfig) -> ProblemAdapter:
    """Resolve the configured problem through the adapter registry."""
    try:
        factory = _FACTORIES[config.data.problem_name]
    except KeyError as error:
        known = ", ".join(sorted(_FACTORIES))
        raise ValueError(
            f"no problem adapter registered for {config.data.problem_name!r}; known: {known}"
        ) from error
    return factory(config)


def create_observable_problem_adapter(config: ProjectConfig) -> ObservableProblemAdapter:
    """Resolve a problem and require the optional decoded-observable contract."""
    adapter = create_problem_adapter(config)
    if not isinstance(adapter, ObservableProblemAdapter):
        raise ValueError(
            f"problem {config.data.problem_name!r} has no observable objective"
        )
    return adapter
