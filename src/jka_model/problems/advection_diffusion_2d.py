"""V0.5 adapter for the periodic two-dimensional reference PDE."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jka_model.config import AdvectionDiffusion2DConfig
from jka_model.contracts import ProblemSpec
from jka_model.data.advection_diffusion_2d import (
    generate_advection_diffusion_2d_trajectories,
    make_advection_diffusion_2d_problem_spec,
)
from jka_model.data.datasets import TrajectoryDataset
from jka_model.physics.constraints import (
    AdvectionDiffusionOperatorConstraint2D,
    MassConservation2DConstraint,
    PhysicsConstraint,
)


@dataclass(frozen=True, slots=True)
class AdvectionDiffusion2DProblemAdapter:
    config: AdvectionDiffusion2DConfig

    def build_problem_spec(self) -> ProblemSpec:
        return make_advection_diffusion_2d_problem_spec(self.config)

    def build_dataset(self, *, seed: int) -> TrajectoryDataset:
        return generate_advection_diffusion_2d_trajectories(self.config, seed=seed).records

    def build_physics_constraints(self) -> Mapping[str, PhysicsConstraint]:
        return {
            "mass": MassConservation2DConstraint(),
            "operator": AdvectionDiffusionOperatorConstraint2D(),
        }

    def compute_reference_metrics(self) -> Mapping[str, float]:
        data = generate_advection_diffusion_2d_trajectories(self.config, seed=0)
        return {
            "angular_frequency": data.reference.angular_frequency,
            "decay_rate": data.reference.decay_rate,
            "domain_area": data.reference.domain_area,
        }

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": "periodic_advection_diffusion_2d",
            "equation": "u_t + cx*u_x + cy*u_y = nu*(u_xx+u_yy)",
            "grid": [self.config.nx, self.config.ny],
            "endpoint": False,
        }
