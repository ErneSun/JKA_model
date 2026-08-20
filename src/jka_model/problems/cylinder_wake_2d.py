"""V0.8 fixed-condition cylinder-wake problem adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jka_model.config import CylinderWake2DConfig
from jka_model.contracts import ProblemSpec
from jka_model.data import (
    TrajectoryDataset,
    generate_cylinder_wake_2d_trajectories,
    load_cylinder_wake_dataset,
    make_cylinder_wake_problem_spec,
)
from jka_model.physics import (
    CylinderBoundaryConstraint2D,
    CylinderDivergenceConstraint2D,
    PhysicsConstraint,
)


@dataclass(frozen=True, slots=True)
class CylinderWake2DProblemAdapter:
    config: CylinderWake2DConfig

    def build_problem_spec(self) -> ProblemSpec:
        return make_cylinder_wake_problem_spec(self.config)

    def build_dataset(self, *, seed: int) -> TrajectoryDataset:
        if self.config.dataset_path:
            return load_cylinder_wake_dataset(self.config.dataset_path, self.config).records
        return generate_cylinder_wake_2d_trajectories(self.config, seed=seed).records

    def build_physics_constraints(self) -> Mapping[str, PhysicsConstraint]:
        # The inherited field trainer expects the canonical names mass/operator.
        # For incompressible flow these slots own divergence and fixed-BC/no-slip
        # consistency respectively; their exact definitions stay problem-owned.
        return {
            "mass": CylinderDivergenceConstraint2D(),
            "operator": CylinderBoundaryConstraint2D(),
        }

    def compute_reference_metrics(self) -> Mapping[str, float]:
        return {
            "reynolds_number": self.config.reynolds_number,
            "snapshot_dt": self.config.snapshot_dt,
            "lattice_relaxation_time": self.config.lattice_relaxation_time,
        }

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": "cylinder_wake_2d",
            "equation": "2D incompressible Navier-Stokes",
            "grid": [self.config.nx, self.config.ny],
            "reynolds_number": self.config.reynolds_number,
            "time_varying_boundary": False,
            "solver": "D2Q9-BGK-low-Mach",
        }
