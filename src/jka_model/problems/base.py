"""Problem-adapter interface separating scientific problems from trainers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from jka_model.config import V09EvaluationConfig, V09TrainingConfig
from jka_model.contracts import ProblemSpec
from jka_model.data import TrajectoryDataset
from jka_model.observables import ObservableObjective
from jka_model.physics import PhysicsConstraint


class ProblemAdapter(Protocol):
    def build_problem_spec(self) -> ProblemSpec: ...
    def build_dataset(self, *, seed: int) -> TrajectoryDataset: ...
    def build_physics_constraints(self) -> Mapping[str, PhysicsConstraint]: ...
    def compute_reference_metrics(self) -> Mapping[str, float]: ...
    def describe(self) -> Mapping[str, Any]: ...


@runtime_checkable
class ObservableProblemAdapter(ProblemAdapter, Protocol):
    """Optional extension for problems that define decoded scientific observables."""

    def build_observable_objective(
        self,
        *,
        training: V09TrainingConfig | None = None,
        evaluation: V09EvaluationConfig | None = None,
    ) -> ObservableObjective: ...
