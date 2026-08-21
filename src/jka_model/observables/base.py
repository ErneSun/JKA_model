"""Minimal interface separating decoded observables from model optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor

from jka_model.evaluation.gates import MetricGateSpec


@dataclass(slots=True)
class ObservableLossResult:
    total: Tensor
    terms: dict[str, Tensor]


class ObservableObjective(Protocol):
    """A problem adapter's differentiable and evaluation-only observable contract."""

    name: str

    def training_loss(
        self,
        predicted_raw: Tensor,
        target_raw: Tensor,
        metadata: Mapping[str, Any],
    ) -> ObservableLossResult: ...

    def evaluation_metrics(
        self,
        predicted_trajectory: Tensor,
        target_trajectory: Tensor,
        metadata: Mapping[str, Any],
    ) -> dict[str, float]: ...

    def evaluation_gate_specs(
        self,
        *,
        sequence_length: int,
        dt: float,
    ) -> Mapping[str, MetricGateSpec]: ...
