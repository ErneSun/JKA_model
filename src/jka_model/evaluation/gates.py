"""Problem-independent scientific gates with numerical-resolution awareness."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class MetricDirection(str, Enum):
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


@dataclass(frozen=True, slots=True)
class MetricGateSpec:
    """One scalar threshold and/or baseline-relative non-inferiority contract."""

    name: str
    direction: MetricDirection
    threshold: float | None = None
    relative_margin: float = 0.0
    absolute_margin: float = 0.0
    resolution_floor: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("metric gate name must not be empty")
        values = (self.relative_margin, self.absolute_margin, self.resolution_floor)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("metric gate margins/resolution must be finite and non-negative")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("metric gate threshold must be finite")


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    status: GateStatus
    value: float | None
    limit: float | None
    reason: str
    sample_count: int = 1
    pass_fraction: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is GateStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "value": self.value,
            "limit": self.limit,
            "reason": self.reason,
            "sample_count": self.sample_count,
            "pass_fraction": self.pass_fraction,
            "details": self.details,
        }


def evaluate_metric_gate(
    candidate: float,
    spec: MetricGateSpec,
    *,
    baseline: float | None = None,
) -> GateResult:
    """Evaluate absolute and baseline-relative requirements without zero division."""
    if not math.isfinite(candidate) or (
        baseline is not None and not math.isfinite(baseline)
    ):
        return GateResult(
            spec.name,
            GateStatus.INCONCLUSIVE,
            candidate if math.isfinite(candidate) else None,
            None,
            "candidate or baseline is non-finite",
        )
    limits: list[tuple[str, float]] = []
    if spec.threshold is not None:
        limits.append(("absolute_threshold", spec.threshold))
    if baseline is not None:
        if spec.direction is MetricDirection.LOWER_IS_BETTER:
            relative_limit = (
                baseline * (1.0 + spec.relative_margin)
                + spec.absolute_margin
                + spec.resolution_floor
            )
        else:
            relative_limit = (
                baseline * (1.0 - spec.relative_margin)
                - spec.absolute_margin
                - spec.resolution_floor
            )
        limits.append(("baseline_noninferiority", relative_limit))
    if not limits:
        return GateResult(
            spec.name,
            GateStatus.INCONCLUSIVE,
            candidate,
            None,
            "metric gate has neither an absolute threshold nor a baseline",
        )
    if spec.direction is MetricDirection.LOWER_IS_BETTER:
        effective_limit = min(value for _, value in limits)
        relation_scale = max(
            abs(candidate),
            abs(effective_limit),
            spec.resolution_floor,
            1.0e-12,
        )
        numerical_tolerance = 8.0 * 2.0**-23 * relation_scale
        passed = candidate <= effective_limit + numerical_tolerance
        relation = "<="
    else:
        effective_limit = max(value for _, value in limits)
        relation_scale = max(
            abs(candidate),
            abs(effective_limit),
            spec.resolution_floor,
            1.0e-12,
        )
        numerical_tolerance = 8.0 * 2.0**-23 * relation_scale
        passed = candidate >= effective_limit - numerical_tolerance
        relation = ">="
    return GateResult(
        spec.name,
        GateStatus.PASS if passed else GateStatus.FAIL,
        candidate,
        effective_limit,
        f"candidate {relation} effective limit" if passed else f"candidate violates {relation}",
        details={
            "baseline": baseline,
            "direction": spec.direction.value,
            "constraints": {name: value for name, value in limits},
            "relative_margin": spec.relative_margin,
            "absolute_margin": spec.absolute_margin,
            "resolution_floor": spec.resolution_floor,
            "comparison_tolerance": numerical_tolerance,
        },
    )


def aggregate_gate_results(
    name: str,
    results: list[GateResult],
    *,
    required_pass_fraction: float = 1.0,
    minimum_count: int = 1,
) -> GateResult:
    """Aggregate independent records while preserving inconclusive evidence."""
    if not 0 < required_pass_fraction <= 1 or minimum_count < 1:
        raise ValueError("invalid aggregate gate requirements")
    conclusive = [result for result in results if result.status is not GateStatus.INCONCLUSIVE]
    if len(conclusive) < minimum_count:
        return GateResult(
            name,
            GateStatus.INCONCLUSIVE,
            None,
            required_pass_fraction,
            "insufficient conclusive records",
            sample_count=len(results),
            details={"conclusive_count": len(conclusive), "minimum_count": minimum_count},
        )
    pass_fraction = sum(result.passed for result in conclusive) / len(conclusive)
    status = (
        GateStatus.PASS
        if pass_fraction >= required_pass_fraction
        else GateStatus.FAIL
    )
    return GateResult(
        name,
        status,
        pass_fraction,
        required_pass_fraction,
        "required conclusive pass fraction satisfied"
        if status is GateStatus.PASS
        else "required conclusive pass fraction not satisfied",
        sample_count=len(results),
        pass_fraction=pass_fraction,
        details={
            "conclusive_count": len(conclusive),
            "inconclusive_count": len(results) - len(conclusive),
            "failed_metrics": [result.name for result in conclusive if not result.passed],
        },
    )
