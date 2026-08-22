"""Stateful augmented Lagrangian for differentiable inequality constraints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


class InequalityAugmentedLagrangian:
    """Enforce named constraints g_k <= 0 while retaining exact-resume state."""

    def __init__(
        self,
        names: Sequence[str],
        *,
        initial_penalty: float,
        penalty_growth: float,
        maximum_penalty: float,
        improvement_ratio: float = 0.9,
    ) -> None:
        unique = tuple(dict.fromkeys(str(name) for name in names))
        if not unique or len(unique) != len(tuple(names)):
            raise ValueError("augmented-Lagrangian constraint names must be unique")
        if initial_penalty <= 0 or penalty_growth < 1 or maximum_penalty < initial_penalty:
            raise ValueError("invalid augmented-Lagrangian penalty schedule")
        if not 0 < improvement_ratio <= 1:
            raise ValueError("invalid augmented-Lagrangian improvement ratio")
        self.names = unique
        self.multipliers = {name: 0.0 for name in unique}
        self.penalties = {name: float(initial_penalty) for name in unique}
        self.previous_violations = {name: float("inf") for name in unique}
        self.penalty_growth = float(penalty_growth)
        self.maximum_penalty = float(maximum_penalty)
        self.improvement_ratio = float(improvement_ratio)
        self.update_count = 0

    def penalty(self, constraints: Mapping[str, Tensor]) -> Tensor:
        if set(constraints) != set(self.names):
            raise ValueError("augmented-Lagrangian constraint set mismatch")
        reference = next(iter(constraints.values()))
        result = reference.new_zeros(())
        for name in self.names:
            value = constraints[name]
            if value.numel() != 1 or not bool(torch.isfinite(value.detach())):
                raise ValueError(f"constraint {name!r} must be a finite scalar")
            violation = torch.relu(value)
            result = result + self.multipliers[name] * violation
            result = result + 0.5 * self.penalties[name] * violation.square()
        return result

    def update(self, constraints: Mapping[str, float]) -> dict[str, float]:
        if set(constraints) != set(self.names):
            raise ValueError("augmented-Lagrangian update constraint set mismatch")
        diagnostics: dict[str, float] = {}
        for name in self.names:
            raw = float(constraints[name])
            if not torch.isfinite(torch.tensor(raw)):
                raise ValueError(f"constraint {name!r} update is non-finite")
            violation = max(raw, 0.0)
            previous = self.previous_violations[name]
            if (
                violation > 0
                and previous < float("inf")
                and violation > self.improvement_ratio * previous
            ):
                self.penalties[name] = min(
                    self.maximum_penalty,
                    self.penalties[name] * self.penalty_growth,
                )
            self.multipliers[name] = max(
                0.0,
                self.multipliers[name] + self.penalties[name] * raw,
            )
            self.previous_violations[name] = violation
            diagnostics[f"constraint_{name}"] = raw
            diagnostics[f"violation_{name}"] = violation
            diagnostics[f"multiplier_{name}"] = self.multipliers[name]
            diagnostics[f"penalty_{name}"] = self.penalties[name]
        self.update_count += 1
        diagnostics["constraint_max_violation"] = max(
            diagnostics[f"violation_{name}"] for name in self.names
        )
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "multipliers": dict(self.multipliers),
            "penalties": dict(self.penalties),
            "previous_violations": dict(self.previous_violations),
            "penalty_growth": self.penalty_growth,
            "maximum_penalty": self.maximum_penalty,
            "improvement_ratio": self.improvement_ratio,
            "update_count": self.update_count,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if tuple(payload["names"]) != self.names:
            raise ValueError("augmented-Lagrangian checkpoint constraint mismatch")
        for field in ("multipliers", "penalties", "previous_violations"):
            values = {str(key): float(value) for key, value in dict(payload[field]).items()}
            if set(values) != set(self.names):
                raise ValueError(f"augmented-Lagrangian checkpoint {field} mismatch")
            setattr(self, field, values)
        self.update_count = int(payload["update_count"])
