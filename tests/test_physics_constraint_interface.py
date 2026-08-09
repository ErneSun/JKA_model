from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from jka_model.contracts import ProblemSpec
from jka_model.physics import PhysicsConstraint


class ToyConstraint:
    """Structural-typing fixture, not a physical-law implementation."""

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
        del prev_state_raw, action, dt, spec, metadata
        return {"toy_zero": pred_state_raw.sum() * 0.0}


def test_physics_constraint_interface_supports_toy_implementation() -> None:
    constraint = ToyConstraint()
    assert isinstance(constraint, PhysicsConstraint)
    losses = constraint.loss(torch.ones(2, 1, 4))
    assert set(losses) == {"toy_zero"}
    assert losses["toy_zero"].shape == ()
    assert losses["toy_zero"].item() == 0.0

