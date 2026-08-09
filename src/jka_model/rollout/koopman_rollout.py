"""Thin rollout utility that delegates to the KoopmanCore source of truth."""

from __future__ import annotations

from numbers import Real

from torch import Tensor

from jka_model.models import ContinuousKoopmanCore


def koopman_rollout(
    core: ContinuousKoopmanCore,
    z0: Tensor,
    dts: Tensor | Real,
    horizon: int | None = None,
) -> Tensor:
    return core.rollout(z0, dts, horizon=horizon)
