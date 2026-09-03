"""Independent causal-condition observer controls for V0.9 Phase 3.7."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from jka_model.config import V09Phase2Config, V09Phase3Config

OBSERVER_VARIANTS = ("history", "instantaneous", "shuffled_history")


def observer_history_variant(history_z: Tensor, variant: str) -> Tensor:
    """Return a label-free, parameter-matched observer input variant.

    ``instantaneous`` removes memory by repeating the current state.  The shuffled
    control reverses the pre-current history while preserving the current state and
    the set of observed latent values.  The deterministic permutation makes the
    formal audit exactly reproducible.
    """
    if history_z.ndim != 3 or history_z.shape[1] < 2:
        raise ValueError("observer controls require history [B,H,d] with H >= 2")
    if variant not in OBSERVER_VARIANTS:
        raise ValueError("unknown observer-control variant")
    if not torch.isfinite(history_z).all():
        raise ValueError("observer-control history must be finite")
    if variant == "history":
        return history_z
    current = history_z[:, -1:]
    if variant == "instantaneous":
        return current.expand_as(history_z)
    return torch.cat((history_z[:, :-1].flip(1), current), dim=1)


def classify_observer_admission(
    metrics: Mapping[str, Mapping[str, float]],
    phase2: V09Phase2Config,
    phase3: V09Phase3Config,
) -> dict[str, float | bool]:
    """Require absolute observer skill and material history/control separation."""
    required = {*OBSERVER_VARIANTS, "mean"}
    if set(metrics) != required:
        raise ValueError("observer admission requires history/instant/shuffle/mean metrics")
    for values in metrics.values():
        if not all(torch.isfinite(torch.tensor(float(value))) for value in values.values()):
            raise ValueError("observer admission metrics must be finite")
    history = metrics["history"]
    instantaneous = metrics["instantaneous"]
    shuffled = metrics["shuffled_history"]
    history_gain = float(instantaneous["normalized_rmse"]) - float(
        history["normalized_rmse"]
    )
    shuffled_gain = float(shuffled["normalized_rmse"]) - float(
        history["normalized_rmse"]
    )
    absolute = (
        float(history["normalized_rmse"])
        <= phase2.max_condition_observer_normalized_rmse
        and float(history["minimum_r2"]) >= phase2.min_condition_observer_r2
    )
    controls = (
        history_gain >= phase3.min_observer_history_gain
        and shuffled_gain >= phase3.min_observer_history_gain
        and float(history["normalized_rmse"])
        < float(metrics["mean"]["normalized_rmse"])
    )
    return {
        "absolute_skill": absolute,
        "history_control": controls,
        "history_gain_vs_instantaneous": history_gain,
        "history_gain_vs_shuffled": shuffled_gain,
        "admitted": absolute and controls,
    }
