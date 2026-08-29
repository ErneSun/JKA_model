"""Matched Phase-3 route comparison in decoded physical coordinates."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from jka_model.config import (
    V09EvaluationConfig,
    V09Phase2Config,
    V09Phase3Config,
)
from jka_model.manifold.joint import classify_phase3_joint_run

DECODED_QUANTITIES = ("field", "velocity", "vorticity")


def matched_decoded_gains(
    candidate: Mapping[str, float],
    frozen: Mapping[str, float],
    horizons: Sequence[int],
) -> dict[str, float]:
    """Return relative decoded-error gains against the exactly matched frozen run."""
    gains: dict[str, float] = {}
    for quantity in DECODED_QUANTITIES:
        for horizon in horizons:
            name = f"decoded_{quantity}_relative_l2_h{horizon}"
            baseline = float(frozen[name])
            value = float(candidate[name])
            if not math.isfinite(baseline) or not math.isfinite(value) or baseline < 0 or value < 0:
                raise ValueError("decoded route metrics must be finite and non-negative")
            gains[f"decoded_{quantity}_gain_vs_frozen_h{horizon}"] = (
                1.0 - value / max(baseline, 1.0e-12)
            )
    return gains


def classify_matched_phase3_run(
    candidate: Mapping[str, float],
    frozen: Mapping[str, float],
    phase3: V09Phase3Config,
    evaluation: V09EvaluationConfig,
    phase2: V09Phase2Config,
    *,
    route: str,
    condition_mode: str,
) -> dict[str, bool | float]:
    """Combine route-internal gates with decoded gains over a matched frozen control."""
    if route not in {"joint", "from_scratch"}:
        raise ValueError("matched comparison requires joint or from_scratch")
    gates = classify_phase3_joint_run(
        candidate,
        phase3,
        evaluation,
        phase2,
        condition_mode=condition_mode,
        route=route,
    )
    gains = matched_decoded_gains(candidate, frozen, evaluation.rollout_horizons)
    primary = all(
        gains[f"decoded_field_gain_vs_frozen_h{horizon}"]
        >= evaluation.material_relative_gain
        for horizon in evaluation.rollout_horizons
    )
    velocity = all(
        gains[f"decoded_velocity_gain_vs_frozen_h{horizon}"] >= 0.0
        for horizon in evaluation.rollout_horizons
    )
    vorticity = all(
        gains[f"decoded_vorticity_gain_vs_frozen_h{horizon}"] >= 0.0
        for horizon in evaluation.rollout_horizons
    )
    return {
        **gates,
        **gains,
        "decoded_field_material_gain": primary,
        "decoded_velocity_noninferiority": velocity,
        "decoded_vorticity_noninferiority": vorticity,
        "matched_route_pass": bool(
            gates["strict_route"] and primary and velocity and vorticity
        ),
    }


def nested_route_support(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_fraction: float,
) -> dict[str, Any]:
    """Require operator-seed robustness in both modes, then backbone-seed robustness."""
    if not 0 < required_fraction <= 1 or not rows:
        raise ValueError("nested route support requires rows and a fraction in (0,1]")
    grouped: dict[tuple[int, str], list[bool]] = defaultdict(list)
    for row in rows:
        mode = str(row["condition_mode"])
        if mode not in {"known", "latent_inferred"}:
            raise ValueError("invalid matched route condition mode")
        grouped[(int(row["seed"]), mode)].append(bool(row["gates"]["matched_route_pass"]))
    seeds = sorted({key[0] for key in grouped})
    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        modes: dict[str, float] = {}
        for mode in ("known", "latent_inferred"):
            values = grouped.get((seed, mode), [])
            if not values:
                raise ValueError("matched route matrix misses a seed/mode cell")
            modes[mode] = sum(values) / len(values)
        seed_rows.append(
            {
                "seed": seed,
                "mode_pass_fraction": modes,
                "seed_pass": all(value >= required_fraction for value in modes.values()),
            }
        )
    backbone_fraction = sum(bool(row["seed_pass"]) for row in seed_rows) / len(seed_rows)
    return {
        "required_fraction": required_fraction,
        "backbone_pass_fraction": backbone_fraction,
        "supported": backbone_fraction >= required_fraction,
        "seeds": seed_rows,
    }
