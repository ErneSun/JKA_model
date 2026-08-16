#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(control: dict, jepa: dict, rollout_margin: float, physics_margin: float) -> dict:
    c_long, j_long = control["rollout"]["long"], jepa["rollout"]["long"]
    rollout_ratio = j_long["rmse"] / max(c_long["rmse"], 1e-12)
    mass_ratio = j_long["mass_drift"] / max(c_long["mass_drift"], 1e-12)
    operator_ratio = j_long["operator"] / max(c_long["operator"], 1e-12)
    # Ratios are ill-conditioned when a constraint is already near zero. In that
    # regime, the inherited V0.5 absolute threshold is the meaningful gate.
    mass_limit = max(
        (1 + physics_margin) * c_long["mass_drift"],
        float(jepa["relative_mass_drift_threshold"]),
    )
    operator_limit = max(
        (1 + physics_margin) * c_long["operator"],
        float(jepa["operator_mse_threshold"]),
    )
    gates = {
        "long_rollout_noninferiority": rollout_ratio <= 1 + rollout_margin,
        "mass_noninferiority": j_long["mass_drift"] <= mass_limit,
        "operator_noninferiority": j_long["operator"] <= operator_limit,
        "no_collapse": bool(jepa["collapse_gate"]),
        "target_not_in_rollout": not bool(jepa["target_used_for_rollout"]),
    }
    return {
        "pass": all(gates.values()),
        "gates": gates,
        "ratios": {"long_rmse": rollout_ratio, "mass": mass_ratio, "operator": operator_ratio},
        "limits": {
            "long_rmse_ratio": 1 + rollout_margin,
            "mass_drift": mass_limit,
            "operator": operator_limit,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--jepa", type=Path, required=True)
    parser.add_argument("--rollout-margin", type=float, default=0.05)
    parser.add_argument("--physics-margin", type=float, default=0.10)
    args = parser.parse_args()
    result = compare(
        json.loads(args.control.read_text()),
        json.loads(args.jepa.read_text()),
        args.rollout_margin,
        args.physics_margin,
    )
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
