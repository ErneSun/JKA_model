#!/usr/bin/env python3
"""Generate the deterministic V0.2 analytic toy dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from jka_model.config import load_config
from jka_model.data import data_fingerprint, generate_advection_diffusion_trajectories


def _plain_record(record: Any) -> dict[str, Any]:
    return {
        "trajectory_id": record.trajectory_id,
        "states_raw": record.states_raw,
        "actions": record.actions,
        "dts": record.dts,
        "mu_static": record.mu_static,
        "coordinates": record.coordinates,
        "cell_weights": record.cell_weights,
        "valid_mask": record.valid_mask,
        "metadata": dict(record.metadata),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs" / "v0_2_smoke.yaml")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    toy = config.data.toy_advection_diffusion
    if toy is None:
        raise ValueError("configuration does not declare toy_advection_diffusion")
    records, spec = generate_advection_diffusion_trajectories(
        toy, seed=config.training.seed
    )
    fingerprint = data_fingerprint(records, spec)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "problem_spec": spec.to_dict(),
                "trajectories": [_plain_record(record) for record in records],
                "data_fingerprint": fingerprint,
            },
            arguments.output,
        )
    summary = {
        "num_trajectories": len(records),
        "states_shape": list(records[0].states_raw.shape),
        "dt_mode": spec.dt_mode.value,
        "data_fingerprint": fingerprint,
        "output": None if arguments.output is None else str(arguments.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

