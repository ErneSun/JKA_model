#!/usr/bin/env python3
"""Generate deterministic analytical damped-oscillator trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from jka_model.config import load_config
from jka_model.data import data_fingerprint, generate_damped_oscillator_trajectories


def _plain_record(record: Any) -> dict[str, Any]:
    return {
        "trajectory_id": record.trajectory_id,
        "states_raw": record.states_raw,
        "dts": record.dts,
        "mu_static": record.mu_static,
        "metadata": dict(record.metadata),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "v0_3_smoke.yaml")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if config.oscillator is None:
        raise ValueError("config does not contain V0.3 oscillator settings")
    dtype = torch.float64 if config.koopman and config.koopman.dtype == "float64" else torch.float32
    records, spec = generate_damped_oscillator_trajectories(
        config.oscillator, seed=config.training.seed, dtype=dtype
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
    print(
        json.dumps(
            {
                "trajectories": len(records),
                "states_shape": list(records[0].states_raw.shape),
                "variable_dt": config.oscillator.variable_dt,
                "dtype": str(dtype),
                "data_fingerprint": fingerprint,
                "output": None if arguments.output is None else str(arguments.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

