#!/usr/bin/env python3
"""Thin CLI wrapper around the canonical V0.5 trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.train_v0_5 import train_v0_5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    result = train_v0_5(
        args.config, device=args.device, resume_from=args.resume_from, run_name=args.run_name
    )
    backend = result.run_dir.parent.name
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "latest_checkpoint": str(result.latest_checkpoint),
                "best_checkpoint": str(result.best_checkpoint),
                "start_epoch": result.start_epoch,
                "completed_epochs": result.completed_epochs,
                "global_step": result.global_step,
                "initial_loss": result.initial_loss,
                "final_loss": result.final_loss,
                "gradient_norms": result.gradient_norms,
                "local_cpu_implementation": "PASS" if backend == "cpu" else "NOT_APPLICABLE",
                "gpu_validation": ("NOT RUN" if backend == "cpu" else "MEASURED_NOT_REVIEWED"),
                "scientific_acceptance": result.evaluation["scientific_acceptance"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
