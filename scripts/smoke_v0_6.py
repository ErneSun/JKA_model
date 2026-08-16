#!/usr/bin/env python3
"""End-to-end V0.5 -> V0.6 -> checkpoint -> evaluation smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.train_v0_5 import train_v0_5
from train.train_v0_6 import train_v0_6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0_6/advection_diffusion_2d_cpu_smoke.yaml")
    parser.add_argument("--v0-5-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    source = args.v0_5_checkpoint
    bootstrap_run = None
    if source is None:
        # Software-path bootstrap only. Formal science must pass a separately validated
        # V0.5 checkpoint with the same model/data contract.
        baseline = train_v0_5(args.config, device=args.device)
        source = baseline.best_checkpoint
        bootstrap_run = str(baseline.run_dir)
    result = train_v0_6(args.config, device=args.device, init_from_v0_5=source)
    reloaded = result.evaluation
    print(
        json.dumps(
            {
                "status": "PASS",
                "bootstrap_v0_5_run": bootstrap_run,
                "v0_5_checkpoint": str(source),
                "v0_6_run": str(result.run_dir),
                "checkpoint": str(result.latest_checkpoint),
                "optimizer_update_step": result.optimizer_update_step,
                "target_used_for_rollout": reloaded["target_used_for_rollout"],
                "collapse_gate": reloaded["collapse_gate"],
                "scientific_acceptance": "PENDING_GPU",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
