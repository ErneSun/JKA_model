#!/usr/bin/env python3
"""CPU-only end-to-end V0.5 implementation smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.train_v0_5 import train_v0_5


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=root / "configs/v0_5/advection_diffusion_2d_cpu_smoke.yaml"
    )
    args = parser.parse_args()
    result = train_v0_5(args.config, device="cpu")
    checks = {
        "checkpoint": result.latest_checkpoint.is_file(),
        "history": (result.run_dir / "logs/epoch_metrics.csv").is_file(),
        "evaluation": (result.run_dir / "evaluation/metrics.json").is_file(),
        "encoder_gradient": result.gradient_norms["encoder"] > 0,
        "decoder_gradient": result.gradient_norms["decoder"] > 0,
        "generator_gradient": result.gradient_norms["generator"] > 0,
        "finite": bool(result.evaluation["finite"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V0.5 smoke failed: {checks}")
    print(
        json.dumps(
            {
                "V0.5 LOCAL CPU IMPLEMENTATION": "PASS",
                "V0.5 GPU VALIDATION": "NOT RUN",
                "V0.5 SCIENTIFIC ACCEPTANCE": "PENDING_GPU",
                "run_dir": str(result.run_dir),
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
