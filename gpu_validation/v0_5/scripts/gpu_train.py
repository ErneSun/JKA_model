#!/usr/bin/env python3
"""Thin GPU wrapper around src.train.train_v0_5.train_v0_5."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from jka_model.config import load_config
from train.train_v0_5 import train_v0_5

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "gpu_validation/v0_5/configs/gpu_full.yaml"
    )
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--precision", choices=("fp32", "amp_fp16", "amp_bf16"), default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.precision is not None:
        if config.v0_5_training is None:
            raise ValueError("GPU training requires a V0.5 training section")
        config = replace(
            config,
            v0_5_training=replace(config.v0_5_training, precision=args.precision),
        )
    result = train_v0_5(config, device="cuda", resume_from=args.resume_from)
    print(
        json.dumps(
            {"run_dir": str(result.run_dir), "checkpoint": str(result.latest_checkpoint)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
