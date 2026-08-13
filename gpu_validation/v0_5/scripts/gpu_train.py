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
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint-epoch",
        type=int,
        action="append",
        default=None,
        help="Retain only the requested numbered epoch checkpoint(s); may be repeated",
    )
    checkpoint_group.add_argument(
        "--no-epoch-checkpoints",
        action="store_true",
        help="Retain latest/last/best checkpoints but no numbered epoch checkpoints",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("seed must be non-negative")
        config = replace(
            config,
            training=replace(config.training, seed=args.seed),
            data=replace(config.data, split=replace(config.data.split, seed=args.seed)),
        )
    if args.precision is not None:
        if config.v0_5_training is None:
            raise ValueError("GPU training requires a V0.5 training section")
        config = replace(
            config,
            v0_5_training=replace(config.v0_5_training, precision=args.precision),
        )
    checkpoint_epochs = (
        set()
        if args.no_epoch_checkpoints
        else None
        if args.checkpoint_epoch is None
        else set(args.checkpoint_epoch)
    )
    if checkpoint_epochs is not None and any(epoch <= 0 for epoch in checkpoint_epochs):
        raise ValueError("checkpoint epochs must be positive")
    result = train_v0_5(
        config,
        device="cuda",
        resume_from=args.resume_from,
        run_name=args.run_name,
        checkpoint_epochs=checkpoint_epochs,
    )
    print(
        json.dumps(
            {"run_dir": str(result.run_dir), "checkpoint": str(result.latest_checkpoint)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
