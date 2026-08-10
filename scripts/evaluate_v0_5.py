#!/usr/bin/env python3
"""Thin CLI wrapper around canonical held-out V0.5 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.evaluate_v0_5 import evaluate_v0_5
from jka_model.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_forecast")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config_path = args.run_dir / "config" / "resolved_config.yaml"
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        checkpoint = args.run_dir / "checkpoints" / f"{args.checkpoint.removesuffix('.pt')}.pt"
    result = evaluate_v0_5(
        load_config(config_path), checkpoint=checkpoint, device=args.device, run_dir=args.run_dir
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
