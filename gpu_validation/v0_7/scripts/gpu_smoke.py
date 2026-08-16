#!/usr/bin/env python3
"""Thin GPU smoke: cache, zero baseline, and one history closure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for item in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(item))

from train.prepare_v0_7 import prepare_v0_7_cache  # noqa: E402
from train.train_v0_7 import train_v0_7  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    cache = args.run_dir / "cache" / "residual_cache.pt"
    prepare_v0_7_cache(
        args.config,
        backbone_checkpoint=args.backbone_checkpoint,
        destination=cache,
        diagnostics_path=args.run_dir / "cache" / "diagnostics.json",
        device="cuda",
    )
    for variant in ("zero", "history"):
        train_v0_7(
            args.config,
            backbone_checkpoint=args.backbone_checkpoint,
            cache_path=cache,
            variant=variant,
            run_dir=args.run_dir / "runs" / variant,
            device="cuda",
        )


if __name__ == "__main__":
    main()
