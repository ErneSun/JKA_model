#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for item in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(item))

from train.train_v0_7 import train_v0_7  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    train_v0_7(
        args.config,
        backbone_checkpoint=args.backbone_checkpoint,
        cache_path=args.cache,
        variant=args.variant,
        run_dir=args.run_dir,
        device="cuda",
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
