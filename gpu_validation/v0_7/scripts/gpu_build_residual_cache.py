#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for item in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(item))

from train.prepare_v0_7 import prepare_v0_7_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    args = parser.parse_args()
    prepare_v0_7_cache(
        args.config,
        backbone_checkpoint=args.backbone_checkpoint,
        destination=args.cache,
        diagnostics_path=args.diagnostics,
        device="cuda",
    )


if __name__ == "__main__":
    main()
