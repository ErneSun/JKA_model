#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eval.evaluate_v0_6 import evaluate_v0_6  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_v0_6(
                args.config, checkpoint=args.checkpoint, device="cuda", run_dir=args.run_dir
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
