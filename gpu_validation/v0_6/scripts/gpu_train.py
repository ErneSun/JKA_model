#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from train.train_v0_6 import train_v0_6  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-from-v0-5")
    parser.add_argument("--resume-from")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    if bool(args.resume_from) == bool(args.init_from_v0_5):
        parser.error("provide exactly one of --init-from-v0-5 or --resume-from")
    if args.resume_from:
        result = train_v0_6(
            args.config, device="cuda", resume_from=args.resume_from, run_name=args.run_name
        )
    else:
        result = train_v0_6(
            args.config, device="cuda", init_from_v0_5=args.init_from_v0_5, run_name=args.run_name
        )
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "checkpoint": str(result.best_checkpoint),
                "evaluation": result.evaluation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
