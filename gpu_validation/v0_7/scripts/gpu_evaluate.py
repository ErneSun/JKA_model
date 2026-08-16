#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for item in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(item))

from eval.evaluate_v0_7 import evaluate_v0_7  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_v0_7(
        args.config,
        checkpoint=args.checkpoint,
        cache_path=args.cache,
        device="cuda",
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
