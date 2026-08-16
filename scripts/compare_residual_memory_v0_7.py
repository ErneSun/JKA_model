#!/usr/bin/env python3
"""Compare already-trained V0.7 history sweeps; this script never trains models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from jka_model.residual import compare_residual_memory_v0_7  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare_residual_memory_v0_7(args.session_dir, args.results_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
