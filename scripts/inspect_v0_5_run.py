#!/usr/bin/env python3
"""Inspect standard V0.5 metadata, history, checkpoints, and evaluation records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.run_dir / "metadata" / "run_manifest.json").read_text())
    config_path = args.run_dir / "config" / "resolved_config.yaml"
    rows = list(csv.DictReader((args.run_dir / "logs" / "epoch_metrics.csv").open()))
    evaluation = json.loads((args.run_dir / "evaluation" / "metrics.json").read_text())
    checkpoints = sorted(path.name for path in (args.run_dir / "checkpoints").glob("*.pt"))
    best_forecast = min(rows, key=lambda row: float(row["val_rollout"])) if rows else None
    best_physics = (
        min(rows, key=lambda row: float(row["val_mass"]) + float(row["val_operator"]))
        if rows
        else None
    )
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir.resolve()),
                "config_path": str(config_path.resolve()),
                "config": config_path.read_text(),
                "metadata": metadata,
                "epochs_recorded": len(rows),
                "last_epoch": rows[-1] if rows else None,
                "best_forecast_epoch": (
                    None if best_forecast is None else int(best_forecast["epoch"])
                ),
                "best_physics_epoch": (
                    None if best_physics is None else int(best_physics["epoch"])
                ),
                "checkpoints": checkpoints,
                "evaluation": evaluation,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
