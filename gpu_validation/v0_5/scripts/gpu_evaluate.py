#!/usr/bin/env python3
"""Thin GPU wrapper around src.eval.evaluate_v0_5.evaluate_v0_5."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from eval.evaluate_v0_5 import evaluate_v0_5

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_forecast")
    args = parser.parse_args()
    result = evaluate_v0_5(
        args.run_dir / "config/resolved_config.yaml",
        checkpoint=args.run_dir / "checkpoints" / f"{args.checkpoint.removesuffix('.pt')}.pt",
        device="cuda",
        run_dir=args.run_dir,
    )
    manifest = json.loads((args.run_dir / "metadata/run_manifest.json").read_text())
    with (args.run_dir / "logs/epoch_metrics.csv").open(encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if not history:
        raise RuntimeError("GPU run has no epoch metrics")
    best = min(history, key=lambda row: float(row["val_rollout"]))
    peak_memory = max(float(row["peak_gpu_memory"]) for row in history)
    samples_per_second = max(float(row["samples_per_sec"]) for row in history)
    technical_pass = bool(result["finite"]) and all(
        math.isfinite(float(value))
        for horizon in result["rollout"].values()
        for value in horizon.values()
    )
    summary = {
        "status": "PASS" if technical_pass else "FAIL",
        "scientific_acceptance": "PENDING_REVIEW",
        "run_id": args.run_dir.name,
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": result["checkpoint"],
        "git_commit": manifest.get("git_commit"),
        "git_branch": manifest.get("git_branch"),
        "git_dirty": manifest.get("git_dirty"),
        "gpu": manifest.get("environment", {}).get("gpu_model"),
        "precision": manifest.get("precision"),
        "config": str((args.run_dir / "config/resolved_config.yaml").resolve()),
        "config_hash": manifest.get("config_hash"),
        "best_epoch": int(best["epoch"]),
        "forecast": result["rollout"],
        "persistence_long_rmse": result["rollout"]["long"]["persistence_rmse"],
        "frequency_relative_error": result["frequency_relative_error"],
        "decay_relative_error": result["decay_relative_error"],
        "mass_drift_long": result["rollout"]["long"]["mass_drift"],
        "operator_long": result["rollout"]["long"]["operator"],
        "peak_gpu_memory_bytes": peak_memory,
        "max_samples_per_second": samples_per_second,
    }
    results_dir = ROOT / "gpu_validation/v0_5/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"{args.run_dir.name}_metrics.json"
    summary_path = results_dir / f"{args.run_dir.name}_summary.md"
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(
        "# V0.5 GPU result summary\n\n"
        f"- technical status: **{summary['status']}**\n"
        "- scientific acceptance: **PENDING_REVIEW**\n"
        f"- run: `{summary['run_id']}`\n"
        f"- commit: `{summary['git_commit']}`; dirty: `{summary['git_dirty']}`\n"
        f"- GPU / precision: `{summary['gpu']}` / `{summary['precision']}`\n"
        f"- best epoch: {summary['best_epoch']}\n"
        f"- long model / persistence RMSE: {result['rollout']['long']['rmse']:.6g} / "
        f"{summary['persistence_long_rmse']:.6g}\n"
        f"- frequency / decay relative error: {summary['frequency_relative_error']:.6g} / "
        f"{summary['decay_relative_error']:.6g}\n"
        f"- long mass drift / operator: {summary['mass_drift_long']:.6g} / "
        f"{summary['operator_long']:.6g}\n"
        f"- peak VRAM bytes / max samples s^-1: {peak_memory:.0f} / {samples_per_second:.6g}\n"
    )
    if not technical_pass:
        raise RuntimeError("GPU evaluation produced non-finite technical metrics")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
