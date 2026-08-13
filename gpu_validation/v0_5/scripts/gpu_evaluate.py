#!/usr/bin/env python3
"""Thin GPU wrapper around src.eval.evaluate_v0_5.evaluate_v0_5."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from eval.evaluate_v0_5 import evaluate_v0_5
from jka_model.config import load_config

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_forecast")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "gpu_validation/v0_5/results",
    )
    args = parser.parse_args()
    checkpoint_name = args.checkpoint.removesuffix(".pt")
    resolved = load_config(args.run_dir / "config/resolved_config.yaml")
    if resolved.field_loss is None:
        raise ValueError("V0.5 GPU evaluation requires field_loss configuration")
    result = evaluate_v0_5(
        resolved,
        checkpoint=args.run_dir / "checkpoints" / f"{checkpoint_name}.pt",
        device="cuda",
        run_dir=args.run_dir,
    )
    manifest = json.loads((args.run_dir / "metadata/run_manifest.json").read_text())
    with (args.run_dir / "logs/epoch_metrics.csv").open(encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if not history:
        raise RuntimeError("GPU run has no epoch metrics")
    warmup_epochs = (
        0 if resolved.v0_5_training is None else resolved.v0_5_training.physics_warmup_epochs
    )
    post_warmup_history = [row for row in history if int(row["epoch"]) >= warmup_epochs + 1]
    if checkpoint_name == "best_forecast":
        selected = min(history, key=lambda row: float(row["val_rollout"]))
        selection_rule = "minimum validation forecast MSE"
    elif checkpoint_name == "best_forecast_post_warmup":
        selected = min(post_warmup_history, key=lambda row: float(row["val_rollout"]))
        selection_rule = "minimum validation forecast MSE after physics warmup"
    elif checkpoint_name == "best_physics":
        selected = min(
            history,
            key=lambda row: resolved.field_loss.lambda_mass * float(row["val_mass"])
            + resolved.field_loss.lambda_operator * float(row["val_operator"]),
        )
        selection_rule = "minimum weighted validation physics objective"
    elif checkpoint_name == "best_physics_post_warmup":
        selected = min(
            post_warmup_history,
            key=lambda row: resolved.field_loss.lambda_mass * float(row["val_mass"])
            + resolved.field_loss.lambda_operator * float(row["val_operator"]),
        )
        selection_rule = "minimum weighted validation physics objective after physics warmup"
    elif checkpoint_name in {"last", "latest"}:
        selected = history[-1]
        selection_rule = "final epoch"
    else:
        selected = history[-1]
        selection_rule = "explicit checkpoint"
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
        "checkpoint_name": checkpoint_name,
        "config": str((args.run_dir / "config/resolved_config.yaml").resolve()),
        "config_hash": manifest.get("config_hash"),
        "selection_epoch": int(selected["epoch"]),
        "selection_rule": selection_rule,
        "forecast": result["rollout"],
        "persistence_long_rmse": result["rollout"]["long"]["persistence_rmse"],
        "frequency_relative_error": result["frequency_relative_error"],
        "decay_relative_error": result["decay_relative_error"],
        "mass_drift_long": result["rollout"]["long"]["mass_drift"],
        "operator_long": result["rollout"]["long"]["operator"],
        "frequency_threshold": result["frequency_threshold"],
        "frequency_gate_pass": result["frequency_relative_error"]
        <= result["frequency_threshold"],
        "decay_threshold": result["decay_threshold"],
        "decay_gate_pass": result["decay_relative_error"] <= result["decay_threshold"],
        "spectral_abscissa": result["spectral_abscissa"],
        "spectral_abscissa_threshold": result["spectral_abscissa_threshold"],
        "stability_gate_pass": result["spectral_abscissa"]
        <= result["spectral_abscissa_threshold"],
        "beats_persistence": {
            horizon: values["rmse"] < values["persistence_rmse"]
            for horizon, values in result["rollout"].items()
        },
        "long_beats_persistence": result["rollout"]["long"]["rmse"]
        < result["rollout"]["long"]["persistence_rmse"],
        "reconstruction_rmse": result["reconstruction_rmse"],
        "latent_min_std": result["latent_min_std"],
        "latent_max_std": result["latent_max_std"],
        "learned_angular_frequency": result["learned_angular_frequency"],
        "true_angular_frequency": result["true_angular_frequency"],
        "learned_decay_rate": result["learned_decay_rate"],
        "true_decay_rate": result["true_decay_rate"],
        "peak_gpu_memory_bytes": peak_memory,
        "max_samples_per_second": samples_per_second,
    }
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = args.run_dir.name
    if checkpoint_name != "best_forecast":
        stem += f"_{checkpoint_name}"
    metrics_path = results_dir / f"{stem}_metrics.json"
    summary_path = results_dir / f"{stem}_summary.md"
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(
        "# V0.5 GPU result summary\n\n"
        f"- technical status: **{summary['status']}**\n"
        "- scientific acceptance: **PENDING_REVIEW**\n"
        f"- run: `{summary['run_id']}`\n"
        f"- commit: `{summary['git_commit']}`; dirty: `{summary['git_dirty']}`\n"
        f"- GPU / precision: `{summary['gpu']}` / `{summary['precision']}`\n"
        f"- checkpoint / selection epoch: `{checkpoint_name}` / "
        f"{summary['selection_epoch']}\n"
        f"- selection rule: {selection_rule}\n"
        f"- long model / persistence RMSE: {result['rollout']['long']['rmse']:.6g} / "
        f"{summary['persistence_long_rmse']:.6g}\n"
        f"- frequency / decay relative error: {summary['frequency_relative_error']:.6g} / "
        f"{summary['decay_relative_error']:.6g}\n"
        f"- frequency hard gate: **{'PASS' if summary['frequency_gate_pass'] else 'FAIL'}** "
        f"(threshold {summary['frequency_threshold']:.6g})\n"
        f"- decay hard gate: **{'PASS' if summary['decay_gate_pass'] else 'FAIL'}** "
        f"(threshold {summary['decay_threshold']:.6g})\n"
        f"- stability hard gate: **{'PASS' if summary['stability_gate_pass'] else 'FAIL'}**; "
        f"spectral abscissa {summary['spectral_abscissa']:.6g}\n"
        f"- beats persistence by horizon: {summary['beats_persistence']}\n"
        f"- long mass drift / operator: {summary['mass_drift_long']:.6g} / "
        f"{summary['operator_long']:.6g}\n"
        f"- peak VRAM bytes / max samples s^-1: {peak_memory:.0f} / {samples_per_second:.6g}\n"
    )
    if not technical_pass:
        raise RuntimeError("GPU evaluation produced non-finite technical metrics")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
