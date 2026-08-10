#!/usr/bin/env python3
"""Compare uninterrupted and resumed deterministic FP32 GPU checkpoints exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jka_model.utils import load_checkpoint

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted-run", type=Path, required=True)
    parser.add_argument("--resumed-run", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "gpu_validation/v0_5/results",
        help="Directory for immutable resume-check artifacts",
    )
    args = parser.parse_args()
    left = load_checkpoint(args.uninterrupted_run / "checkpoints/latest.pt", map_location="cpu")
    right = load_checkpoint(args.resumed_run / "checkpoints/latest.pt", map_location="cpu")
    if left.online_model_state is None or right.online_model_state is None:
        raise RuntimeError("resume comparison requires both model states")
    names_match = left.online_model_state.keys() == right.online_model_state.keys()
    weights_match = names_match and all(
        torch.equal(left.online_model_state[name], right.online_model_state[name])
        for name in left.online_model_state
    )
    uninterrupted = args.uninterrupted_run.resolve()
    resumed = args.resumed_run.resolve()
    report = {
        "status": "PASS"
        if (
            weights_match
            and left.epoch == right.epoch
            and left.global_step == right.global_step
            and left.split_manifest == right.split_manifest
            and left.config_hash == right.config_hash
        )
        else "FAIL",
        "uninterrupted_run": str(uninterrupted),
        "uninterrupted_run_id": uninterrupted.name,
        "resumed_run": str(resumed),
        "resumed_run_id": resumed.name,
        "epoch": [left.epoch, right.epoch],
        "global_step": [left.global_step, right.global_step],
        "config_hash": [left.config_hash, right.config_hash],
        "config_hash_match": left.config_hash == right.config_hash,
        "split_manifest_match": left.split_manifest == right.split_manifest,
        "weights_bitwise_equal": weights_match,
    }
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"resume_check_{uninterrupted.name}_vs_{resumed.name}"
    metrics_path = results_dir / f"{stem}.json"
    summary_path = results_dir / f"{stem}.md"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(
        "# V0.5 GPU resume check\n\n"
        f"- status: **{report['status']}**\n"
        f"- uninterrupted: `{report['uninterrupted_run_id']}`\n"
        f"- resumed: `{report['resumed_run_id']}`\n"
        f"- epoch: {report['epoch'][0]} / {report['epoch'][1]}\n"
        f"- global_step: {report['global_step'][0]} / {report['global_step'][1]}\n"
        f"- config_hash_match: `{report['config_hash_match']}`\n"
        f"- split_manifest_match: `{report['split_manifest_match']}`\n"
        f"- weights_bitwise_equal: `{report['weights_bitwise_equal']}`\n"
        f"- artifacts: `{metrics_path.name}`, `{summary_path.name}`\n"
    )
    print(
        json.dumps(
            {
                **report,
                "metrics_path": str(metrics_path.resolve()),
                "summary_path": str(summary_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["status"] != "PASS":
        raise RuntimeError("deterministic GPU resume comparison failed")


if __name__ == "__main__":
    main()
