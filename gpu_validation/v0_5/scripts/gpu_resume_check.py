#!/usr/bin/env python3
"""Compare uninterrupted and resumed deterministic FP32 GPU checkpoints exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jka_model.utils import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted-run", type=Path, required=True)
    parser.add_argument("--resumed-run", type=Path, required=True)
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
        "uninterrupted_run": str(args.uninterrupted_run.resolve()),
        "resumed_run": str(args.resumed_run.resolve()),
        "epoch": [left.epoch, right.epoch],
        "global_step": [left.global_step, right.global_step],
        "config_hash_match": left.config_hash == right.config_hash,
        "split_manifest_match": left.split_manifest == right.split_manifest,
        "weights_bitwise_equal": weights_match,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise RuntimeError("deterministic GPU resume comparison failed")


if __name__ == "__main__":
    main()
