#!/usr/bin/env python3
"""Run canonical V0.5 GPU smoke in FP32 and capability-selected AMP."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from jka_model.config import load_config
from jka_model.evaluation.v0_5_diagnostics import (
    prepare_v0_5_diagnostic_case,
    run_v0_5_diagnostic_step,
)
from train.train_v0_5 import train_v0_5

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    config = load_config(ROOT / "gpu_validation/v0_5/configs/gpu_smoke.yaml")
    fp32 = train_v0_5(config, device="cuda")
    assert config.v0_5_training is not None
    amp_precision = "amp_bf16" if torch.cuda.is_bf16_supported() else "amp_fp16"
    amp_config = replace(
        config, v0_5_training=replace(config.v0_5_training, precision=amp_precision)
    )
    amp = train_v0_5(amp_config, device="cuda")
    physics_gradients = {}
    for name, run_config, result in (("fp32", config, fp32), ("amp", amp_config, amp)):
        diagnostic = run_v0_5_diagnostic_step(
            prepare_v0_5_diagnostic_case(run_config, device="cuda"), backward_physics=True
        )
        physics_gradients[name] = diagnostic["gradient_norms"]
        if not all(
            value > 0 and torch.isfinite(torch.tensor(value))
            for value in physics_gradients[name].values()
        ):
            raise RuntimeError(f"{name} isolated physics loss has invalid gradients")
        if not all(value > 0 for value in result.gradient_norms.values()):
            raise RuntimeError(f"{name} smoke has a missing encoder/decoder/A gradient")
        if not result.evaluation["finite"]:
            raise RuntimeError(f"{name} smoke produced non-finite evaluation")
    print(
        json.dumps(
            {
                "fp32_run": str(fp32.run_dir),
                "amp_run": str(amp.run_dir),
                "amp_precision": amp_precision,
                "isolated_physics_gradient_norms": physics_gradients,
                "status": "PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
