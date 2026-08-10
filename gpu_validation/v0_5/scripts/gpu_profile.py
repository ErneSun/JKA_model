#!/usr/bin/env python3
"""Profile three bounded V0.5 diagnostic steps and export a Chrome trace."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from jka_model.evaluation.v0_5_diagnostics import (
    prepare_v0_5_diagnostic_case,
    run_v0_5_diagnostic_step,
)

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    case = prepare_v0_5_diagnostic_case(
        ROOT / "gpu_validation/v0_5/configs/gpu_smoke.yaml", device="cuda"
    )
    output = (
        ROOT
        / "runs"
        / "v0_5"
        / "gpu"
        / ("profile_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    )
    output.mkdir(parents=True, exist_ok=False)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as profile:
        for _ in range(3):
            run_v0_5_diagnostic_step(case, backward_physics=True)
            profile.step()
    trace = output / "trace.json"
    profile.export_chrome_trace(str(trace))
    summary = profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
    (output / "summary.txt").write_text(summary + "\n")
    print(json.dumps({"steps": 3, "output_dir": str(output), "trace": str(trace)}, indent=2))


if __name__ == "__main__":
    main()
