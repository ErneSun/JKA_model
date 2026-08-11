#!/usr/bin/env python3
"""Profile three bounded V0.5 diagnostic steps and export a Chrome trace."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from jka_model.evaluation.v0_5_diagnostics import (
    prepare_v0_5_diagnostic_case,
    run_v0_5_diagnostic_step,
)

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "gpu_validation/v0_5/results",
    )
    args = parser.parse_args()
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
    summary_path = output / "summary.txt"
    summary_path.write_text(summary + "\n")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    report = {
        "status": "PASS",
        "steps": 3,
        "output_dir": str(output.resolve()),
        "trace": str(trace.resolve()),
        "trace_bytes": trace.stat().st_size,
        "summary": str(summary_path.resolve()),
        "git_commit": commit,
        "device": torch.cuda.get_device_properties(torch.cuda.current_device()).name,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_path, args.results_dir / "profile_summary.txt")
    (args.results_dir / "profile.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.results_dir / "profile.md").write_text(
        "# V0.5 bounded GPU profile\n\n"
        f"- status: **{report['status']}**\n"
        f"- steps: {report['steps']}\n"
        f"- commit / device: `{commit}` / `{report['device']}`\n"
        f"- trace bytes: {report['trace_bytes']}\n"
        f"- output: `{report['output_dir']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
