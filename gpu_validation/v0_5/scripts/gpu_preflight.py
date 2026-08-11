#!/usr/bin/env python3
"""Validate CUDA environment and seeded component-level CPU/GPU FP32 parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
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
        raise RuntimeError("CUDA is unavailable; GPU validation cannot be claimed")
    configs = {
        name: ROOT / "gpu_validation" / "v0_5" / "configs" / name
        for name in ("gpu_smoke.yaml", "gpu_full.yaml", "gpu_full_no_physics.yaml")
    }
    missing = [str(path) for path in configs.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"GPU validation config(s) missing: {missing}")
    cpu_case = prepare_v0_5_diagnostic_case(configs["gpu_smoke.yaml"], device="cpu")
    gpu_case = prepare_v0_5_diagnostic_case(
        configs["gpu_smoke.yaml"], device="cuda", state_dict=cpu_case.model.state_dict()
    )
    cpu = run_v0_5_diagnostic_step(cpu_case, backward_physics=False)
    gpu = run_v0_5_diagnostic_step(gpu_case, backward_physics=False)
    torch_tol = {"rtol": 2e-4, "atol": 2e-5}
    math_tol = {"rel_tol": 2e-4, "abs_tol": 2e-5}
    parity: dict[str, float] = {}
    for name in ("encoder_output", "koopman_step_output", "decoder_output"):
        parity[name] = float((cpu[name] - gpu[name]).abs().max())
        if not torch.allclose(cpu[name], gpu[name], **torch_tol):
            raise RuntimeError(f"CPU/GPU {name} parity failed: max_abs={parity[name]}")
    for name in ("mass_penalty", "operator_penalty"):
        parity[name] = abs(cpu[name] - gpu[name])
        if not math.isclose(cpu[name], gpu[name], **math_tol):
            raise RuntimeError(f"CPU/GPU {name} parity failed: abs={parity[name]}")
    current_device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(current_device)
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )
    report = {
        "status": "FAIL" if dirty else "PASS",
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_version": torch.__version__,
        "device_count": torch.cuda.device_count(),
        "current_device": current_device,
        "device": properties.name,
        "total_memory": properties.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "fp16_supported": True,
        "configs": {name: str(path) for name, path in configs.items()},
        "parity_tolerances": {"torch": torch_tol, "math": math_tol},
        "component_parity_max_abs": parity,
        "matrix_exp_device": str(torch.matrix_exp(torch.eye(2, device="cuda")).device),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / "preflight.json"
    summary_path = args.results_dir / "preflight.md"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(
        "# V0.5 GPU preflight\n\n"
        f"- status: **{report['status']}**\n"
        f"- commit: `{report['git_commit']}`; dirty: `{report['git_dirty']}`\n"
        f"- device: `{report['device']}`; count: {report['device_count']}\n"
        f"- CUDA / cuDNN / PyTorch: `{report['cuda_version']}` / "
        f"`{report['cudnn_version']}` / `{report['torch_version']}`\n"
        f"- BF16 supported: `{report['bf16_supported']}`\n"
        f"- component parity max abs: `{report['component_parity_max_abs']}`\n"
        f"- matrix_exp device: `{report['matrix_exp_device']}`\n",
        encoding="utf-8",
    )
    print("=== V0.5 GPU Preflight ===")
    print(json.dumps(report, indent=2, sort_keys=True))
    if dirty:
        raise RuntimeError("GPU preflight requires a clean Git worktree")


if __name__ == "__main__":
    main()
