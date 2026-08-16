#!/usr/bin/env python3
"""One-command fresh V0.6 GPU software and matched scientific validation."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gpu_validation.v0_6.scripts.gpu_compare import compare  # noqa: E402
from gpu_validation.v0_6.scripts.gpu_report import build_review, write_bundle  # noqa: E402
from jka_model.config import ProjectConfig, load_config, save_config  # noqa: E402
from train.train_v0_6 import load_v0_5_initialization, train_v0_6  # noqa: E402


def _payload(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return value


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _seed_of_checkpoint(path: Path) -> int | None:
    config = _payload(path).get("config")
    if not isinstance(config, dict) or not isinstance(config.get("training"), dict):
        return None
    return int(config["training"]["seed"])


def discover_v0_5_checkpoints(root: Path, configs: dict[int, ProjectConfig]) -> dict[int, Path]:
    candidates: dict[int, list[Path]] = {seed: [] for seed in configs}
    for path in root.glob("*/checkpoints/best_forecast_post_warmup.pt"):
        seed = _seed_of_checkpoint(path)
        if seed not in candidates:
            continue
        try:
            load_v0_5_initialization(path, configs[seed])
        except ValueError:
            # Reject no-physics and any other scientifically unmatched checkpoint.
            continue
        candidates[seed].append(path)
    missing = [seed for seed, paths in candidates.items() if not paths]
    if missing:
        raise ValueError(f"no validated V0.5 checkpoint found for seeds {missing} below {root}")
    return {
        seed: max(paths, key=lambda path: path.stat().st_mtime)
        for seed, paths in candidates.items()
    }


def _resolved(template: ProjectConfig, seed: int, enabled: bool) -> ProjectConfig:
    data = template.to_dict()
    data["training"]["seed"] = seed
    data["data"]["split"]["seed"] = seed
    data["jepa_loss"] = {
        "lambda_one_step": 1.0 if enabled else 0.0,
        "lambda_multi_step": 1.0 if enabled else 0.0,
    }
    data["tags"] = ["v0.6", "gpu-full", "jepa" if enabled else "no-jepa-control", f"seed-{seed}"]
    return ProjectConfig.from_dict(data)


def _assert_matched(control: ProjectConfig, jepa: ProjectConfig) -> None:
    left, right = control.to_dict(), jepa.to_dict()
    for value in (left, right):
        value.pop("jepa_loss")
        value.pop("tags")
    if left != right:
        raise ValueError("matched configs differ outside JEPA objective and descriptive tags")


def _run_checked(command: list[str], log: Path, *, label: str) -> None:
    print(f"[V0.6][validation] {label}: START", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
    if returncode:
        print(f"[V0.6][validation] {label}: FAIL (log={log})", flush=True)
        raise RuntimeError(f"step failed ({returncode}); log={log}")
    print(f"[V0.6][validation] {label}: PASS (log={log})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-id", default=datetime.now(timezone.utc).strftime("v06-%Y%m%dT%H%M%SZ")
    )
    parser.add_argument("--v0-5-root", type=Path, default=Path("runs/v0_5/gpu"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[47, 53, 59])
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) < 3:
        raise SystemExit("scientific V0.6 validation requires at least three unique seeds")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    session = ROOT / "runs" / "v0_6" / "gpu" / "validation_sessions" / args.validation_id
    compact = ROOT / "gpu_validation" / "v0_6" / "results" / args.validation_id
    if session.exists():
        raise SystemExit(f"validation id already exists; choose a fresh id: {session}")
    if compact.exists():
        raise SystemExit(f"compact result id already exists; choose a fresh id: {compact}")
    for name in ("logs", "configs", "artifacts"):
        (session / name).mkdir(parents=True, exist_ok=True)
    python = str(ROOT / ".venv" / "bin" / "python")
    if not Path(python).is_file():
        python = "python3"
    if not args.skip_pytest:
        _run_checked(
            [python, "-m", "pytest", "-q"],
            session / "logs" / "pytest.log",
            label="pytest",
        )
    _run_checked(
        [python, "gpu_validation/v0_6/scripts/gpu_preflight.py"],
        session / "logs" / "preflight.log",
        label="CUDA preflight",
    )
    if not args.skip_smoke:
        _run_checked(
            [python, "scripts/smoke_v0_6.py", "--device", "cuda"],
            session / "logs" / "smoke_fp32.log",
            label="FP32 smoke",
        )
        smoke = load_config(ROOT / "gpu_validation/v0_6/configs/gpu_smoke.yaml").to_dict()
        smoke["v0_5_training"]["precision"] = "amp_fp16"
        amp_path = session / "configs" / "gpu_smoke_amp.yaml"
        save_config(ProjectConfig.from_dict(smoke), amp_path)
        _run_checked(
            [python, "scripts/smoke_v0_6.py", "--device", "cuda", "--config", str(amp_path)],
            session / "logs" / "smoke_amp.log",
            label="AMP smoke",
        )
    template = load_config(ROOT / "gpu_validation/v0_6/configs/gpu_jepa_multiseed.yaml")
    jepa_configs = {seed: _resolved(template, seed, True) for seed in args.seeds}
    checkpoints = discover_v0_5_checkpoints((ROOT / args.v0_5_root).resolve(), jepa_configs)
    comparisons: dict[str, Any] = {}
    runs: dict[str, Any] = {}
    for seed in args.seeds:
        control = _resolved(template, seed, False)
        jepa = jepa_configs[seed]
        _assert_matched(control, jepa)
        control_path = session / "configs" / f"seed_{seed}_control.yaml"
        jepa_path = session / "configs" / f"seed_{seed}_jepa.yaml"
        save_config(control, control_path)
        save_config(jepa, jepa_path)
        source = checkpoints[seed]
        print(
            f"[V0.6][validation] seed={seed} control training: START "
            f"(V0.5 checkpoint={source})",
            flush=True,
        )
        control_result = train_v0_6(
            control,
            device="cuda",
            init_from_v0_5=source,
            run_name=f"{args.validation_id}-control-seed{seed}",
        )
        print(
            f"[V0.6][validation] seed={seed} control training: PASS "
            f"(run={control_result.run_dir})",
            flush=True,
        )
        print(f"[V0.6][validation] seed={seed} JEPA training: START", flush=True)
        jepa_result = train_v0_6(
            jepa,
            device="cuda",
            init_from_v0_5=source,
            run_name=f"{args.validation_id}-jepa-seed{seed}",
        )
        print(
            f"[V0.6][validation] seed={seed} JEPA training: PASS "
            f"(run={jepa_result.run_dir})",
            flush=True,
        )
        print(f"[V0.6][validation] seed={seed} comparison: START", flush=True)
        assert jepa.v0_6_evaluation is not None
        comparison = compare(
            control_result.evaluation,
            jepa_result.evaluation,
            jepa.v0_6_evaluation.max_long_rollout_degradation,
            jepa.v0_6_evaluation.max_physics_degradation,
        )
        comparisons[str(seed)] = comparison
        runs[str(seed)] = {
            "v0_5_checkpoint": _portable_path(source),
            "control": _portable_path(control_result.run_dir),
            "jepa": _portable_path(jepa_result.run_dir),
        }
        (session / "artifacts" / f"comparison_seed_{seed}.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"[V0.6][validation] seed={seed} comparison: "
            f"{'PASS' if comparison['pass'] else 'FAIL'}",
            flush=True,
        )
    ratios = [comparisons[str(seed)]["ratios"]["long_rmse"] for seed in args.seeds]
    automated_pass = all(item["pass"] for item in comparisons.values())
    summary = {
        "validation_id": args.validation_id,
        "seeds": args.seeds,
        "runs": runs,
        "comparisons": comparisons,
        "long_rmse_ratio_mean": statistics.mean(ratios),
        "long_rmse_ratio_std": statistics.stdev(ratios),
        "implementation": "PASS",
        "automated_gates": "PASS" if automated_pass else "FAIL",
        "gpu_validation": "MEASURED_PASS_PENDING_REVIEW" if automated_pass else "FAIL",
        "scientific_acceptance": "PENDING_REVIEW" if automated_pass else "FAIL",
    }
    (session / "artifacts" / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compact.mkdir(parents=True, exist_ok=False)
    (compact / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    automated_review = build_review(
        args.validation_id,
        args.seeds,
        ROOT / "runs" / "v0_6" / "gpu",
        reviewed=False,
    )
    write_bundle(compact, automated_review)
    print(json.dumps(summary, indent=2))
    if not automated_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
