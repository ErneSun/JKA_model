#!/usr/bin/env python3
"""Run only the V0.5 GPU gates affected by the scientific-model revision."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .gpu_validate_all import (
        CONFIGS,
        FINAL_RESULTS,
        ROOT,
        RUN_ROOT,
        SCRIPTS,
        _choose_run_name,
        _load_json,
        _run_step,
        _save_state,
        _utc_now,
        _write_json,
    )
else:
    from gpu_validate_all import (
        CONFIGS,
        FINAL_RESULTS,
        ROOT,
        RUN_ROOT,
        SCRIPTS,
        _choose_run_name,
        _load_json,
        _run_step,
        _save_state,
        _utc_now,
        _write_json,
    )
from jka_model.config import load_config


def _default_validation_id() -> str:
    return "v05science-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _statistics(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _train(
    *,
    seed: int,
    variant: str,
    validation_id: str,
    state: dict[str, Any],
    state_path: Path,
    logs_dir: Path,
) -> Path:
    step = f"train_{variant}_seed_{seed}"
    run_name = _choose_run_name(f"{validation_id}-{variant}-seed{seed}", step, state)
    _save_state(state_path, state)
    config = CONFIGS / ("gpu_full.yaml" if variant == "physics" else "gpu_full_no_physics.yaml")
    _run_step(
        step,
        [
            sys.executable,
            str(SCRIPTS / "gpu_train.py"),
            "--config",
            str(config),
            "--run-name",
            run_name,
            "--seed",
            str(seed),
            "--no-epoch-checkpoints",
        ],
        state=state,
        state_path=state_path,
        logs_dir=logs_dir,
    )
    run_dir = RUN_ROOT / run_name
    manifest = _load_json(run_dir / "metadata/run_manifest.json")
    training = load_config(config).v0_5_training
    if training is None or manifest.get("completed_epochs") != training.epochs:
        raise RuntimeError(f"{step} did not complete all configured epochs")
    return run_dir


def _evaluate(
    *,
    seed: int,
    variant: str,
    run_dir: Path,
    artifacts_dir: Path,
    state: dict[str, Any],
    state_path: Path,
    logs_dir: Path,
) -> Path:
    step = f"evaluate_{variant}_seed_{seed}"
    _run_step(
        step,
        [
            sys.executable,
            str(SCRIPTS / "gpu_evaluate.py"),
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            "best_forecast_post_warmup",
            "--results-dir",
            str(artifacts_dir),
        ],
        state=state,
        state_path=state_path,
        logs_dir=logs_dir,
    )
    return artifacts_dir / f"{run_dir.name}_best_forecast_post_warmup_metrics.json"


def _seed_gates(result: dict[str, Any]) -> dict[str, bool]:
    return {
        "frequency": bool(result["frequency_gate_pass"]),
        "decay": bool(result["decay_gate_pass"]),
        "stability": bool(result["stability_gate_pass"]),
        "reconstruction": float(result["reconstruction_rmse"])
        < float(result["forecast"]["short"]["persistence_rmse"]),
        **{
            f"{horizon}_beats_persistence": bool(result["beats_persistence"][horizon])
            for horizon in ("short", "medium", "long")
        },
    }


def _build_report(
    *,
    validation_id: str,
    seeds: list[int],
    results: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    physics = [results[seed]["physics"] for seed in seeds]
    gates_by_seed = {str(seed): _seed_gates(results[seed]["physics"]) for seed in seeds}
    aggregate: dict[str, Any] = {
        metric: _statistics([float(item[metric]) for item in physics])
        for metric in (
            "frequency_relative_error",
            "decay_relative_error",
            "spectral_abscissa",
            "reconstruction_rmse",
        )
    }
    aggregate["forecast"] = {
        horizon: {
            metric: _statistics(
                [float(item["forecast"][horizon][metric]) for item in physics]
            )
            for metric in ("rmse", "relative_l2", "persistence_rmse", "mass_drift", "operator")
        }
        for horizon in ("short", "medium", "long")
    }
    paired_changes = {
        horizon: {
            metric: _statistics(
                [
                    float(results[seed]["physics"]["forecast"][horizon][metric])
                    / float(results[seed]["no_physics"]["forecast"][horizon][metric])
                    - 1.0
                    for seed in seeds
                ]
            )
            for metric in ("rmse", "mass_drift", "operator")
        }
        for horizon in ("short", "medium", "long")
    }
    all_seed_gates = all(all(gates.values()) for gates in gates_by_seed.values())
    physics_ablation_consistent = all(
        paired_changes[horizon][metric]["median"] <= 0
        for horizon in ("short", "medium", "long")
        for metric in ("rmse", "mass_drift", "operator")
    )
    scientific_status = (
        "PENDING_REVIEW" if all_seed_gates and physics_ablation_consistent else "FAIL"
    )
    return {
        "validation_id": validation_id,
        "generated_at": _utc_now(),
        "workflow_status": "PASS",
        "scientific_status": scientific_status,
        "overall_acceptance": (
            "PENDING_RESEARCHER_REVIEW"
            if scientific_status == "PENDING_REVIEW"
            else "NOT_ACCEPTED"
        ),
        "seeds": seeds,
        "scientific_checkpoint": "best_forecast_post_warmup",
        "gates_by_seed": gates_by_seed,
        "all_seed_gates_pass": all_seed_gates,
        "physics_ablation_consistent": physics_ablation_consistent,
        "physics_aggregate": aggregate,
        "physics_vs_no_physics_paired_relative_change": paired_changes,
        "runs": {
            str(seed): {
                variant: results[seed][variant]["run_dir"]
                for variant in ("physics", "no_physics")
            }
            for seed in seeds
        },
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    aggregate = report["physics_aggregate"]
    lines = [
        "# V0.5 incremental scientific GPU validation",
        "",
        f"- validation id: `{report['validation_id']}`",
        f"- workflow status: **{report['workflow_status']}**",
        f"- scientific status: **{report['scientific_status']}**",
        f"- overall acceptance: **{report['overall_acceptance']}**",
        f"- seeds: `{report['seeds']}`",
        "- reused prior evidence: CUDA preflight/parity, profiler, and exact-resume validation",
        "",
        "## Per-seed hard gates",
        "",
        "| Seed | Frequency | Decay | Stability | Reconstruction | Short | Medium | Long |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def mark(value: bool) -> str:
        return "PASS" if value else "FAIL"

    for seed, gates in report["gates_by_seed"].items():
        lines.append(
            f"| {seed} | {mark(gates['frequency'])} | {mark(gates['decay'])} | "
            f"{mark(gates['stability'])} | {mark(gates['reconstruction'])} | "
            f"{mark(gates['short_beats_persistence'])} | "
            f"{mark(gates['medium_beats_persistence'])} | "
            f"{mark(gates['long_beats_persistence'])} |"
        )
    lines.extend(
        (
            "",
            "## Physics aggregate",
            "",
            "| Metric | Mean | Median | Std |",
            "|---|---:|---:|---:|",
        )
    )
    for metric in (
        "frequency_relative_error",
        "decay_relative_error",
        "spectral_abscissa",
        "reconstruction_rmse",
    ):
        values = aggregate[metric]
        lines.append(
            f"| {metric} | {values['mean']:.6g} | {values['median']:.6g} | "
            f"{values['std']:.6g} |"
        )
    lines.extend(
        (
            "",
            "## Median physics vs no-physics relative change",
            "",
            "| Horizon | RMSE | Mass drift | Operator |",
            "|---|---:|---:|---:|",
        )
    )
    changes = report["physics_vs_no_physics_paired_relative_change"]
    for horizon in ("short", "medium", "long"):
        lines.append(
            f"| {horizon} | {100 * changes[horizon]['rmse']['median']:+.3f}% | "
            f"{100 * changes[horizon]['mass_drift']['median']:+.3f}% | "
            f"{100 * changes[horizon]['operator']['median']:+.3f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-id", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[47, 53, 59])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) < 3 or any(seed < 0 for seed in seeds):
        raise ValueError("scientific validation requires at least three unique non-negative seeds")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("scientific validation requires a clean tracked worktree")
    validation_id = args.validation_id or _default_validation_id()
    session_dir = RUN_ROOT / "validation_sessions" / validation_id
    artifacts_dir = session_dir / "artifacts"
    logs_dir = session_dir / "logs"
    state_path = session_dir / "state.json"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    state = _load_json(state_path) if state_path.is_file() else {
        "validation_id": validation_id,
        "created_at": _utc_now(),
        "python": sys.executable,
        "seeds": seeds,
        "steps": {},
    }
    if state.get("seeds") != seeds:
        raise ValueError("resume seeds differ from the original validation session")
    if state.get("workflow_status") == "PASS" and state.get("final_results"):
        completed_results = Path(str(state["final_results"]))
        final_report = completed_results / "final_validation.json"
        if final_report.is_file():
            completed = _load_json(final_report)
            print("=== V0.5 INCREMENTAL SCIENTIFIC VALIDATION ALREADY COMPLETE ===")
            print(f"workflow_status={completed['workflow_status']}")
            print(f"scientific_status={completed['scientific_status']}")
            print(f"overall_acceptance={completed['overall_acceptance']}")
            print(f"final_results={completed_results.resolve()}")
            return 0
    _save_state(state_path, state)
    continuation = (
        f"{sys.executable} {Path(__file__).resolve()} --validation-id {validation_id} "
        f"--seeds {' '.join(map(str, seeds))}"
    )
    try:
        _run_step(
            "model_smoke",
            [
                sys.executable,
                str(SCRIPTS / "gpu_smoke.py"),
                "--results-dir",
                str(artifacts_dir),
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
        )
        results: dict[int, dict[str, dict[str, Any]]] = {}
        for seed in seeds:
            results[seed] = {}
            for variant in ("physics", "no_physics"):
                run_dir = _train(
                    seed=seed,
                    variant=variant,
                    validation_id=validation_id,
                    state=state,
                    state_path=state_path,
                    logs_dir=logs_dir,
                )
                metrics_path = _evaluate(
                    seed=seed,
                    variant=variant,
                    run_dir=run_dir,
                    artifacts_dir=artifacts_dir,
                    state=state,
                    state_path=state_path,
                    logs_dir=logs_dir,
                )
                results[seed][variant] = _load_json(metrics_path)
        report = _build_report(validation_id=validation_id, seeds=seeds, results=results)
        _write_json(artifacts_dir / "final_validation.json", report)
        _write_markdown(artifacts_dir / "final_validation.md", report)
        destination = FINAL_RESULTS / validation_id
        if destination.exists():
            raise FileExistsError(f"final result directory already exists: {destination}")
        shutil.copytree(artifacts_dir, destination)
    except Exception as error:
        state["workflow_status"] = "FAIL"
        state["error"] = str(error)
        _save_state(state_path, state)
        print(f"\nSCIENTIFIC VALIDATION FAILED: {error}", file=sys.stderr)
        print(f"Preserved session: {session_dir.resolve()}", file=sys.stderr)
        print(f"Continue without repeating passed steps:\n{continuation}", file=sys.stderr)
        return 1
    state["workflow_status"] = "PASS"
    state["final_results"] = str(destination.resolve())
    _save_state(state_path, state)
    print("\n=== V0.5 INCREMENTAL SCIENTIFIC VALIDATION COMPLETE ===")
    print(json.dumps({key: report[key] for key in (
        "workflow_status", "scientific_status", "overall_acceptance"
    )}, indent=2))
    print(f"final_results={destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
