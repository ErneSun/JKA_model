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
        **{
            f"{horizon}_mass": bool(result["mass_gate_pass"][horizon])
            for horizon in ("short", "medium", "long")
        },
        **{
            f"{horizon}_operator": bool(result["operator_gate_pass"][horizon])
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
                    (
                        float(results[seed]["physics"]["forecast"][horizon][metric])
                        - float(results[seed]["no_physics"]["forecast"][horizon][metric])
                    )
                    / max(
                        abs(float(results[seed]["no_physics"]["forecast"][horizon][metric])),
                        1.0e-12,
                    )
                    for seed in seeds
                ]
            )
            for metric in ("rmse", "mass_drift", "operator")
        }
        for horizon in ("short", "medium", "long")
    }
    paired_skill_degradation = {
        horizon: _statistics(
            [
                (
                    float(results[seed]["physics"]["forecast"][horizon]["rmse"])
                    - float(results[seed]["no_physics"]["forecast"][horizon]["rmse"])
                )
                / max(
                    float(
                        results[seed]["physics"]["forecast"][horizon]["persistence_rmse"]
                    ),
                    1.0e-12,
                )
                for seed in seeds
            ]
        )
        for horizon in ("short", "medium", "long")
    }
    paired_constraint_degradation = {
        horizon: {
            metric: _statistics(
                [
                    (
                        float(results[seed]["physics"]["forecast"][horizon][metric])
                        - float(results[seed]["no_physics"]["forecast"][horizon][metric])
                    )
                    / float(
                        results[seed]["physics"][
                            "relative_mass_drift_threshold"
                            if metric == "mass_drift"
                            else "operator_mse_threshold"
                        ]
                    )
                    for seed in seeds
                ]
            )
            for metric in ("mass_drift", "operator")
        }
        for horizon in ("short", "medium", "long")
    }
    all_seed_gates = all(all(gates.values()) for gates in gates_by_seed.values())
    ablation_threshold = float(physics[0]["ablation_skill_degradation_threshold"])
    constraint_ablation_threshold = float(
        physics[0]["ablation_constraint_degradation_threshold"]
    )
    forecast_ablation_noninferior = all(
        paired_skill_degradation[horizon]["median"] <= ablation_threshold
        for horizon in ("short", "medium", "long")
    )
    constraint_ablation_noninferior = all(
        paired_constraint_degradation[horizon][metric]["median"]
        <= constraint_ablation_threshold
        for horizon in ("short", "medium", "long")
        for metric in ("mass_drift", "operator")
    )
    physics_ablation_noninferior = (
        forecast_ablation_noninferior and constraint_ablation_noninferior
    )
    scientific_status = (
        "PENDING_REVIEW" if all_seed_gates and physics_ablation_noninferior else "FAIL"
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
        "physics_ablation_consistent": physics_ablation_noninferior,
        "physics_ablation_noninferior": physics_ablation_noninferior,
        "ablation_skill_degradation_threshold": ablation_threshold,
        "ablation_constraint_degradation_threshold": constraint_ablation_threshold,
        "physics_vs_no_physics_skill_degradation": paired_skill_degradation,
        "physics_vs_no_physics_constraint_degradation": paired_constraint_degradation,
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
        f"- reused no-physics seed checkpoints: "
        f"`{report.get('baseline_reuse', {}).get('reused_seeds', [])}`",
        "",
        "## Per-seed hard gates",
        "",
        "| Seed | Frequency | Decay | Stability | Reconstruction | Forecast | Mass | Operator |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def mark(value: bool) -> str:
        return "PASS" if value else "FAIL"

    for seed, gates in report["gates_by_seed"].items():
        lines.append(
            f"| {seed} | {mark(gates['frequency'])} | {mark(gates['decay'])} | "
            f"{mark(gates['stability'])} | {mark(gates['reconstruction'])} | "
            f"{mark(all(gates[f'{h}_beats_persistence'] for h in ('short', 'medium', 'long')))} | "
            f"{mark(all(gates[f'{h}_mass'] for h in ('short', 'medium', 'long')))} | "
            f"{mark(all(gates[f'{h}_operator'] for h in ('short', 'medium', 'long')))} |"
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
    lines.extend(
        (
            "",
            "## Forecast non-inferiority vs no-physics",
            "",
            f"Threshold: {100 * report['ablation_skill_degradation_threshold']:.3f}% "
            "of persistence RMSE.",
            "",
            "| Horizon | Median added RMSE / persistence RMSE | Pass |",
            "|---|---:|---:|",
        )
    )
    for horizon, values in report["physics_vs_no_physics_skill_degradation"].items():
        lines.append(
            f"| {horizon} | {100 * values['median']:+.3f}% | "
            f"{mark(values['median'] <= report['ablation_skill_degradation_threshold'])} |"
        )
    lines.extend(
        (
            "",
            "## Constraint non-inferiority vs no-physics",
            "",
            f"Threshold: {100 * report['ablation_constraint_degradation_threshold']:.3f}% "
            "of each constraint hard limit.",
            "",
            "| Horizon | Added mass drift / limit | Added operator MSE / limit | Pass |",
            "|---|---:|---:|---:|",
        )
    )
    constraint_changes = report["physics_vs_no_physics_constraint_degradation"]
    constraint_threshold = report["ablation_constraint_degradation_threshold"]
    for horizon, values in constraint_changes.items():
        passed = all(values[metric]["median"] <= constraint_threshold for metric in values)
        lines.append(
            f"| {horizon} | {100 * values['mass_drift']['median']:+.3f}% | "
            f"{100 * values['operator']['median']:+.3f}% | {mark(passed)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-id", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[47, 53, 59])
    parser.add_argument(
        "--reuse-baseline-from",
        default=None,
        help="Reuse no-physics run checkpoints referenced by a prior result id when available",
    )
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
        "reuse_baseline_from": args.reuse_baseline_from,
        "steps": {},
    }
    if state.get("seeds") != seeds:
        raise ValueError("resume seeds differ from the original validation session")
    if state.get("reuse_baseline_from") != args.reuse_baseline_from:
        raise ValueError("resume baseline source differs from the original validation session")
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
    if args.reuse_baseline_from:
        continuation += f" --reuse-baseline-from {args.reuse_baseline_from}"
    reusable_baselines: dict[str, Any] = {}
    if args.reuse_baseline_from:
        prior_report_path = FINAL_RESULTS / args.reuse_baseline_from / "final_validation.json"
        if prior_report_path.is_file():
            reusable_baselines = _load_json(prior_report_path).get("runs", {})
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
        reused_baseline_seeds: list[int] = []
        for seed in seeds:
            results[seed] = {}
            for variant in ("physics", "no_physics"):
                reusable = None
                if variant == "no_physics":
                    reusable_value = reusable_baselines.get(str(seed), {}).get("no_physics")
                    reusable = None if reusable_value is None else Path(str(reusable_value))
                if (
                    reusable is not None
                    and (reusable / "checkpoints/best_forecast_post_warmup.pt").is_file()
                    and (reusable / "config/resolved_config.yaml").is_file()
                ):
                    run_dir = reusable
                    state.setdefault("steps", {})[f"reuse_no_physics_seed_{seed}"] = {
                        "status": "PASS",
                        "run_dir": str(run_dir.resolve()),
                        "source_validation_id": args.reuse_baseline_from,
                    }
                    _save_state(state_path, state)
                    reused_baseline_seeds.append(seed)
                    print(f"\n=== reusing no-physics seed {seed}: {run_dir} ===", flush=True)
                else:
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
        report["baseline_reuse"] = {
            "requested_result_id": args.reuse_baseline_from,
            "reused_seeds": reused_baseline_seeds,
            "retrained_no_physics_seeds": [
                seed for seed in seeds if seed not in reused_baseline_seeds
            ],
        }
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
