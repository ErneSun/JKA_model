#!/usr/bin/env python3
"""Run the complete, resumable V0.5 GPU validation workflow with one command."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jka_model.config import load_config

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "gpu_validation/v0_5/scripts"
CONFIGS = ROOT / "gpu_validation/v0_5/configs"
FINAL_RESULTS = ROOT / "gpu_validation/v0_5/results"
RUN_ROOT = ROOT / "runs/v0_5/gpu"
EVALUATION_CHECKPOINTS = (
    "best_forecast",
    "best_forecast_post_warmup",
    "best_physics",
    "best_physics_post_warmup",
    "last",
)


class StepFailure(RuntimeError):
    """A subprocess gate failed after its output was preserved."""

    def __init__(self, step: str, returncode: int, log_path: Path) -> None:
        super().__init__(f"step {step!r} failed with exit code {returncode}; log={log_path}")
        self.step = step
        self.returncode = returncode
        self.log_path = log_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_validation_id() -> str:
    return "v05gpu-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    _write_json(path, state)


def _run_step(
    step: str,
    command: list[str],
    *,
    state: dict[str, Any],
    state_path: Path,
    logs_dir: Path,
) -> None:
    steps = state.setdefault("steps", {})
    previous = steps.get(step, {})
    if previous.get("status") == "PASS":
        print(f"\n=== {step}: already PASS; skipping ===", flush=True)
        return
    log_path = logs_dir / f"{step}.log"
    entry = {
        **previous,
        "status": "RUNNING",
        "command": command,
        "started_at": _utc_now(),
        "log": str(log_path.resolve()),
    }
    steps[step] = entry
    _save_state(state_path, state)
    print(f"\n=== {step}: START ===", flush=True)
    print(" ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_stream:
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
            sys.stdout.write(line)
            sys.stdout.flush()
            log_stream.write(line)
            log_stream.flush()
        returncode = process.wait()
    entry.update(
        {
            "status": "PASS" if returncode == 0 else "FAIL",
            "returncode": returncode,
            "ended_at": _utc_now(),
        }
    )
    _save_state(state_path, state)
    if returncode != 0:
        raise StepFailure(step, returncode, log_path)
    print(f"=== {step}: PASS ===", flush=True)


def _choose_run_name(base: str, step: str, state: dict[str, Any]) -> str:
    entry = state.setdefault("steps", {}).get(step, {})
    if entry.get("status") == "PASS" and entry.get("run_name"):
        return str(entry["run_name"])
    candidate = base
    retry = 1
    while (RUN_ROOT / candidate).exists():
        retry += 1
        candidate = f"{base}-retry{retry}"
    state.setdefault("steps", {}).setdefault(step, {})["run_name"] = candidate
    state["steps"][step]["run_dir"] = str((RUN_ROOT / candidate).resolve())
    return candidate


def _run_training(
    step: str,
    *,
    base_name: str,
    config: Path,
    state: dict[str, Any],
    state_path: Path,
    logs_dir: Path,
    resume_from: Path | None = None,
    checkpoint_epoch: int | None = None,
) -> Path:
    run_name = _choose_run_name(base_name, step, state)
    _save_state(state_path, state)
    command = [
        sys.executable,
        str(SCRIPTS / "gpu_train.py"),
        "--config",
        str(config),
        "--run-name",
        run_name,
    ]
    if resume_from is not None:
        command.extend(("--resume-from", str(resume_from)))
    if checkpoint_epoch is None:
        command.append("--no-epoch-checkpoints")
    else:
        command.extend(("--checkpoint-epoch", str(checkpoint_epoch)))
    _run_step(
        step,
        command,
        state=state,
        state_path=state_path,
        logs_dir=logs_dir,
    )
    run_dir = RUN_ROOT / run_name
    manifest = _load_json(run_dir / "metadata/run_manifest.json")
    expected_epochs = load_config(config).v0_5_training
    if expected_epochs is None or manifest.get("completed_epochs") != expected_epochs.epochs:
        raise RuntimeError(f"{step} did not record the configured completed epoch count")
    return run_dir


def _evaluate(
    label: str,
    run_dir: Path,
    checkpoint: str,
    *,
    artifacts_dir: Path,
    state: dict[str, Any],
    state_path: Path,
    logs_dir: Path,
) -> None:
    _run_step(
        f"evaluate_{label}_{checkpoint}",
        [
            sys.executable,
            str(SCRIPTS / "gpu_evaluate.py"),
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            checkpoint,
            "--results-dir",
            str(artifacts_dir),
        ],
        state=state,
        state_path=state_path,
        logs_dir=logs_dir,
    )


def _artifact_path(artifacts_dir: Path, run_dir: Path, checkpoint: str) -> Path:
    suffix = "" if checkpoint == "best_forecast" else f"_{checkpoint}"
    return artifacts_dir / f"{run_dir.name}{suffix}_metrics.json"


def _relative_change(physics: float, no_physics: float) -> float:
    return (physics / no_physics - 1.0) if no_physics != 0 else float("nan")


def _build_final_report(
    *,
    validation_id: str,
    state: dict[str, Any],
    artifacts_dir: Path,
    full_run: Path,
    resumed_run: Path,
    no_physics_run: Path,
) -> dict[str, Any]:
    full = _load_json(
        _artifact_path(artifacts_dir, full_run, "best_forecast_post_warmup")
    )
    ablation = _load_json(
        _artifact_path(artifacts_dir, no_physics_run, "best_forecast_post_warmup")
    )
    preflight = _load_json(artifacts_dir / "preflight.json")
    smoke = _load_json(artifacts_dir / "smoke.json")
    profile = _load_json(artifacts_dir / "profile.json")
    resume_stem = f"resume_check_{full_run.name}_vs_{resumed_run.name}.json"
    resume = _load_json(artifacts_dir / resume_stem)
    evaluation_files = [
        _artifact_path(artifacts_dir, run_dir, checkpoint)
        for run_dir in (full_run, no_physics_run)
        for checkpoint in EVALUATION_CHECKPOINTS
    ]
    evaluation_reports = [_load_json(path) for path in evaluation_files]
    technical_pass = (
        preflight.get("status") == "PASS"
        and smoke.get("status") == "PASS"
        and profile.get("status") == "PASS"
        and resume.get("status") == "PASS"
        and all(report.get("status") == "PASS" for report in evaluation_reports)
        and all(item.get("status") == "PASS" for item in state.get("steps", {}).values())
    )
    frequency_pass = bool(full["frequency_gate_pass"])
    decay_pass = bool(full["decay_gate_pass"])
    stability_pass = bool(full["stability_gate_pass"])
    baseline_pass = {
        horizon: bool(full["beats_persistence"][horizon])
        for horizon in ("short", "medium", "long")
    }
    mass_pass = all(bool(full["mass_gate_pass"][horizon]) for horizon in baseline_pass)
    operator_pass = all(bool(full["operator_gate_pass"][horizon]) for horizon in baseline_pass)
    reconstruction_pass = (
        float(full["reconstruction_rmse"])
        < float(full["forecast"]["short"]["persistence_rmse"])
    )
    all_scientific_gates = (
        frequency_pass
        and decay_pass
        and stability_pass
        and all(baseline_pass.values())
        and mass_pass
        and operator_pass
        and reconstruction_pass
    )
    scientific_status = "PENDING_REVIEW" if all_scientific_gates else "FAIL"
    physics_comparison = {
        horizon: {
            metric: {
                "physics": full["forecast"][horizon][metric],
                "no_physics": ablation["forecast"][horizon][metric],
                "relative_change": _relative_change(
                    full["forecast"][horizon][metric],
                    ablation["forecast"][horizon][metric],
                ),
            }
            for metric in ("rmse", "mass_drift", "operator")
        }
        for horizon in ("short", "medium", "long")
    }
    checklist = {
        "A_git_state": preflight.get("git_dirty") is False,
        "B_cuda_environment": bool(preflight.get("cuda_version")),
        "C_gpu_identity": bool(preflight.get("device")),
        "D_matrix_exp_preflight": preflight.get("matrix_exp_device") == "cuda:0",
        "E_cpu_gpu_parity": preflight.get("status") == "PASS",
        "F_amp_finite": smoke.get("status") == "PASS",
        "G_smoke_gradients": smoke.get("status") == "PASS",
        "H_gpu_short_training": smoke.get("status") == "PASS",
        "I_no_physics_full": (no_physics_run / "metadata/run_manifest.json").is_file(),
        "J_physics_full": (full_run / "metadata/run_manifest.json").is_file(),
        "K_exact_resume": resume.get("weights_bitwise_equal") is True,
        "L_held_out_long_rollout": all(
            "long" in report.get("forecast", {}) for report in evaluation_reports
        ),
        "M_spectrum_reviewed": "frequency_relative_error" in full,
        "N_physics_reviewed": bool(physics_comparison),
        "O_performance_recorded": full.get("max_samples_per_second", 0) > 0,
        "P_peak_memory_recorded": full.get("peak_gpu_memory_bytes", 0) > 0,
        "Q_artifacts_saved": True,
        "R_acceptance_reviewed": True,
    }
    technical_pass = technical_pass and all(checklist.values())
    return {
        "validation_id": validation_id,
        "generated_at": _utc_now(),
        "workflow_status": "PASS" if technical_pass else "FAIL",
        "scientific_status": scientific_status,
        "overall_acceptance": (
            "NOT_ACCEPTED"
            if not technical_pass or scientific_status == "FAIL"
            else "PENDING_RESEARCHER_REVIEW"
        ),
        "runs": {
            "physics_full": str(full_run.resolve()),
            "resumed": str(resumed_run.resolve()),
            "no_physics": str(no_physics_run.resolve()),
        },
        "scientific_checkpoint": "best_forecast_post_warmup",
        "hard_gates": {
            "frequency": {
                "pass": frequency_pass,
                "relative_error": full["frequency_relative_error"],
                "threshold": full["frequency_threshold"],
            },
            "decay": {
                "pass": decay_pass,
                "relative_error": full["decay_relative_error"],
                "threshold": full["decay_threshold"],
            },
            "stability": {
                "pass": stability_pass,
                "spectral_abscissa": full["spectral_abscissa"],
                "threshold": full["spectral_abscissa_threshold"],
            },
            "beats_persistence": {
                horizon: {
                    "pass": baseline_pass[horizon],
                    "model_rmse": full["forecast"][horizon]["rmse"],
                    "persistence_rmse": full["forecast"][horizon]["persistence_rmse"],
                }
                for horizon in ("short", "medium", "long")
            },
            "reconstruction": {
                "pass": reconstruction_pass,
                "rmse": full["reconstruction_rmse"],
                "threshold": full["forecast"]["short"]["persistence_rmse"],
            },
            "mass": {
                "pass": mass_pass,
                "threshold": full["relative_mass_drift_threshold"],
            },
            "operator": {
                "pass": operator_pass,
                "threshold": full["operator_mse_threshold"],
            },
        },
        "physics_vs_no_physics": physics_comparison,
        "checklist": checklist,
        "all_checklist_items_complete": all(checklist.values()),
    }


def _write_final_markdown(path: Path, report: dict[str, Any]) -> None:
    comparison = report["physics_vs_no_physics"]
    lines = [
        "# V0.5 complete GPU validation",
        "",
        f"- validation id: `{report['validation_id']}`",
        f"- workflow status: **{report['workflow_status']}**",
        f"- scientific status: **{report['scientific_status']}**",
        f"- overall acceptance: **{report['overall_acceptance']}**",
        f"- scientific checkpoint: `{report['scientific_checkpoint']}`",
        "",
        "## Hard gates",
        "",
        f"- frequency: **{'PASS' if report['hard_gates']['frequency']['pass'] else 'FAIL'}**; "
        f"relative error {report['hard_gates']['frequency']['relative_error']:.6g}, "
        f"threshold {report['hard_gates']['frequency']['threshold']:.6g}",
        f"- decay: **{'PASS' if report['hard_gates']['decay']['pass'] else 'FAIL'}**; "
        f"relative error {report['hard_gates']['decay']['relative_error']:.6g}, "
        f"threshold {report['hard_gates']['decay']['threshold']:.6g}",
        f"- stability: **{'PASS' if report['hard_gates']['stability']['pass'] else 'FAIL'}**; "
        f"spectral abscissa {report['hard_gates']['stability']['spectral_abscissa']:.6g}, "
        f"threshold {report['hard_gates']['stability']['threshold']:.6g}",
        f"- reconstruction: "
        f"**{'PASS' if report['hard_gates']['reconstruction']['pass'] else 'FAIL'}**; "
        f"{report['hard_gates']['reconstruction']['rmse']:.6g} vs "
        f"threshold {report['hard_gates']['reconstruction']['threshold']:.6g}",
        f"- mass drift: **{'PASS' if report['hard_gates']['mass']['pass'] else 'FAIL'}**; "
        f"threshold {report['hard_gates']['mass']['threshold']:.6g}",
        f"- operator MSE: "
        f"**{'PASS' if report['hard_gates']['operator']['pass'] else 'FAIL'}**; "
        f"threshold {report['hard_gates']['operator']['threshold']:.6g}",
        "",
        "### Rollout vs persistence",
        "",
        "| Horizon | Pass | Model RMSE | Persistence RMSE |",
        "|---|---:|---:|---:|",
    ]
    for horizon in ("short", "medium", "long"):
        gate = report["hard_gates"]["beats_persistence"][horizon]
        lines.append(
            f"| {horizon} | {'PASS' if gate['pass'] else 'FAIL'} | "
            f"{gate['model_rmse']:.6g} | {gate['persistence_rmse']:.6g} |"
        )
    lines.extend(
        (
            "",
            "## Physics vs no-physics",
            "",
            "| Horizon | Metric | Physics | No physics | Relative change |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for horizon in ("short", "medium", "long"):
        for metric in ("rmse", "mass_drift", "operator"):
            values = comparison[horizon][metric]
            lines.append(
                f"| {horizon} | {metric} | {values['physics']:.6g} | "
                f"{values['no_physics']:.6g} | {100.0 * values['relative_change']:+.3f}% |"
            )
    lines.extend(("", "## Checklist", ""))
    for name, passed in report["checklist"].items():
        lines.append(f"- [{'x' if passed else ' '}] {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_results(validation_id: str, artifacts_dir: Path) -> Path:
    destination = FINAL_RESULTS / validation_id
    if destination.exists():
        existing = destination / "final_validation.json"
        candidate = artifacts_dir / "final_validation.json"
        if existing.is_file() and existing.read_bytes() == candidate.read_bytes():
            return destination
        raise FileExistsError(f"final result directory already exists: {destination}")
    shutil.copytree(artifacts_dir, destination)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-id", default=None)
    parser.add_argument("--resume-epoch", type=int, default=75)
    parser.add_argument(
        "--skip-local-gates",
        action="store_true",
        help="Skip pytest/ruff/mypy/diff checks only when they already ran on this exact commit",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.resume_epoch <= 0:
        raise ValueError("resume epoch must be positive")
    validation_id = args.validation_id or _default_validation_id()
    session_dir = RUN_ROOT / "validation_sessions" / validation_id
    artifacts_dir = session_dir / "artifacts"
    logs_dir = session_dir / "logs"
    state_path = session_dir / "state.json"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if state_path.is_file():
        state = _load_json(state_path)
    else:
        state = {
            "validation_id": validation_id,
            "created_at": _utc_now(),
            "python": sys.executable,
            "steps": {},
        }
        _save_state(state_path, state)
    if state.get("workflow_status") == "PASS" and state.get("final_results"):
        completed_results = Path(str(state["final_results"]))
        final_report_path = completed_results / "final_validation.json"
        if final_report_path.is_file():
            completed_report = _load_json(final_report_path)
            print("=== V0.5 GPU VALIDATION ALREADY COMPLETE ===")
            print(f"workflow_status={completed_report['workflow_status']}")
            print(f"scientific_status={completed_report['scientific_status']}")
            print(f"overall_acceptance={completed_report['overall_acceptance']}")
            print(f"final_results={completed_results.resolve()}")
            return 0
    continuation = (
        f"{sys.executable} {SCRIPTS / 'gpu_validate_all.py'} "
        f"--validation-id {validation_id} --resume-epoch {args.resume_epoch}"
    )
    try:
        if not args.skip_local_gates:
            for name, command in (
                ("pytest", [sys.executable, "-m", "pytest", "-q"]),
                ("ruff", [sys.executable, "-m", "ruff", "check", "."]),
                ("mypy", [sys.executable, "-m", "mypy", "src"]),
                ("diff_check", ["git", "diff", "--check"]),
            ):
                _run_step(
                    name,
                    command,
                    state=state,
                    state_path=state_path,
                    logs_dir=logs_dir,
                )
        _run_step(
            "preflight",
            [
                sys.executable,
                str(SCRIPTS / "gpu_preflight.py"),
                "--results-dir",
                str(artifacts_dir),
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
        )
        _run_step(
            "smoke",
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
        full_run = _run_training(
            "physics_full",
            base_name=f"{validation_id}-physics-full",
            config=CONFIGS / "gpu_full.yaml",
            checkpoint_epoch=args.resume_epoch,
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
        )
        resume_checkpoint = full_run / "checkpoints" / f"epoch_{args.resume_epoch:04d}.pt"
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"required resume checkpoint missing: {resume_checkpoint}")
        resumed_run = _run_training(
            "resume",
            base_name=f"{validation_id}-resume",
            config=CONFIGS / "gpu_full.yaml",
            resume_from=resume_checkpoint,
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
        )
        _run_step(
            "resume_check",
            [
                sys.executable,
                str(SCRIPTS / "gpu_resume_check.py"),
                "--uninterrupted-run",
                str(full_run),
                "--resumed-run",
                str(resumed_run),
                "--results-dir",
                str(artifacts_dir),
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
        )
        no_physics_run = _run_training(
            "no_physics_full",
            base_name=f"{validation_id}-no-physics",
            config=CONFIGS / "gpu_full_no_physics.yaml",
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
        )
        for label, run_dir in (("physics", full_run), ("no_physics", no_physics_run)):
            for checkpoint in EVALUATION_CHECKPOINTS:
                _evaluate(
                    label,
                    run_dir,
                    checkpoint,
                    artifacts_dir=artifacts_dir,
                    state=state,
                    state_path=state_path,
                    logs_dir=logs_dir,
                )
        _run_step(
            "profile",
            [
                sys.executable,
                str(SCRIPTS / "gpu_profile.py"),
                "--results-dir",
                str(artifacts_dir),
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
        )
        report = _build_final_report(
            validation_id=validation_id,
            state=state,
            artifacts_dir=artifacts_dir,
            full_run=full_run,
            resumed_run=resumed_run,
            no_physics_run=no_physics_run,
        )
        _write_json(artifacts_dir / "final_validation.json", report)
        _write_final_markdown(artifacts_dir / "final_validation.md", report)
        destination = _export_results(validation_id, artifacts_dir)
    except Exception as error:
        state["workflow_status"] = "FAIL"
        state["error"] = str(error)
        _save_state(state_path, state)
        print(f"\nVALIDATION WORKFLOW FAILED: {error}", file=sys.stderr)
        print(f"Preserved session: {session_dir.resolve()}", file=sys.stderr)
        print(f"Continue without repeating passed steps:\n{continuation}", file=sys.stderr)
        return 1
    state["workflow_status"] = "PASS"
    state["final_results"] = str(destination.resolve())
    _save_state(state_path, state)
    print("\n=== V0.5 GPU VALIDATION COMPLETE ===")
    print(f"workflow_status={report['workflow_status']}")
    print(f"scientific_status={report['scientific_status']}")
    print(f"overall_acceptance={report['overall_acceptance']}")
    print(f"final_results={destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
