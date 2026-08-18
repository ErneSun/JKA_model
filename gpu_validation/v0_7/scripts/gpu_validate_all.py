#!/usr/bin/env python3
"""One-command non-silent V0.7 residual and memory characterization workflow."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import torch

ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from eval.evaluate_v0_7 import evaluate_v0_7  # noqa: E402
from jka_model.config import ProjectConfig, load_config, save_config  # noqa: E402
from jka_model.residual import compare_residual_memory_v0_7  # noqa: E402
from jka_model.utils import create_versioned_session, get_git_commit, load_checkpoint  # noqa: E402
from train.prepare_v0_7 import prepare_v0_7_cache  # noqa: E402
from train.train_v0_7 import _backbone_contract, train_v0_7  # noqa: E402


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def validate_completion_payload(payload: dict[str, Any]) -> None:
    required = {
        "requested_validation_id",
        "resolved_validation_id",
        "git_commit",
        "backbone_seeds",
        "closure_initialization_seeds",
        "expected_evaluation_records",
        "actual_evaluation_records",
        "all_expected_runs_completed",
        "provenance_checks_passed",
        "required_reports_produced",
        "status",
    }
    if required - set(payload):
        raise ValueError("V0.7 completion payload is incomplete")
    if payload["status"] != "PASS" or not all(
        bool(payload[key])
        for key in (
            "all_expected_runs_completed",
            "provenance_checks_passed",
            "required_reports_produced",
        )
    ):
        raise ValueError("V0.7 completion payload cannot claim PASS")
    if int(payload["actual_evaluation_records"]) != int(payload["expected_evaluation_records"]):
        raise ValueError("V0.7 completion record count mismatch")


def validate_failure_payload(payload: dict[str, Any]) -> None:
    required = {
        "validation_id",
        "status",
        "failed_stage",
        "failed_run",
        "error",
        "completed_record_count",
        "expected_record_count",
        "last_valid_checkpoint",
        "git_commit",
    }
    if required - set(payload) or payload.get("status") != "FAILED_INCOMPLETE":
        raise ValueError("V0.7 failure payload is incomplete")
    if int(payload["completed_record_count"]) > int(payload["expected_record_count"]):
        raise ValueError("V0.7 failure record count is invalid")


def _run_checked(command: list[str], log: Path, label: str) -> None:
    print(f"[V0.7][validation] {label}: START", flush=True)
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
        code = process.wait()
    if code:
        print(f"[V0.7][validation] {label}: FAIL log={log}", flush=True)
        raise RuntimeError(f"{label} failed with exit code {code}")
    print(f"[V0.7][validation] {label}: PASS log={log}", flush=True)


def _resolved(template: ProjectConfig, seed: int) -> ProjectConfig:
    payload = template.to_dict()
    payload["training"]["seed"] = seed
    payload["data"]["split"]["seed"] = seed
    payload["tags"] = [
        "v0.7",
        "gpu-full",
        "residual-learnability",
        "memory-characterization",
        "inherited-v0.6-problem",
        f"seed-{seed}",
    ]
    return ProjectConfig.from_dict(payload)


def _with_history(config: ProjectConfig, history: int, closure_seed: int) -> ProjectConfig:
    payload = config.to_dict()
    payload["residual_closure"]["history"] = history
    payload["residual_training"]["initialization_seed"] = closure_seed
    payload["tags"] = [
        *payload["tags"],
        f"history-{history}",
        f"closure-seed-{closure_seed}",
    ]
    return ProjectConfig.from_dict(payload)


def _discover_backbones(root: Path, configs: dict[int, ProjectConfig]) -> dict[int, Path]:
    candidates: dict[int, list[Path]] = {seed: [] for seed in configs}
    for path in root.glob("*/checkpoints/best_forecast_post_warmup.pt"):
        try:
            saved = load_checkpoint(path, map_location="cpu")
        except (OSError, ValueError):
            continue
        if saved.config is None or saved.train_stage.value != "jepa":
            continue
        seed = saved.config.training.seed
        if seed in configs and _backbone_contract(saved.config) == _backbone_contract(
            configs[seed]
        ):
            candidates[seed].append(path)
    missing = [seed for seed, paths in candidates.items() if not paths]
    if missing:
        raise ValueError(f"no compatible V0.6 JEPA checkpoint for seeds {missing} below {root}")
    return {
        seed: max(paths, key=lambda item: item.stat().st_mtime)
        for seed, paths in candidates.items()
    }


def _train_and_evaluate(
    *,
    config: ProjectConfig,
    backbone: Path,
    cache: Path,
    variant: str,
    run_dir: Path,
    evaluation_path: Path,
) -> dict[str, Any]:
    result = train_v0_7(
        config,
        backbone_checkpoint=backbone,
        cache_path=cache,
        variant=variant,
        run_dir=run_dir,
        device="cuda",
    )
    return evaluate_v0_7(
        config,
        checkpoint=result.best_checkpoint,
        cache_path=cache,
        device="cuda",
        output_path=evaluation_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-id", default=datetime.now(timezone.utc).strftime("v07-%Y%m%dT%H%M%SZ")
    )
    parser.add_argument("--v0-6-root", type=Path, default=Path("runs/v0_6/gpu"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[47, 53, 59])
    parser.add_argument("--skip-software-tests", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise SystemExit("V0.7 scientific validation requires exactly three unique seeds")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    results_root = ROOT / "gpu_validation" / "v0_7" / "results"
    session = create_versioned_session(
        ROOT / "runs" / "v0_7",
        args.validation_id,
        reserved_roots=(results_root,),
    )
    root = session.path
    for name in ("software", "configs", "seeds", "artifacts", "logs"):
        (root / name).mkdir()
    compact = results_root / session.resolved_id
    compact.mkdir(parents=True, exist_ok=False)
    workflow_log = (root / "logs" / "workflow.log").open("w", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, workflow_log)  # type: ignore[assignment]
    sys.stderr = _Tee(original_stderr, workflow_log)  # type: ignore[assignment]
    current_stage = "session_initialization"
    current_run: dict[str, Any] | None = None
    completed_record_count = 0
    expected_record_count = 144
    last_valid_checkpoint: str | None = None
    try:
        print(
            f"[V0.7][validation] SESSION requested={args.validation_id} "
            f"resolved={session.resolved_id} path={root}",
            flush=True,
        )
        python = str(ROOT / ".venv" / "bin" / "python")
        if not Path(python).is_file():
            python = sys.executable
        if not args.skip_software_tests:
            current_stage = "software_tests"
            _run_checked(
                [
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_v0_7_residual.py",
                    "tests/test_v0_7_gpu_workflow.py",
                    "tests/test_v0_7_integration.py",
                ],
                root / "software" / "pytest.log",
                "converged V0.7 tests",
            )
        print(
            f"[V0.7][validation] CUDA preflight: PASS gpu={torch.cuda.get_device_name(0)} "
            f"torch={torch.__version__}",
            flush=True,
        )
        template = load_config(
            ROOT / "gpu_validation" / "v0_7" / "configs" / "gpu_residual_multiseed.yaml"
        )
        assert template.memory_sweep and template.v0_7_evaluation
        expected_record_count = template.v0_7_evaluation.formal_record_count
        configs = {seed: _resolved(template, seed) for seed in args.seeds}
        backbones = _discover_backbones((ROOT / args.v0_6_root).resolve(), configs)
        for seed in args.seeds:
            print(f"[V0.7][validation] seed={seed}: START", flush=True)
            seed_root = root / "seeds" / f"seed_{seed}"
            for name in ("cache", "runs", "evaluation"):
                (seed_root / name).mkdir(parents=True)
            cache_path = seed_root / "cache" / "residual_cache.pt"
            prepare_v0_7_cache(
                configs[seed],
                backbone_checkpoint=backbones[seed],
                destination=cache_path,
                diagnostics_path=seed_root / "cache" / "residual_diagnostics.json",
                device="cuda",
            )
            for closure_seed in template.memory_sweep.initialization_seeds:
                for history in template.memory_sweep.history_lengths:
                    history_config = _with_history(configs[seed], history, closure_seed)
                    save_config(
                        history_config,
                        root / "configs" / f"seed_{seed}_closure_{closure_seed}_h{history}.yaml",
                    )
                    variants = ["history", "instantaneous"]
                    if history == 1:
                        variants = ["zero", "linear", *variants]
                    else:
                        variants.append("shuffled_history")
                    for variant in variants:
                        label = f"seed={seed} closure_seed={closure_seed} H={history} {variant}"
                        current_stage = "train_and_evaluate"
                        current_run = {
                            "backbone_data_seed": seed,
                            "closure_init_seed": closure_seed,
                            "history_length_steps": history,
                            "variant": variant,
                        }
                        print(f"[V0.7][validation] {label}: START", flush=True)
                        run_dir = (
                            seed_root / "runs" / f"closure_{closure_seed}" / f"h{history}" / variant
                        )
                        _train_and_evaluate(
                            config=history_config,
                            backbone=backbones[seed],
                            cache=cache_path,
                            variant=variant,
                            run_dir=run_dir,
                            evaluation_path=(
                                seed_root
                                / "evaluation"
                                / f"closure_{closure_seed}"
                                / f"h{history}"
                                / f"{variant}.json"
                            ),
                        )
                        completed_record_count += 1
                        last_valid_checkpoint = str(run_dir / "checkpoints" / "best.pt")
                        print(f"[V0.7][validation] {label}: PASS", flush=True)
            print(f"[V0.7][validation] seed={seed}: PASS", flush=True)
        current_stage = "residual_structure_comparison"
        current_run = None
        print("[V0.7][validation] trained-result comparison: START", flush=True)
        classification = compare_residual_memory_v0_7(root, compact)
        classification["workflow"] = {
            "requested_validation_id": session.requested_id,
            "resolved_validation_id": session.resolved_id,
            "git_commit": get_git_commit(ROOT),
            "software_tests_skipped": bool(args.skip_software_tests),
            "software_tests_passed": not args.skip_software_tests,
            "cuda_device": torch.cuda.get_device_name(0),
            "backbone_seeds": args.seeds,
            "closure_initialization_seeds": list(template.memory_sweep.initialization_seeds),
            "expected_evaluation_records": expected_record_count,
            "actual_evaluation_records": completed_record_count,
            "all_expected_runs_completed": completed_record_count == expected_record_count,
            "provenance_checks_passed": True,
            "required_reports_produced": True,
            "status": "PASS",
        }
        (compact / "evaluation" / "memory_classification.json").write_text(
            json.dumps(classification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (compact / "evaluation" / "residual_structure_assessment.json").write_text(
            json.dumps(classification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name in ("evaluation", "plots", "reports"):
            shutil.copytree(compact / name, root / "artifacts" / name)
        (compact / "summary.json").write_text(
            json.dumps(classification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copy2(compact / "reports" / "residual_decision_report.md", compact / "report.md")
        completion = classification["workflow"]
        validate_completion_payload(completion)
        (compact / "completion.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(compact / "completion.json", root / "artifacts" / "completion.json")
        print("[V0.7][validation] trained-result comparison: PASS", flush=True)
        print(
            "V0.7 VALIDATION COMPLETE status=PASS "
            f"id={session.resolved_id} "
            f"learnability={classification['residual_learnability']} "
            f"route={classification['residual_route']} "
            f"utility={classification['closed_loop_utility']} "
            f"memory={classification['memory_class']} report={compact / 'report.md'}",
            flush=True,
        )
    except BaseException as error:
        failure = {
            "validation_id": session.resolved_id,
            "status": "FAILED_INCOMPLETE",
            "failed_stage": current_stage,
            "failed_run": current_run,
            "error": f"{type(error).__name__}: {error}",
            "completed_record_count": completed_record_count,
            "expected_record_count": expected_record_count,
            "last_valid_checkpoint": last_valid_checkpoint,
            "git_commit": get_git_commit(ROOT),
            "traceback": traceback.format_exc(),
        }
        validate_failure_payload(failure)
        (compact / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"V0.7 VALIDATION FAILED id={session.resolved_id} report={compact / 'failure.json'}",
            flush=True,
        )
        raise
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        workflow_log.close()


if __name__ == "__main__":
    main()
