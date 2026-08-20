#!/usr/bin/env python3
"""One-command, non-silent, route-dependent V0.8 validation on one CUDA GPU."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, TextIO

import torch

ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from eval.evaluate_v0_7 import evaluate_v0_7  # noqa: E402
from eval.evaluate_v0_8 import evaluate_v0_8  # noqa: E402
from jka_model.config import ProjectConfig, load_config, save_config  # noqa: E402
from jka_model.context import aggregate_v0_8_results  # noqa: E402
from jka_model.data import (  # noqa: E402
    generate_cylinder_wake_2d_trajectories,
    save_cylinder_wake_dataset,
    validate_cylinder_wake_dataset,
)
from jka_model.residual import compare_residual_memory_v0_7  # noqa: E402
from jka_model.utils import create_versioned_session, get_git_commit  # noqa: E402
from train.prepare_v0_7 import prepare_v0_7_cache  # noqa: E402
from train.train_v0_6 import train_v0_6  # noqa: E402
from train.train_v0_7 import train_v0_7  # noqa: E402
from train.train_v0_8 import train_v0_8  # noqa: E402


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


def _stage(label: str, action):
    print(f"[V0.8][validation] {label}: START", flush=True)
    result = action()
    print(f"[V0.8][validation] {label}: PASS", flush=True)
    return result


def _run_tests(python: str, log_path: Path) -> None:
    command = [
        python,
        "-m",
        "pytest",
        "-q",
        "tests/test_v0_8_physical_problem.py",
        "tests/test_v0_8_context.py",
        "tests/test_v0_8_workflow.py",
    ]
    with log_path.open("w", encoding="utf-8") as stream:
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
        raise RuntimeError(f"targeted V0.8 tests failed with exit code {code}")


def _resolved(
    template: ProjectConfig,
    *,
    seed: int,
    dataset_path: Path,
    stage: str,
    run_root: Path,
    context_seed: int | None = None,
    family: str | None = None,
) -> ProjectConfig:
    payload = template.to_dict()
    payload["training"].update({"seed": seed, "stage": stage, "run_root": str(run_root.resolve())})
    payload["data"]["split"]["seed"] = seed
    payload["cylinder_wake_2d"]["dataset_path"] = str(dataset_path.resolve())
    if context_seed is not None:
        payload["v0_8_training"]["context_initialization_seed"] = context_seed
    if family is not None:
        payload["v0_8_context"]["family"] = family
    payload["tags"] = [
        "v0.8",
        "gpu-formal",
        "cylinder-wake",
        f"flow-backbone-seed-{seed}",
        f"stage-{stage}",
        *([] if context_seed is None else [f"context-seed-{context_seed}"]),
        *([] if family is None else [f"family-{family}"]),
    ]
    return ProjectConfig.from_dict(payload)


def _with_residual_sweep(
    config: ProjectConfig,
    *,
    history: int,
    initialization_seed: int,
) -> ProjectConfig:
    payload = config.to_dict()
    payload["training"]["stage"] = "residual"
    payload["residual_closure"]["history"] = history
    payload["residual_training"]["initialization_seed"] = initialization_seed
    return ProjectConfig.from_dict(payload)


def _backbone_accepted(result: Any) -> dict[str, Any]:
    evaluation = result.evaluation
    long = evaluation["rollout"]["long"]
    checks = {
        "finite": bool(evaluation["finite"]),
        "latent_not_collapsed": bool(evaluation["collapse_gate"]),
        "beats_persistence_long": float(long["rmse"]) < float(long["persistence_rmse"]),
        "physical_metric_finite": all(
            torch.isfinite(torch.tensor(float(long[name]))) for name in ("mass_drift", "operator")
        ),
        "divergence_within_limit": float(long["mass_drift"])
        <= float(evaluation["relative_mass_drift_threshold"]),
        "boundary_within_limit": float(long["operator"])
        <= float(evaluation["operator_mse_threshold"]),
        "frequency_retained": float(evaluation["frequency_relative_error"])
        <= float(evaluation["frequency_threshold"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {
            "reconstruction_rmse": evaluation["reconstruction_rmse"],
            "long_rollout": long,
            "frequency_relative_error": evaluation["frequency_relative_error"],
            "online_min_std": evaluation["tracking"]["online"]["min_dimension_std"],
        },
        "checkpoint": str(result.best_checkpoint),
    }


def _grid_adequacy(template: ProjectConfig, nominal: Any, seed: int) -> dict[str, Any]:
    assert template.cylinder_wake_2d is not None
    nominal_config = template.cylinder_wake_2d
    coarse = replace(
        nominal_config,
        num_trajectories=3,
        nx=nominal_config.nx // 2,
        ny=nominal_config.ny // 2,
        solver_steps_per_snapshot=max(1, nominal_config.solver_steps_per_snapshot // 2),
        dataset_path="",
    )
    coarse_data = generate_cylinder_wake_2d_trajectories(coarse, seed=seed, device="cuda")
    coarse_report = validate_cylinder_wake_dataset(coarse_data, coarse, require_shedding=True)
    nominal_report = validate_cylinder_wake_dataset(nominal, nominal_config, require_shedding=True)
    coarse_frequency = statistics.median(coarse_report["metrics"]["dominant_lift_frequencies"])
    nominal_frequency = statistics.median(nominal_report["metrics"]["dominant_lift_frequencies"])
    frequency_relative_change = abs(nominal_frequency - coarse_frequency) / max(
        abs(nominal_frequency), 1e-12
    )
    result = {
        "status": "PASS"
        if coarse_report["status"] == nominal_report["status"] == "PASS"
        and frequency_relative_change <= 0.35
        else "FAIL",
        "coarse": coarse_report,
        "nominal": nominal_report,
        "frequency_relative_change": frequency_relative_change,
        "acceptance_limit": 0.35,
    }
    return result


def _write_stop_report(compact: Path, classification: dict[str, Any]) -> dict[str, Any]:
    route = classification["residual_route"]
    status = "DIAGNOSTIC_ONLY" if route == "R1" else "INCONCLUSIVE"
    decision = {
        "schema_version": 1,
        "physical_problem": "cylinder_wake_2d",
        "backbone_status": "PASS",
        "v0_7_route_on_new_problem": route,
        "context_family": "NONE",
        "dynamic_context": "NOT_ASSESSED",
        "closed_loop_utility": "NOT_ASSESSED",
        "physics_status": "PASS",
        "v0_9_operator_adaptation_readiness": "NOT_READY",
        "v0_9_ready": False,
        "scientific_status": status,
    }
    (compact / "evaluation").mkdir(parents=True, exist_ok=True)
    (compact / "reports").mkdir(parents=True, exist_ok=True)
    (compact / "evaluation" / "v0_8_scientific_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (compact / "evaluation" / "residual_structure_assessment.json").write_text(
        json.dumps(classification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (compact / "summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = (
        "# V0.8 scientific report\n\n"
        f"V0.7 classified the new cylinder problem as `{route}`. V0.8 therefore stopped "
        "before context training, as required by the evidence-owned route.\n"
    )
    (compact / "reports" / "v0_8_scientific_report.md").write_text(report, encoding="utf-8")
    (compact / "report.md").write_text(report, encoding="utf-8")
    return decision


def validate_completion_payload(payload: dict[str, Any]) -> None:
    required = {
        "requested_validation_id",
        "resolved_validation_id",
        "status",
        "git_commit",
        "all_required_stages_completed",
        "backbone_run_count",
        "v0_7_evaluation_run_count",
        "v0_8_candidate_run_count",
        "v0_8_test_run_count",
        "final_route",
        "final_context_family",
        "scientific_status",
        "output_paths",
    }
    if required - set(payload) or payload["status"] != "PASS":
        raise ValueError("V0.8 completion payload is incomplete")
    if not payload["all_required_stages_completed"]:
        raise ValueError("V0.8 completion cannot claim PASS with incomplete stages")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-id",
        default=datetime.now(timezone.utc).strftime("v08-%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[47, 53, 59])
    parser.add_argument("--skip-software-tests", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise SystemExit("V0.8 formal validation requires exactly three unique seeds")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty and not args.allow_dirty:
        raise SystemExit("formal V0.8 validation requires a clean git working tree")

    results_root = ROOT / "gpu_validation" / "v0_8" / "results"
    session = create_versioned_session(
        ROOT / "runs" / "v0_8",
        args.validation_id,
        reserved_roots=(results_root,),
    )
    root = session.path
    compact = results_root / session.resolved_id
    compact.mkdir(parents=True, exist_ok=False)
    for name in ("software", "configs", "data", "seeds", "v0_7_assessment", "logs"):
        (root / name).mkdir()
    workflow_stream = (root / "logs" / "workflow.log").open("w", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, workflow_stream)  # type: ignore[assignment]
    sys.stderr = _Tee(original_stderr, workflow_stream)  # type: ignore[assignment]
    current_stage = "session_initialization"
    current_run: dict[str, Any] | None = None
    completed = {"backbones": 0, "v0_7": 0, "v0_8_candidates": 0, "v0_8_tests": 0}
    last_checkpoint: str | None = None
    try:
        print(
            f"[V0.8][validation] SESSION requested={session.requested_id} "
            f"resolved={session.resolved_id} gpu={torch.cuda.get_device_name(0)}",
            flush=True,
        )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("formal RTX-5080 workflow requires CUDA BF16 support")
        python = str(ROOT / ".venv" / "bin" / "python")
        if not Path(python).is_file():
            python = sys.executable
        if not args.skip_software_tests:
            current_stage = "targeted_software_tests"
            _stage(
                "G0 targeted software tests",
                lambda: _run_tests(python, root / "software" / "pytest.log"),
            )

        template = load_config(
            ROOT / "gpu_validation" / "v0_8" / "configs" / "gpu_cylinder_context.yaml"
        )
        assert template.cylinder_wake_2d and template.memory_sweep and template.v0_8_evaluation
        datasets: dict[int, Path] = {}
        configs: dict[int, ProjectConfig] = {}
        first_nominal = None
        current_stage = "physical_problem_validation"
        for seed in args.seeds:
            current_run = {"seed": seed, "kind": "physical_data"}
            dataset_path = root / "data" / f"cylinder_wake_seed_{seed}.pt"
            data = _stage(
                f"G1 physical dataset seed={seed}",
                lambda seed=seed: generate_cylinder_wake_2d_trajectories(
                    template.cylinder_wake_2d, seed=seed, device="cuda"
                ),
            )
            report = validate_cylinder_wake_dataset(data, template.cylinder_wake_2d)
            if report["status"] != "PASS":
                raise RuntimeError(f"physical problem acceptance failed for seed {seed}")
            save_cylinder_wake_dataset(data, template.cylinder_wake_2d, dataset_path)
            if seed == args.seeds[0]:
                first_nominal = data
            (root / "data" / f"physical_acceptance_seed_{seed}.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            datasets[seed] = dataset_path
            configs[seed] = _resolved(
                template,
                seed=seed,
                dataset_path=dataset_path,
                stage="jepa",
                run_root=root / "seeds" / f"seed_{seed}" / "backbone_runs",
            )
        assert first_nominal is not None
        grid = _stage(
            "G1 coarse-to-nominal grid adequacy",
            partial(_grid_adequacy, template, first_nominal, args.seeds[0]),
        )
        del first_nominal
        del data
        (root / "data" / "grid_adequacy.json").write_text(
            json.dumps(grid, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if grid["status"] != "PASS":
            raise RuntimeError("coarse-to-nominal grid adequacy failed")

        backbones: dict[int, Path] = {}
        caches: dict[int, Path] = {}
        current_stage = "cylinder_backbone_training"
        for seed in args.seeds:
            current_run = {"seed": seed, "kind": "v0_6_backbone"}
            save_config(configs[seed], root / "configs" / f"seed_{seed}_backbone.yaml")
            result = _stage(
                f"G2 cylinder JEPA-Koopman backbone seed={seed}",
                lambda seed=seed: train_v0_6(configs[seed], device="cuda", run_name=f"seed-{seed}"),
            )
            acceptance = _backbone_accepted(result)
            (root / "seeds" / f"seed_{seed}" / "backbone_acceptance.json").write_text(
                json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if acceptance["status"] != "PASS":
                raise RuntimeError(f"cylinder backbone acceptance failed for seed {seed}")
            backbones[seed] = result.best_checkpoint
            last_checkpoint = str(result.best_checkpoint)
            completed["backbones"] += 1

            residual_config = _resolved(
                template,
                seed=seed,
                dataset_path=datasets[seed],
                stage="residual",
                run_root=root,
            )
            configs[seed] = residual_config
            cache_path = root / "seeds" / f"seed_{seed}" / "cache" / "residual_cache.pt"
            _stage(
                f"G3 frozen residual cache seed={seed}",
                lambda seed=seed, cache_path=cache_path: prepare_v0_7_cache(
                    configs[seed],
                    backbone_checkpoint=backbones[seed],
                    destination=cache_path,
                    diagnostics_path=cache_path.with_name("residual_diagnostics.json"),
                    device="cuda",
                ),
            )
            caches[seed] = cache_path

        current_stage = "v0_7_route_assessment"
        expected_v07 = template.v0_7_evaluation.formal_record_count
        for seed in args.seeds:
            seed_root = root / "seeds" / f"seed_{seed}"
            for initialization_seed in template.memory_sweep.initialization_seeds:
                for history in template.memory_sweep.history_lengths:
                    sweep_config = _with_residual_sweep(
                        configs[seed],
                        history=history,
                        initialization_seed=initialization_seed,
                    )
                    variants = ["history", "instantaneous"]
                    if history == 1:
                        variants = ["zero", "linear", *variants]
                    else:
                        variants.append("shuffled_history")
                    for variant in variants:
                        current_run = {
                            "seed": seed,
                            "closure_seed": initialization_seed,
                            "history": history,
                            "variant": variant,
                        }
                        label = (
                            f"G4 V0.7 seed={seed} init={initialization_seed} H={history} {variant}"
                        )
                        print(f"[V0.8][validation] {label}: START", flush=True)
                        run_dir = (
                            seed_root
                            / "v0_7_runs"
                            / f"init_{initialization_seed}"
                            / f"h{history}"
                            / variant
                        )
                        trained = train_v0_7(
                            sweep_config,
                            backbone_checkpoint=backbones[seed],
                            cache_path=caches[seed],
                            variant=variant,
                            run_dir=run_dir,
                            device="cuda",
                        )
                        evaluation_path = (
                            seed_root
                            / "evaluation"
                            / f"init_{initialization_seed}"
                            / f"h{history}"
                            / f"{variant}.json"
                        )
                        evaluate_v0_7(
                            sweep_config,
                            checkpoint=trained.best_checkpoint,
                            cache_path=caches[seed],
                            device="cuda",
                            output_path=evaluation_path,
                        )
                        completed["v0_7"] += 1
                        last_checkpoint = str(trained.best_checkpoint)
                        print(f"[V0.8][validation] {label}: PASS", flush=True)
        if completed["v0_7"] != expected_v07:
            raise RuntimeError(f"V0.7 matrix incomplete: {completed['v0_7']} != {expected_v07}")
        classification = _stage(
            "G4 V0.7 route decision",
            lambda: compare_residual_memory_v0_7(root, root / "v0_7_assessment"),
        )
        route_path = root / "v0_7_assessment" / "evaluation" / "memory_classification.json"
        route = str(classification["residual_route"])

        current_stage = "v0_8_context_candidates"
        selected_family: str | None = None
        decision: dict[str, Any]
        if route in {"R1", "INCONCLUSIVE"}:
            decision = _write_stop_report(compact, classification)
        else:
            families = (
                ["instantaneous"]
                if route == "R2"
                else [
                    "instantaneous",
                    "instantaneous_matched",
                    "history_mlp",
                    "attention",
                ]
            )
            candidates: dict[str, list[tuple[int, int, Any, ProjectConfig]]] = {
                family: [] for family in families
            }
            for family in families:
                for seed in args.seeds:
                    for context_seed in template.v0_8_evaluation.context_initialization_seeds:
                        context_config = _resolved(
                            template,
                            seed=seed,
                            dataset_path=datasets[seed],
                            stage="context",
                            run_root=root,
                            context_seed=context_seed,
                            family=family,
                        )
                        run_dir = (
                            root
                            / "seeds"
                            / f"seed_{seed}"
                            / "candidates"
                            / family
                            / f"init_{context_seed}"
                        )
                        current_run = {
                            "seed": seed,
                            "context_seed": context_seed,
                            "family": family,
                        }
                        trained = _stage(
                            f"G5 context candidate seed={seed} init={context_seed} family={family}",
                            lambda context_config=context_config, run_dir=run_dir, seed=seed: (
                                train_v0_8(
                                    context_config,
                                    backbone_checkpoint=backbones[seed],
                                    residual_cache=caches[seed],
                                    v0_7_route_result=route_path,
                                    run_dir=run_dir,
                                    device="cuda",
                                )
                            ),
                        )
                        candidates[family].append((seed, context_seed, trained, context_config))
                        completed["v0_8_candidates"] += 1
                        if trained.best_checkpoint is not None:
                            last_checkpoint = str(trained.best_checkpoint)
            if route == "R2":
                selected_family = "instantaneous"
            else:
                temporal_scores = {
                    family: statistics.mean(
                        run.validation_metrics["residual_standardized_mse"]
                        for _, _, run, _ in candidates[family]
                    )
                    for family in ("history_mlp", "attention")
                }
                selected_family = min(temporal_scores, key=temporal_scores.get)
                selection = {
                    "selection_split": "validation",
                    "test_opened": False,
                    "route": route,
                    "candidate_mean_validation_standardized_mse": {
                        family: statistics.mean(
                            run.validation_metrics["residual_standardized_mse"]
                            for _, _, run, _ in family_runs
                        )
                        for family, family_runs in candidates.items()
                    },
                    "selected_family": selected_family,
                }
                (root / "v0_8_family_selection.json").write_text(
                    json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            current_stage = "locked_test_and_physics_evaluation"
            for seed, context_seed, trained, context_config in candidates[selected_family]:
                if trained.best_checkpoint is None:
                    raise RuntimeError("selected V0.8 candidate lacks a checkpoint")
                output = root / "seeds" / f"seed_{seed}" / "contexts" / f"init_{context_seed}"
                _stage(
                    f"G6 locked test seed={seed} init={context_seed} family={selected_family}",
                    partial(
                        evaluate_v0_8,
                        context_config,
                        checkpoint=trained.best_checkpoint,
                        backbone_checkpoint=backbones[seed],
                        residual_cache=caches[seed],
                        output_dir=output,
                        device="cuda",
                    ),
                )
                completed["v0_8_tests"] += 1
            decision = _stage(
                "G7 nested-seed aggregation and report",
                lambda: aggregate_v0_8_results(root, compact),
            )

        current_stage = "completion_record"
        completion = {
            "requested_validation_id": session.requested_id,
            "resolved_validation_id": session.resolved_id,
            "status": "PASS",
            "git_commit": get_git_commit(ROOT),
            "all_required_stages_completed": True,
            "backbone_run_count": completed["backbones"],
            "v0_7_evaluation_run_count": completed["v0_7"],
            "v0_8_candidate_run_count": completed["v0_8_candidates"],
            "v0_8_test_run_count": completed["v0_8_tests"],
            "final_route": route,
            "final_context_family": selected_family,
            "scientific_status": decision.get("dynamic_context", decision.get("scientific_status")),
            "output_paths": {
                "raw_run": str(root),
                "compact_result": str(compact),
                "report": str(compact / "report.md"),
            },
        }
        validate_completion_payload(completion)
        (compact / "completion.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (root / "completion.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"V0.8 VALIDATION COMPLETE status=PASS id={session.resolved_id} "
            f"route={route} family={selected_family} report={compact / 'report.md'}",
            flush=True,
        )
    except BaseException as error:
        failure = {
            "validation_id": session.resolved_id,
            "status": "FAILED_INCOMPLETE",
            "failed_stage": current_stage,
            "failed_run": current_run,
            "exception_summary": f"{type(error).__name__}: {error}",
            "completed_work": completed,
            "expected_work": {
                "backbones": 3,
                "v0_7_evaluations": 144,
                "v0_8_candidates": "route-dependent",
                "v0_8_locked_tests": "0 or 9",
            },
            "last_valid_checkpoint": last_checkpoint,
            "git_commit": get_git_commit(ROOT),
            "traceback": traceback.format_exc(),
        }
        (compact / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (root / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"V0.8 VALIDATION FAILED id={session.resolved_id} report={compact / 'failure.json'}",
            flush=True,
        )
        raise
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        workflow_stream.close()


if __name__ == "__main__":
    main()
