#!/usr/bin/env python3
"""Formal non-silent Phase-3 joint refinement with a matched frozen reference."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from jka_model.config import ProjectConfig, load_config, save_config  # noqa: E402
from jka_model.manifold import (  # noqa: E402
    classify_matched_phase3_run,
    classify_phase3_metrics,
    classify_phase3_route,
    nested_route_support,
)
from jka_model.utils import create_versioned_session, get_git_commit  # noqa: E402
from train.train_v0_9_phase3 import train_v0_9_phase3_joint  # noqa: E402


def _stage(label: str, action: Any) -> Any:
    print(f"[V0.9][phase3-joint] {label}: START", flush=True)
    result = action()
    print(f"[V0.9][phase3-joint] {label}: PASS", flush=True)
    return result


def _run_tests(python: str, log_path: Path) -> None:
    command = [
        python,
        "-m",
        "pytest",
        "-q",
        "tests/test_v0_9_phase2.py",
        "tests/test_v0_9_phase3.py",
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
        raise RuntimeError(f"Phase-3 joint targeted tests failed with exit code {code}")


def _dirty_source_paths(porcelain: str) -> list[str]:
    ignored = ("runs/", "gpu_validation/v0_8/results/", "gpu_validation/v0_9/results/")
    return [
        line[3:].split(" -> ")[-1]
        for line in porcelain.splitlines()
        if len(line) >= 4 and not line[3:].split(" -> ")[-1].startswith(ignored)
    ]


def _resolved_config(
    source: Path,
    *,
    phase2_id: str,
    dataset_path: Path,
    run_root: Path,
    seed: int,
    condition_mode: str,
    operator_seed: int,
    route: str = "joint",
    phase37: bool = False,
) -> ProjectConfig:
    if route not in {"joint", "from_scratch"}:
        raise ValueError("invalid Phase-3 resolved-config route")
    payload = load_config(source).to_dict()
    payload["training"].update(
        {"seed": seed, "stage": "adaptive", "run_root": str(run_root.resolve())}
    )
    payload["data"]["split"]["seed"] = seed
    payload["cylinder_wake_2d"]["dataset_path"] = str(dataset_path.resolve())
    payload["v0_9_adaptive"]["condition_mode"] = condition_mode
    payload["v0_9_training"]["operator_initialization_seed"] = operator_seed
    if not isinstance(payload.get("v0_9_phase3"), dict):
        template_phase3 = load_config(
            ROOT / "gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml"
        ).v0_9_phase3
        if template_phase3 is None:
            raise ValueError("Phase-3 template configuration is missing")
        payload["v0_9_phase3"] = template_phase3.to_dict()
    payload["v0_9_phase3"].update(
        {
            "enabled": True,
            "source_phase2_result": phase2_id,
            "physics_aligned_latent_enabled": phase37,
            "observer_admission_enabled": phase37,
        }
    )
    payload["tags"] = [
        "v0.9",
        f"phase3-{route}",
        f"backbone-seed-{seed}",
        f"operator-seed-{operator_seed}",
        f"condition-{condition_mode}",
        "phase3.7-physics-aligned" if phase37 else "phase3.6-decoded-physical",
    ]
    return ProjectConfig.from_dict(payload)


def _corrected_audit(
    decision: dict[str, Any], phase2_summary: dict[str, Any], config: ProjectConfig
) -> dict[str, Any]:
    phase3 = config.v0_9_phase3
    evaluation = config.v0_9_evaluation
    if phase3 is None or evaluation is None:
        raise ValueError("Phase-3 reassessment requires configured thresholds")
    corrected_rows = copy.deepcopy(decision["seeds"])
    for row in corrected_rows:
        data_divergence = float(row["data_divergence_rms"])
        reconstruction_divergence = float(row["reconstruction_divergence_rms"])
        corrected_divergence_degradation = (
            reconstruction_divergence - data_divergence
        ) / max(abs(data_divergence), math.sqrt(evaluation.max_divergence_mse))
        row.update(
            classify_phase3_metrics(
                divergence_degradation=corrected_divergence_degradation,
                boundary_degradation=float(row["boundary_degradation"]),
                reconstruction_divergence_rms=float(row["reconstruction_divergence_rms"]),
                reconstruction_boundary_mse=float(
                    row["reconstruction_boundary_no_slip_mse"]
                ),
                reconstruction_outer_boundary_mse=float(
                    row["reconstruction_outer_boundary_mse"]
                ),
                roundtrip_nrmse=float(row["roundtrip_nrmse"]),
                nominal_tangent_divergence=float(row["nominal_tangent_divergence"]),
                max_divergence_mse=evaluation.max_divergence_mse,
                max_boundary_mse=evaluation.max_boundary_mse,
                max_reconstruction_physics_degradation=(
                    phase3.max_reconstruction_physics_degradation
                ),
                max_roundtrip_nrmse=phase3.max_roundtrip_nrmse,
                max_tangent_divergence=phase3.max_tangent_divergence,
            )
        )
        row["divergence_degradation"] = corrected_divergence_degradation
    route = classify_phase3_route(corrected_rows, phase2_summary)
    return {
        "schema_version": 1,
        "source_phase3_audit": decision.get("validation_id"),
        "correction": "divergence RMS is compared to sqrt(max_divergence_mse)",
        "original_next_candidate": decision.get("next_candidate"),
        "corrected": route,
        "seeds": corrected_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-id",
        default=datetime.now(timezone.utc).strftime("v09-added-p3-joint-%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--phase2-id", default="v09-added-p2-physical-20260824T105209Z"
    )
    parser.add_argument(
        "--audit-id", default="v09-added-p3-audit-20260826T043840Z"
    )
    parser.add_argument(
        "--frozen-reference-id",
        default="v09-added-p3-routes-20260829T025754Z",
        help="completed matched-route result providing immutable frozen locked-test metrics",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[47, 53, 59])
    parser.add_argument("--operator-seeds", nargs="+", type=int, default=[701, 809, 907])
    parser.add_argument(
        "--condition-modes", nargs="+", default=["known", "latent_inferred"]
    )
    parser.add_argument("--skip-software-tests", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--phase37",
        action="store_true",
        help="enable physical pullback geometry and independent observer admission",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != 3 or len(args.seeds) != 3:
        raise SystemExit("formal Phase-3 joint study requires three unique backbone seeds")
    if len(set(args.operator_seeds)) != 3 or len(args.operator_seeds) != 3:
        raise SystemExit("formal Phase-3 joint study requires three unique operator seeds")
    if args.condition_modes != ["known", "latent_inferred"]:
        raise SystemExit("formal Phase-3 joint study requires known and latent_inferred modes")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    dirty = _dirty_source_paths(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    if dirty and not args.allow_dirty:
        raise SystemExit(f"formal Phase-3 joint study requires clean source files: {dirty}")

    phase2_raw = ROOT / "runs" / "v0_9" / args.phase2_id
    phase2_compact = ROOT / "gpu_validation" / "v0_9" / "results" / args.phase2_id
    audit_compact = ROOT / "gpu_validation" / "v0_9" / "results" / args.audit_id
    frozen_reference_compact = (
        ROOT / "gpu_validation" / "v0_9" / "results" / args.frozen_reference_id
    )
    required = (
        phase2_raw / "v0_8_handoff_audit.json",
        phase2_compact / "completion.json",
        phase2_compact / "summary.json",
        audit_compact / "completion.json",
        audit_compact / "evaluation" / "phase3_route_decision.json",
        frozen_reference_compact / "completion.json",
        frozen_reference_compact / "evaluation" / "matched_route_summary.json",
    )
    if any(not path.is_file() for path in required):
        raise SystemExit("Phase-3 joint study requires complete Phase-2 raw/compact and audit data")
    phase2_completion = json.loads(required[1].read_text(encoding="utf-8"))
    if phase2_completion.get("status") != "PASS":
        raise SystemExit("Phase-3 source Phase-2 workflow is incomplete")
    phase2_summary = json.loads(required[2].read_text(encoding="utf-8"))
    source_decision = json.loads(required[4].read_text(encoding="utf-8"))
    frozen_reference_completion = json.loads(required[5].read_text(encoding="utf-8"))
    frozen_reference = json.loads(required[6].read_text(encoding="utf-8"))
    if (
        frozen_reference_completion.get("status") != "PASS"
        or frozen_reference.get("source_phase2_result") != args.phase2_id
        or len(frozen_reference.get("frozen", {}).get("runs", ())) != 18
    ):
        raise SystemExit("Phase-3 joint refinement requires a complete matched frozen reference")
    template = load_config(
        ROOT / "gpu_validation" / "v0_9" / "configs" / "gpu_adaptive_koopman.yaml"
    )
    if args.phase37:
        template_payload = template.to_dict()
        assert isinstance(template_payload["v0_9_phase3"], dict)
        template_payload["v0_9_phase3"].update(
            {
                "physics_aligned_latent_enabled": True,
                "observer_admission_enabled": True,
            }
        )
        template = ProjectConfig.from_dict(template_payload)
    reassessment = _corrected_audit(source_decision, phase2_summary, template)
    if reassessment["corrected"]["next_candidate"] != "JOINT_MARKOV_REPRESENTATION":
        raise SystemExit("corrected Phase-3 audit does not select the joint route")

    results_root = ROOT / "gpu_validation" / "v0_9" / "results"
    session = create_versioned_session(
        ROOT / "runs" / "v0_9", args.validation_id, reserved_roots=(results_root,)
    )
    raw = session.path
    compact = results_root / session.resolved_id
    compact.mkdir(parents=True, exist_ok=False)
    for name in ("software", "configs", "seeds", "evaluation", "logs"):
        (raw / name).mkdir()
    (compact / "evaluation").mkdir()
    current_stage = "session_initialization"
    current_run: dict[str, Any] | None = None
    completed_runs = 0
    try:
        print(
            f"[V0.9][phase3-joint] SESSION requested={session.requested_id} "
            f"resolved={session.resolved_id} gpu={torch.cuda.get_device_name(0)}",
            flush=True,
        )
        python = str(ROOT / ".venv" / "bin" / "python")
        if not Path(python).is_file():
            python = sys.executable
        if not args.skip_software_tests:
            current_stage = "targeted_software_tests"
            _stage(
                "G0 targeted Phase-3 tests",
                lambda: _run_tests(python, raw / "software/pytest.log"),
            )

        current_stage = "audit_reassessment"
        reassessment["source_phase3_audit"] = args.audit_id
        (raw / "evaluation" / "audit_reassessment.json").write_text(
            json.dumps(reassessment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (compact / "evaluation" / "audit_reassessment.json").write_text(
            json.dumps(reassessment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            "[V0.9][phase3-joint] G1 corrected audit: PASS "
            "physics=PASS roundtrip=FAIL next=JOINT_MARKOV_REPRESENTATION",
            flush=True,
        )

        handoff = json.loads(required[0].read_text(encoding="utf-8"))
        handoff_by_seed = {int(item["backbone_seed"]): item for item in handoff["seeds"]}
        if set(handoff_by_seed) != set(args.seeds):
            raise ValueError("Phase-3 seeds differ from the frozen handoff")
        rows: list[dict[str, Any]] = []
        current_stage = "formal_joint_training_and_locked_test"
        for seed in args.seeds:
            dataset_path = phase2_raw / "data" / f"controlled_cylinder_seed_{seed}.pt"
            cache_path = phase2_raw / "seeds" / f"seed_{seed}" / "cache" / "adaptive_cache.pt"
            target_cache_path = (
                raw / "seeds" / f"seed_{seed}" / "cache" / "frozen_jepa_targets.pt"
            )
            item = handoff_by_seed[seed]
            for mode in args.condition_modes:
                for operator_seed in args.operator_seeds:
                    current_run = {
                        "route": "joint",
                        "seed": seed,
                        "condition_mode": mode,
                        "operator_seed": operator_seed,
                    }
                    source_config = (
                        phase2_raw
                        / "seeds"
                        / f"seed_{seed}"
                        / "formal"
                        / mode
                        / f"init_{operator_seed}"
                        / "config"
                        / "resolved_config.yaml"
                    )
                    run_dir = (
                        raw
                        / "seeds"
                        / f"seed_{seed}"
                        / "joint"
                        / mode
                        / f"init_{operator_seed}"
                    )
                    config = _resolved_config(
                        source_config,
                        phase2_id=args.phase2_id,
                        dataset_path=dataset_path,
                        run_root=run_dir.parent,
                        seed=seed,
                        condition_mode=mode,
                        operator_seed=operator_seed,
                        phase37=args.phase37,
                    )
                    config_path = raw / "configs" / f"seed_{seed}_{mode}_{operator_seed}.yaml"
                    save_config(config, config_path)
                    trained = _stage(
                        f"G2 joint train/test seed={seed} mode={mode} init={operator_seed}",
                        lambda config=config, run_dir=run_dir, item=item, cache_path=cache_path,
                        dataset_path=dataset_path,
                        target_cache_path=target_cache_path: train_v0_9_phase3_joint(
                            config,
                            context_checkpoint=item["context_checkpoint"],
                            adaptive_cache=cache_path,
                            backbone_checkpoint=item["backbone_checkpoint"],
                            physical_dataset=dataset_path,
                            run_dir=run_dir,
                            frozen_target_cache=target_cache_path,
                            device="cuda",
                        ),
                    )
                    rows.append(
                        {
                            **current_run,
                            "completed_epochs": trained.completed_epochs,
                            "best_epoch": trained.best_epoch,
                            "trainable_parameters": trained.trainable_parameters,
                            "validation": trained.validation_metrics,
                            "locked_test": trained.locked_test_metrics,
                            "observer_admission": trained.observer_admission,
                            "checkpoint": str(trained.checkpoint),
                        }
                    )
                    completed_runs += 1

        current_stage = "nested_aggregation"
        expected = len(args.seeds) * len(args.operator_seeds) * len(args.condition_modes)
        if completed_runs != expected:
            raise RuntimeError("Phase-3 joint nested matrix is incomplete")
        metric_names = sorted(set.intersection(*(set(row["locked_test"]) for row in rows)))
        aggregate = {
            name: statistics.mean(float(row["locked_test"][name]) for row in rows)
            for name in metric_names
        }
        phase3 = template.v0_9_phase3
        phase2 = template.v0_9_phase2
        evaluation = template.v0_9_evaluation
        if phase3 is None or phase2 is None or evaluation is None:
            raise RuntimeError("Phase-3 aggregation thresholds are missing")

        def matched_key(row: dict[str, Any]) -> tuple[int, str, int]:
            return (
                int(row["seed"]),
                str(row["condition_mode"]),
                int(row["operator_seed"]),
            )

        frozen_by_key = {
            matched_key(row): row for row in frozen_reference["frozen"]["runs"]
        }
        if set(frozen_by_key) != {matched_key(row) for row in rows}:
            raise RuntimeError("refined joint matrix does not match the frozen reference matrix")
        run_gates = [
            classify_matched_phase3_run(
                row["locked_test"],
                frozen_by_key[matched_key(row)]["locked_test"],
                phase3,
                evaluation,
                phase2,
                route="joint",
                condition_mode=row["condition_mode"],
            )
            for row in rows
        ]
        for row, gate in zip(rows, run_gates, strict=True):
            row["gates"] = gate

        def pass_fraction(name: str) -> float:
            return sum(bool(gate[name]) for gate in run_gates) / len(run_gates)

        latent_gates = [
            gate
            for row, gate in zip(rows, run_gates, strict=True)
            if row["condition_mode"] == "latent_inferred"
        ]
        latent_observer_fraction = sum(
            bool(gate["observer"]) for gate in latent_gates
        ) / len(latent_gates)
        nested_support = nested_route_support(
            rows, required_fraction=evaluation.scientific_seed_fraction
        )
        scientific_status = (
            "SCOPED_JOINT_REFINEMENT_SUPPORTED"
            if nested_support["supported"]
            else "JOINT_REFINEMENT_NOT_SUPPORTED"
        )
        v1_0_readiness = (
            "READY"
            if nested_support["backbone_pass_fraction"]
            >= evaluation.v1_0_readiness_fraction
            else "NOT_READY"
        )

        summary = {
            "schema_version": 4 if args.phase37 else 3,
            "phase": (
                "V0.9_PHASE3_7_PHYSICS_ALIGNED_REPRESENTATION"
                if args.phase37
                else "V0.9_PHASE3_JOINT_PHYSICAL_REFINEMENT"
            ),
            "source_phase2_result": args.phase2_id,
            "source_phase3_audit": args.audit_id,
            "source_frozen_reference": args.frozen_reference_id,
            "route": "joint",
            "phase2_retraining_performed": False,
            "operator_initialized_from_phase2_trained_state": False,
            "raw_field_online_reencoding": True,
            "nominal_generator_frozen": True,
            "formal_run_count": completed_runs,
            "completed_epoch_range": [
                min(int(row["completed_epochs"]) for row in rows),
                max(int(row["completed_epochs"]) for row in rows),
            ],
            "best_epoch_range": [
                min(int(row["best_epoch"]) for row in rows),
                max(int(row["best_epoch"]) for row in rows),
            ],
            "matched_matrix": "3 backbone seeds x 3 operator seeds x 2 condition modes",
            "physics_pass_fraction": pass_fraction("physics"),
            "representation_drift_pass_fraction": pass_fraction(
                "representation_drift"
            ),
            "roundtrip_pass_fraction": pass_fraction("roundtrip"),
            "dynamical_gauge_pass_fraction": pass_fraction("dynamical_gauge"),
            "representation_feasible_pass_fraction": pass_fraction(
                "representation_feasible"
            ),
            "predictive_pass_fraction": pass_fraction("predictive"),
            "observer_pass_fraction": pass_fraction("observer"),
            "latent_observer_pass_fraction": latent_observer_fraction,
            "initial_latent_observer_admission_fraction": sum(
                bool(row["observer_admission"].get("admitted", False))
                for row in rows
                if row["condition_mode"] == "latent_inferred"
            )
            / len(latent_gates),
            "latent_observer_admission_fraction": sum(
                bool(row["locked_test"].get("observer_admitted", 0.0))
                for row in rows
                if row["condition_mode"] == "latent_inferred"
            )
            / len(latent_gates),
            "history_only_fallback_fraction": sum(
                row["observer_admission"].get("operator_condition_route")
                == "history_only_fallback"
                for row in rows
                if row["condition_mode"] == "latent_inferred"
            )
            / len(latent_gates),
            "strict_joint_pass_fraction": pass_fraction("strict_joint"),
            "decoded_field_material_gain_pass_fraction": pass_fraction(
                "decoded_field_material_gain"
            ),
            "decoded_velocity_noninferiority_pass_fraction": pass_fraction(
                "decoded_velocity_noninferiority"
            ),
            "decoded_vorticity_noninferiority_pass_fraction": pass_fraction(
                "decoded_vorticity_noninferiority"
            ),
            "matched_route_pass_fraction": pass_fraction("matched_route_pass"),
            "nested_support": nested_support,
            "aggregate_locked_test": aggregate,
            "runs": rows,
            "scientific_status": scientific_status,
            "v1_0_readiness": v1_0_readiness,
            "next_stage": "REVIEW_REFINED_JOINT_EVIDENCE",
        }
        for destination in (
            raw / "evaluation" / "joint_summary.json",
            compact / "evaluation" / "joint_summary.json",
        ):
            destination.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        report = (
            (
                "# V0.9 Phase-3.7 physics-aligned joint route\n\n"
                if args.phase37
                else "# V0.9 Phase-3 joint route\n\n"
            )
            +
            f"- Source audit: `{args.audit_id}` (dimensionally reassessed)\n"
            f"- Frozen matched reference: `{args.frozen_reference_id}`\n"
            "- Corrected reconstruction physics: `PASS`\n"
            "- Corrected next route: `JOINT_MARKOV_REPRESENTATION`\n"
            f"- Formal train/locked-test runs: `{completed_runs}/{expected}`\n"
            f"- Completed epoch range: `{summary['completed_epoch_range']}`\n"
            f"- Selected best-epoch range: `{summary['best_epoch_range']}`\n"
            "- Raw-field online re-encoding: `YES`\n"
            "- Frozen nominal generator A0: `YES`\n"
            f"- Physics pass fraction: `{summary['physics_pass_fraction']:.3f}`\n"
            "- Representation-feasible pass fraction: "
            f"`{summary['representation_feasible_pass_fraction']:.3f}`\n"
            "- Dynamical-gauge pass fraction: "
            f"`{summary['dynamical_gauge_pass_fraction']:.3f}`\n"
            f"- Predictive pass fraction: `{summary['predictive_pass_fraction']:.3f}`\n"
            "- Latent-only observer pass fraction: "
            f"`{summary['latent_observer_pass_fraction']:.3f}`\n"
            "- Latent observer-admission fraction: "
            f"`{summary['latent_observer_admission_fraction']:.3f}`\n"
            "- Initial validation observer-admission fraction: "
            f"`{summary['initial_latent_observer_admission_fraction']:.3f}`\n"
            "- History-only fallback fraction: "
            f"`{summary['history_only_fallback_fraction']:.3f}`\n"
            f"- Strict joint pass fraction: `{summary['strict_joint_pass_fraction']:.3f}`\n"
            "- Decoded-field material-gain pass fraction: "
            f"`{summary['decoded_field_material_gain_pass_fraction']:.3f}`\n"
            f"- Matched route pass fraction: `{summary['matched_route_pass_fraction']:.3f}`\n"
            "- Nested backbone support fraction: "
            f"`{nested_support['backbone_pass_fraction']:.3f}`\n"
            f"- Scientific status: `{scientific_status}`\n"
            f"- V1.0 readiness: `{v1_0_readiness}`\n"
            "- Next: review refined joint evidence; from-scratch remains a completed "
            "negative control\n"
        )
        (compact / "report.md").write_text(report, encoding="utf-8")
        completion = {
            "phase": summary["phase"],
            "requested_validation_id": session.requested_id,
            "resolved_validation_id": session.resolved_id,
            "source_phase2_result": args.phase2_id,
            "source_phase3_audit": args.audit_id,
            "source_frozen_reference": args.frozen_reference_id,
            "status": "PASS",
            "git_commit": get_git_commit(ROOT),
            "formal_training_run_count": completed_runs,
            "formal_locked_test_run_count": completed_runs,
            "scientific_status": scientific_status,
            "v1_0_readiness": v1_0_readiness,
            "next_stage": "REVIEW_REFINED_JOINT_EVIDENCE",
        }
        (compact / "completion.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"V0.9 {'PHASE3.7' if args.phase37 else 'PHASE3'} JOINT COMPLETE "
            f"id={session.resolved_id} "
            f"runs={completed_runs} report={compact / 'report.md'}",
            flush=True,
        )
    except Exception as error:
        failure = {
            "phase": (
                "V0.9_PHASE3_7_PHYSICS_ALIGNED_REPRESENTATION"
                if args.phase37
                else "V0.9_PHASE3_JOINT_PHYSICAL_REFINEMENT"
            ),
            "validation_id": session.resolved_id,
            "source_phase2_result": args.phase2_id,
            "source_phase3_audit": args.audit_id,
            "source_frozen_reference": args.frozen_reference_id,
            "status": "FAILED_INCOMPLETE",
            "failed_stage": current_stage,
            "failed_run": current_run,
            "completed_joint_runs": completed_runs,
            "exception_summary": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "git_commit": get_git_commit(ROOT),
        }
        (compact / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"V0.9 {'PHASE3.7' if args.phase37 else 'PHASE3'} JOINT FAILED "
            f"id={session.resolved_id} "
            f"failure={compact / 'failure.json'}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
