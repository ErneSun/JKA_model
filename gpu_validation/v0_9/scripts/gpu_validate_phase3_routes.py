#!/usr/bin/env python3
"""One-command Phase-3 frozen/joint/from-scratch matched comparison."""

from __future__ import annotations

import argparse
import json
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

from gpu_validation.v0_9.scripts.gpu_validate_phase3_joint import (  # noqa: E402
    _dirty_source_paths,
    _resolved_config,
    _run_tests,
)
from jka_model.config import load_config, save_config  # noqa: E402
from jka_model.manifold import (  # noqa: E402
    classify_matched_phase3_run,
    nested_route_support,
)
from jka_model.utils import create_versioned_session, get_git_commit  # noqa: E402
from train.train_v0_9_phase3 import (  # noqa: E402
    evaluate_v0_9_phase3_frozen,
    train_v0_9_phase3_from_scratch,
)


def _stage(label: str, action: Any) -> Any:
    print(f"[V0.9][phase3-routes] {label}: START", flush=True)
    result = action()
    print(f"[V0.9][phase3-routes] {label}: PASS", flush=True)
    return result


def _key(row: dict[str, Any]) -> tuple[int, str, int]:
    return int(row["seed"]), str(row["condition_mode"]), int(row["operator_seed"])


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    common = sorted(set.intersection(*(set(row["locked_test"]) for row in rows)))
    return {
        name: statistics.mean(float(row["locked_test"][name]) for row in rows)
        for name in common
    }


def _pass_fraction(rows: list[dict[str, Any]], name: str) -> float:
    return sum(bool(row["gates"][name]) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-id",
        default=datetime.now(timezone.utc).strftime("v09-added-p3-routes-%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--phase2-id", default="v09-added-p2-physical-20260824T105209Z"
    )
    parser.add_argument(
        "--audit-id", default="v09-added-p3-audit-20260826T043840Z"
    )
    parser.add_argument(
        "--joint-id", default="v09-added-p3-joint-r1-20260827T070153Z"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[47, 53, 59])
    parser.add_argument("--operator-seeds", nargs="+", type=int, default=[701, 809, 907])
    parser.add_argument(
        "--condition-modes", nargs="+", default=["known", "latent_inferred"]
    )
    parser.add_argument("--skip-software-tests", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise SystemExit("formal Phase-3 comparison requires three unique backbone seeds")
    if len(args.operator_seeds) != 3 or len(set(args.operator_seeds)) != 3:
        raise SystemExit("formal Phase-3 comparison requires three unique operator seeds")
    if args.condition_modes != ["known", "latent_inferred"]:
        raise SystemExit("formal Phase-3 comparison requires known and latent_inferred modes")
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
        raise SystemExit(f"formal Phase-3 comparison requires clean source files: {dirty}")

    phase2_raw = ROOT / "runs" / "v0_9" / args.phase2_id
    phase2_compact = ROOT / "gpu_validation" / "v0_9" / "results" / args.phase2_id
    audit_compact = ROOT / "gpu_validation" / "v0_9" / "results" / args.audit_id
    joint_compact = ROOT / "gpu_validation" / "v0_9" / "results" / args.joint_id
    required = (
        phase2_raw / "v0_8_handoff_audit.json",
        phase2_compact / "completion.json",
        phase2_compact / "summary.json",
        joint_compact / "completion.json",
        joint_compact / "evaluation" / "joint_summary.json",
        audit_compact / "completion.json",
    )
    if any(not path.is_file() for path in required):
        raise SystemExit("Phase-3 comparison requires complete Phase-2 and joint artifacts")
    phase2_completion = json.loads(required[1].read_text(encoding="utf-8"))
    joint_completion = json.loads(required[3].read_text(encoding="utf-8"))
    joint_summary = json.loads(required[4].read_text(encoding="utf-8"))
    audit_completion = json.loads(required[5].read_text(encoding="utf-8"))
    if phase2_completion.get("status") != "PASS":
        raise SystemExit("Phase-3 comparison source Phase-2 workflow is incomplete")
    if joint_completion.get("status") != "PASS" or joint_summary.get("formal_run_count") != 18:
        raise SystemExit("Phase-3 comparison requires a complete 18-run joint result")
    if joint_summary.get("source_phase2_result") != args.phase2_id:
        raise SystemExit("joint result and Phase-2 source do not match")
    if joint_summary.get("source_phase3_audit") != args.audit_id:
        raise SystemExit("joint result and Phase-3 audit do not match")
    if audit_completion.get("status") != "PASS":
        raise SystemExit("Phase-3 source audit is incomplete")
    handoff = json.loads(required[0].read_text(encoding="utf-8"))
    handoff_by_seed = {int(item["backbone_seed"]): item for item in handoff["seeds"]}
    if set(handoff_by_seed) != set(args.seeds):
        raise SystemExit("Phase-3 seeds differ from the frozen handoff")
    source_paths: list[Path] = []
    for seed in args.seeds:
        source_paths.extend(
            (
                phase2_raw / "data" / f"controlled_cylinder_seed_{seed}.pt",
                phase2_raw / "seeds" / f"seed_{seed}" / "cache" / "adaptive_cache.pt",
                Path(handoff_by_seed[seed]["context_checkpoint"]),
                Path(handoff_by_seed[seed]["backbone_checkpoint"]),
            )
        )
        for mode in args.condition_modes:
            for operator_seed in args.operator_seeds:
                source_root = (
                    phase2_raw
                    / "seeds"
                    / f"seed_{seed}"
                    / "formal"
                    / mode
                    / f"init_{operator_seed}"
                )
                source_paths.extend(
                    (
                        source_root / "config" / "resolved_config.yaml",
                        source_root / "checkpoints" / "best_scientific_gate.pt",
                    )
                )
    missing_sources = [str(path) for path in source_paths if not path.is_file()]
    if missing_sources:
        raise SystemExit(f"Phase-3 matched-route source artifacts are missing: {missing_sources}")

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
    completed_frozen = completed_scratch = 0
    try:
        print(
            f"[V0.9][phase3-routes] SESSION requested={session.requested_id} "
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
                lambda: _run_tests(python, raw / "software" / "pytest.log"),
            )

        frozen_rows: list[dict[str, Any]] = []
        scratch_rows: list[dict[str, Any]] = []
        current_stage = "matched_frozen_evaluation_and_from_scratch_training"
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
                        "seed": seed,
                        "condition_mode": mode,
                        "operator_seed": operator_seed,
                    }
                    source_root = (
                        phase2_raw
                        / "seeds"
                        / f"seed_{seed}"
                        / "formal"
                        / mode
                        / f"init_{operator_seed}"
                    )
                    source_config = source_root / "config" / "resolved_config.yaml"
                    source_checkpoint = (
                        source_root / "checkpoints" / "best_scientific_gate.pt"
                    )
                    frozen_dir = (
                        raw
                        / "seeds"
                        / f"seed_{seed}"
                        / "frozen"
                        / mode
                        / f"init_{operator_seed}"
                    )
                    scratch_dir = (
                        raw
                        / "seeds"
                        / f"seed_{seed}"
                        / "from_scratch"
                        / mode
                        / f"init_{operator_seed}"
                    )
                    frozen_config = _resolved_config(
                        source_config,
                        phase2_id=args.phase2_id,
                        dataset_path=dataset_path,
                        run_root=frozen_dir.parent,
                        seed=seed,
                        condition_mode=mode,
                        operator_seed=operator_seed,
                        route="joint",
                    )
                    scratch_config = _resolved_config(
                        source_config,
                        phase2_id=args.phase2_id,
                        dataset_path=dataset_path,
                        run_root=scratch_dir.parent,
                        seed=seed,
                        condition_mode=mode,
                        operator_seed=operator_seed,
                        route="from_scratch",
                    )
                    save_config(
                        scratch_config,
                        raw / "configs" / f"seed_{seed}_{mode}_{operator_seed}.yaml",
                    )
                    frozen = _stage(
                        f"G1 frozen locked-test seed={seed} mode={mode} init={operator_seed}",
                        lambda frozen_config=frozen_config, frozen_dir=frozen_dir,
                        source_checkpoint=source_checkpoint, item=item,
                        cache_path=cache_path, dataset_path=dataset_path,
                        target_cache_path=target_cache_path: evaluate_v0_9_phase3_frozen(
                            frozen_config,
                            adaptive_checkpoint=source_checkpoint,
                            context_checkpoint=item["context_checkpoint"],
                            adaptive_cache=cache_path,
                            backbone_checkpoint=item["backbone_checkpoint"],
                            physical_dataset=dataset_path,
                            run_dir=frozen_dir,
                            frozen_target_cache=target_cache_path,
                            device="cuda",
                        ),
                    )
                    frozen_rows.append(
                        {
                            **current_run,
                            "route": "frozen",
                            "trainable_parameters": 0,
                            "locked_test": frozen.locked_test_metrics,
                            "source_checkpoint": str(source_checkpoint),
                        }
                    )
                    completed_frozen += 1
                    scratch = _stage(
                        f"G2 from-scratch train/test seed={seed} mode={mode} init={operator_seed}",
                        lambda scratch_config=scratch_config, scratch_dir=scratch_dir,
                        item=item, cache_path=cache_path,
                        dataset_path=dataset_path: train_v0_9_phase3_from_scratch(
                            scratch_config,
                            context_checkpoint=item["context_checkpoint"],
                            adaptive_cache=cache_path,
                            backbone_checkpoint=item["backbone_checkpoint"],
                            physical_dataset=dataset_path,
                            run_dir=scratch_dir,
                            device="cuda",
                        ),
                    )
                    scratch_rows.append(
                        {
                            **current_run,
                            "route": "from_scratch",
                            "completed_epochs": scratch.completed_epochs,
                            "best_epoch": scratch.best_epoch,
                            "trainable_parameters": scratch.trainable_parameters,
                            "validation": scratch.validation_metrics,
                            "locked_test": scratch.locked_test_metrics,
                            "checkpoint": str(scratch.checkpoint),
                        }
                    )
                    completed_scratch += 1

        expected = len(args.seeds) * len(args.operator_seeds) * len(args.condition_modes)
        if completed_frozen != expected or completed_scratch != expected:
            raise RuntimeError("Phase-3 matched route matrix is incomplete")
        joint_rows = [dict(row) for row in joint_summary["runs"]]
        frozen_by_key = {_key(row): row for row in frozen_rows}
        if set(frozen_by_key) != {_key(row) for row in joint_rows} or set(frozen_by_key) != {
            _key(row) for row in scratch_rows
        }:
            raise RuntimeError("Phase-3 frozen/joint/from-scratch matrix keys differ")

        template = load_config(
            ROOT / "gpu_validation" / "v0_9" / "configs" / "gpu_adaptive_koopman.yaml"
        )
        phase3 = template.v0_9_phase3
        phase2 = template.v0_9_phase2
        evaluation = template.v0_9_evaluation
        if phase3 is None or phase2 is None or evaluation is None:
            raise RuntimeError("Phase-3 route thresholds are missing")
        for route, rows in (("joint", joint_rows), ("from_scratch", scratch_rows)):
            for row in rows:
                frozen_metrics = frozen_by_key[_key(row)]["locked_test"]
                row["gates"] = classify_matched_phase3_run(
                    row["locked_test"],
                    frozen_metrics,
                    phase3,
                    evaluation,
                    phase2,
                    route=route,
                    condition_mode=row["condition_mode"],
                )
        required_fraction = evaluation.scientific_seed_fraction
        joint_nested = nested_route_support(
            joint_rows, required_fraction=required_fraction
        )
        scratch_nested = nested_route_support(
            scratch_rows, required_fraction=required_fraction
        )
        if joint_nested["supported"]:
            decision = "ADOPT_SCOPED_JOINT"
        elif scratch_nested["supported"]:
            decision = "REPRESENTATION_ROUTE_INCOMPATIBILITY_SUPPORTED"
        else:
            decision = "NO_PHASE3_ROUTE_SUPPORTED"
        summary = {
            "schema_version": 1,
            "phase": "V0.9_PHASE3_MATCHED_ROUTES",
            "source_phase2_result": args.phase2_id,
            "source_phase3_audit": args.audit_id,
            "source_joint_result": args.joint_id,
            "matched_matrix": "3 backbone seeds x 3 operator seeds x 2 condition modes",
            "frozen_retraining_performed": False,
            "from_scratch_inherits_v0_8_validation": False,
            "raw_field_online_reencoding": True,
            "coordinate_invariant_diagnostics": [
                "centered_linear_cka",
                "orthogonal_procrustes_nrmse",
                "effective_rank",
            ],
            "frozen": {
                "formal_run_count": len(frozen_rows),
                "aggregate_locked_test": _aggregate(frozen_rows),
                "runs": frozen_rows,
            },
            "joint": {
                "formal_run_count": len(joint_rows),
                "matched_run_pass_fraction": _pass_fraction(
                    joint_rows, "matched_route_pass"
                ),
                "latent_observer_pass_fraction": sum(
                    bool(row["gates"]["observer"])
                    for row in joint_rows
                    if row["condition_mode"] == "latent_inferred"
                )
                / 9,
                "nested_support": joint_nested,
                "aggregate_locked_test": _aggregate(joint_rows),
                "runs": joint_rows,
            },
            "from_scratch": {
                "formal_run_count": len(scratch_rows),
                "matched_run_pass_fraction": _pass_fraction(
                    scratch_rows, "matched_route_pass"
                ),
                "latent_observer_pass_fraction": sum(
                    bool(row["gates"]["observer"])
                    for row in scratch_rows
                    if row["condition_mode"] == "latent_inferred"
                )
                / 9,
                "nested_support": scratch_nested,
                "aggregate_locked_test": _aggregate(scratch_rows),
                "runs": scratch_rows,
            },
            "scientific_decision": decision,
            "v1_0_readiness": "NOT_READY",
        }
        for destination in (
            raw / "evaluation" / "matched_route_summary.json",
            compact / "evaluation" / "matched_route_summary.json",
        ):
            destination.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        report = (
            "# V0.9 Phase-3 matched route comparison\n\n"
            f"- Source Phase-2: `{args.phase2_id}`\n"
            f"- Source joint: `{args.joint_id}`\n"
            f"- Frozen evaluations: `{len(frozen_rows)}/{expected}`\n"
            f"- From-scratch train/locked-test runs: `{len(scratch_rows)}/{expected}`\n"
            "- Joint matched pass fraction: "
            f"`{summary['joint']['matched_run_pass_fraction']:.3f}`\n"
            "- From-scratch matched pass fraction: "
            f"`{summary['from_scratch']['matched_run_pass_fraction']:.3f}`\n"
            "- Joint latent-only observer fraction: "
            f"`{summary['joint']['latent_observer_pass_fraction']:.3f}`\n"
            "- From-scratch latent-only observer fraction: "
            f"`{summary['from_scratch']['latent_observer_pass_fraction']:.3f}`\n"
            f"- Scientific decision: `{decision}`\n"
            "- V1.0 readiness: `NOT_READY`\n"
        )
        (compact / "report.md").write_text(report, encoding="utf-8")
        completion = {
            "requested_validation_id": session.requested_id,
            "resolved_validation_id": session.resolved_id,
            "status": "PASS",
            "git_commit": get_git_commit(ROOT),
            "source_phase2_result": args.phase2_id,
            "source_joint_result": args.joint_id,
            "formal_frozen_evaluation_count": completed_frozen,
            "formal_from_scratch_training_count": completed_scratch,
            "scientific_decision": decision,
            "v1_0_readiness": "NOT_READY",
        }
        (compact / "completion.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"V0.9 PHASE3 ROUTES COMPLETE id={session.resolved_id} "
            f"decision={decision} report={compact / 'report.md'}",
            flush=True,
        )
    except Exception as error:
        failure = {
            "validation_id": session.resolved_id,
            "status": "FAILED_INCOMPLETE",
            "failed_stage": current_stage,
            "failed_run": current_run,
            "completed_frozen_evaluations": completed_frozen,
            "completed_from_scratch_runs": completed_scratch,
            "exception_summary": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "git_commit": get_git_commit(ROOT),
        }
        (compact / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"V0.9 PHASE3 ROUTES FAILED id={session.resolved_id} "
            f"report={compact / 'failure.json'}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
