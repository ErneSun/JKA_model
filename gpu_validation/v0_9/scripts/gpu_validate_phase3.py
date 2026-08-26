#!/usr/bin/env python3
"""One-command Phase-3 entry audit without repeating Phase-2 training."""

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

from jka_model.adaptive import load_adaptive_cache  # noqa: E402
from jka_model.config import ProjectConfig, load_config, save_config  # noqa: E402
from jka_model.manifold import audit_representation_checkpoint  # noqa: E402
from jka_model.utils import create_versioned_session, get_git_commit  # noqa: E402


def _stage(label: str, action: Any) -> Any:
    print(f"[V0.9][phase3] {label}: START", flush=True)
    result = action()
    print(f"[V0.9][phase3] {label}: PASS", flush=True)
    return result


def _run_tests(python: str, log_path: Path) -> None:
    command = [python, "-m", "pytest", "-q", "tests/test_v0_9_phase3.py"]
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
        raise RuntimeError(f"Phase-3 targeted tests failed with exit code {code}")


def _dirty_source_paths(porcelain: str) -> list[str]:
    ignored = ("runs/", "gpu_validation/v0_8/results/", "gpu_validation/v0_9/results/")
    return [
        line[3:].split(" -> ")[-1]
        for line in porcelain.splitlines()
        if len(line) >= 4 and not line[3:].split(" -> ")[-1].startswith(ignored)
    ]


def _phase3_config(source: Path, phase2_id: str) -> ProjectConfig:
    config = load_config(source)
    payload = config.to_dict()
    phase3 = payload.get("v0_9_phase3")
    if not isinstance(phase3, dict):
        template = load_config(
            ROOT / "gpu_validation" / "v0_9" / "configs" / "gpu_adaptive_koopman.yaml"
        )
        if template.v0_9_phase3 is None:
            raise ValueError("Phase-3 template config section is missing")
        phase3 = template.v0_9_phase3.to_dict()
        payload["v0_9_phase3"] = phase3
    phase3.update({"enabled": True, "source_phase2_result": phase2_id})
    return ProjectConfig.from_dict(payload)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-id",
        default=datetime.now(timezone.utc).strftime("v09-added-p3-audit-%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--phase2-id",
        default="v09-added-p2-physical-20260824T105209Z",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[47, 53, 59])
    parser.add_argument("--skip-software-tests", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise SystemExit("Phase-3 formal audit requires exactly three unique seeds")
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
        raise SystemExit(f"formal Phase-3 audit requires clean source/config files: {dirty}")

    phase2_compact = ROOT / "gpu_validation" / "v0_9" / "results" / args.phase2_id
    phase2_raw = ROOT / "runs" / "v0_9" / args.phase2_id
    completion_path = phase2_compact / "completion.json"
    summary_path = phase2_compact / "summary.json"
    if not completion_path.is_file() or not summary_path.is_file() or not phase2_raw.is_dir():
        raise SystemExit("Phase-3 requires the complete compact and raw Phase-2 result")
    phase2_completion = json.loads(completion_path.read_text(encoding="utf-8"))
    phase2_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if phase2_completion.get("status") != "PASS" or not phase2_completion.get(
        "all_required_stages_completed"
    ):
        raise SystemExit("Phase-3 source Phase-2 workflow is incomplete")

    results_root = ROOT / "gpu_validation" / "v0_9" / "results"
    session = create_versioned_session(
        ROOT / "runs" / "v0_9", args.validation_id, reserved_roots=(results_root,)
    )
    raw = session.path
    compact = results_root / session.resolved_id
    compact.mkdir(parents=True, exist_ok=False)
    for name in ("software", "configs", "evaluation", "logs"):
        (raw / name).mkdir()
    (compact / "evaluation").mkdir()
    current_stage = "session_initialization"
    try:
        print(
            f"[V0.9][phase3] SESSION requested={session.requested_id} "
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

        current_stage = "phase2_evidence_freeze"
        source_audit = json.loads(
            (phase2_raw / "v0_8_handoff_audit.json").read_text(encoding="utf-8")
        )
        handoff_by_seed = {
            int(item["backbone_seed"]): item for item in source_audit.get("seeds", [])
        }
        if set(handoff_by_seed) != set(args.seeds):
            raise ValueError("Phase-3 seeds do not match the frozen Phase-2 source")
        frozen_evidence = {
            "phase2_id": args.phase2_id,
            "phase2_git_commit": phase2_completion.get("git_commit"),
            "phase2_scientific_status": phase2_completion.get("scientific_status"),
            "selected_rank": phase2_completion.get("selected_rank"),
            "physics_status": phase2_summary.get("physics_status"),
            "representation_physical_floor": phase2_summary.get(
                "representation_physical_floor"
            ),
            "condition_observer": phase2_summary.get("condition_observer"),
            "dynamic_operator_adaptation": phase2_summary.get(
                "dynamic_operator_adaptation"
            ),
        }
        (raw / "phase2_frozen_evidence.json").write_text(
            json.dumps(frozen_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[V0.9][phase3] G1 freeze Phase-2 evidence id={args.phase2_id}: PASS", flush=True)

        current_stage = "raw_field_representation_audit"
        audits: list[dict[str, Any]] = []
        for seed in args.seeds:
            config_source = (
                phase2_raw
                / "seeds"
                / f"seed_{seed}"
                / "formal"
                / "known"
                / "init_701"
                / "config"
                / "resolved_config.yaml"
            )
            if not config_source.is_file():
                raise ValueError(f"Phase-3 source config is missing for seed {seed}")
            config = _phase3_config(config_source, args.phase2_id)
            save_config(config, raw / "configs" / f"seed_{seed}_phase3.yaml")
            dataset_path = phase2_raw / "data" / f"controlled_cylinder_seed_{seed}.pt"
            cache = load_adaptive_cache(
                phase2_raw / "seeds" / f"seed_{seed}" / "cache" / "adaptive_cache.pt"
            )
            item = handoff_by_seed[seed]
            audit = _stage(
                f"G2 raw-field representation/tangent audit seed={seed}",
                lambda config=config, item=item, dataset_path=dataset_path,
                cache=cache, seed=seed: (
                    audit_representation_checkpoint(
                        config,
                        backbone_checkpoint=item["backbone_checkpoint"],
                        physical_dataset=dataset_path,
                        split_manifest=cache.split_manifest,
                        device="cuda",
                        output_path=raw / "evaluation" / f"seed_{seed}_representation_audit.json",
                    )
                ),
            )
            audits.append(audit.to_dict())

        current_stage = "phase3_route_decision"
        reconstruction_pass = all(
            row["reconstruction_physics_status"] == "PASS" for row in audits
        )
        roundtrip_pass = all(row["roundtrip_status"] == "PASS" for row in audits)
        tangent_pass = all(row["tangent_status"] == "PASS" for row in audits)
        observer_supported = phase2_summary.get("condition_observer") == "SUPPORTED"
        dynamic_supported = phase2_summary.get("dynamic_operator_adaptation") == "SUPPORTED"
        if not reconstruction_pass:
            next_candidate = "PHYSICAL_MANIFOLD_DECODER"
        elif not roundtrip_pass or not tangent_pass or not observer_supported:
            next_candidate = "JOINT_MARKOV_REPRESENTATION"
        elif not dynamic_supported:
            next_candidate = "HISTORY_NOT_REQUIRED_CONTROL"
        else:
            next_candidate = "FROZEN_REPRESENTATION_ADEQUATE"
        decision = {
            "schema_version": 1,
            "phase": "V0.9_PHASE3_ENTRY_AUDIT",
            "source_phase2_result": args.phase2_id,
            "routes_locked": ["frozen", "joint", "from_scratch"],
            "phase2_retraining_performed": False,
            "raw_field_online_reencoding_required_for_trainable_routes": True,
            "seed_count": len(audits),
            "reconstruction_physics_status": "PASS" if reconstruction_pass else "FAIL",
            "roundtrip_status": "PASS" if roundtrip_pass else "FAIL",
            "nominal_tangent_status": "PASS" if tangent_pass else "FAIL",
            "phase2_observer_status": phase2_summary.get("condition_observer"),
            "phase2_dynamic_status": phase2_summary.get("dynamic_operator_adaptation"),
            "next_candidate": next_candidate,
            "matched_route_training_status": "PENDING",
            "scientific_status": "AUDIT_COMPLETE_TRAINING_PENDING",
            "aggregate": {
                key: _mean(audits, key)
                for key in (
                    "reconstruction_relative_l2",
                    "roundtrip_nrmse",
                    "divergence_degradation",
                    "boundary_degradation",
                    "nominal_tangent_divergence",
                    "nominal_tangent_boundary",
                    "nominal_tangent_outer_boundary",
                )
            },
            "seeds": audits,
        }
        (raw / "evaluation" / "phase3_route_decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (compact / "evaluation" / "phase3_route_decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for source in sorted((raw / "evaluation").glob("seed_*_representation_audit.json")):
            (compact / "evaluation" / source.name).write_bytes(source.read_bytes())
        report = (
            "# V0.9 Phase-3 entry audit\n\n"
            f"- Source Phase-2 result: `{args.phase2_id}`\n"
            "- Phase-2 retraining: `NO`\n"
            f"- Reconstruction physics: `{decision['reconstruction_physics_status']}`\n"
            f"- Round-trip consistency: `{decision['roundtrip_status']}`\n"
            f"- Nominal tangent consistency: `{decision['nominal_tangent_status']}`\n"
            f"- Next candidate: `{next_candidate}`\n"
            "- Matched frozen/joint/from_scratch training: `PENDING`\n"
            "- Scientific status: `AUDIT_COMPLETE_TRAINING_PENDING`\n"
        )
        (compact / "report.md").write_text(report, encoding="utf-8")
        completion = {
            "requested_validation_id": session.requested_id,
            "resolved_validation_id": session.resolved_id,
            "source_phase2_result": args.phase2_id,
            "status": "PASS",
            "git_commit": get_git_commit(ROOT),
            "phase2_retraining_performed": False,
            "audit_seed_count": len(audits),
            "next_candidate": next_candidate,
            "matched_route_training_status": "PENDING",
            "scientific_status": "AUDIT_COMPLETE_TRAINING_PENDING",
        }
        (compact / "completion.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"V0.9 PHASE3 AUDIT COMPLETE id={session.resolved_id} "
            f"next={next_candidate} report={compact / 'report.md'}",
            flush=True,
        )
    except Exception as error:
        failure = {
            "validation_id": session.resolved_id,
            "source_phase2_result": args.phase2_id,
            "status": "FAILED_INCOMPLETE",
            "failed_stage": current_stage,
            "exception_summary": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "git_commit": get_git_commit(ROOT),
        }
        (compact / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"V0.9 PHASE3 AUDIT FAILED id={session.resolved_id} "
            f"failure={compact / 'failure.json'}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
