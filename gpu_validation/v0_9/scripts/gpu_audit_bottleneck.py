#!/usr/bin/env python3
"""One-command, non-training V0.9 Stage-1 diagnostic with complete failure artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Must be set before the first CUDA context is created; preserve an explicit server setting.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch  # noqa: E402

from eval.audit_v0_9_bottleneck import (  # noqa: E402
    _write_json,
    audit_checkpoint,
    relocate_run_path,
    write_audit_report,
)
from jka_model.residual.cache import file_sha256  # noqa: E402
from jka_model.utils import create_versioned_session, get_git_commit  # noqa: E402


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("invalid versioned identifier")
    return value


def run_audit(args: argparse.Namespace, *, root: Path = ROOT) -> dict[str, Any]:
    compact_root = root / "gpu_validation" / "v0_9" / "results"
    session = create_versioned_session(
        root / "runs" / "v0_9", args.validation_id, reserved_roots=(compact_root,)
    )
    raw, compact = session.path, compact_root / session.resolved_id
    compact.mkdir(parents=True, exist_ok=False)
    for folder in ("logs", "configs", "cells", "evaluation"):
        (raw / folder).mkdir()
    completed: list[dict[str, Any]] = []
    stage = "preflight"
    active_cell = None
    log_path = raw / "logs" / "audit.log"

    def announce(message: str) -> None:
        line = f"[V0.9][bottleneck] {message}"
        print(line, flush=True)
        with log_path.open("a") as stream:
            stream.write(line + "\n")

    metadata = {
        "requested_id": args.validation_id,
        "validation_id": session.resolved_id,
        "source_id": args.source_id,
        "git_commit": get_git_commit(root),
        "device": args.device,
        "training_performed": False,
        "scientific_acceptance": "NOT_EVALUATED",
    }
    _write_json(raw / "configs" / "audit_request.json", metadata)
    try:
        announce(f"SESSION id={session.resolved_id}: START")
        announce("preflight: START")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run on the server with checkpoint backups")
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True
        )
        _write_json(raw / "configs" / "working_tree.json", {"porcelain": dirty})
        _write_json(
            raw / "configs" / "source_hashes.json",
            {
                str(path.relative_to(root)): file_sha256(path)
                for folder in (root / "src", root / "gpu_validation" / "v0_9" / "scripts")
                for path in sorted(folder.rglob("*.py"))
            },
        )
        # Read-only diagnostic permits a dirty tree but records it. It is never a
        # formal scientific pass, and no dirty-result-path gate is involved.
        source = compact_root / args.source_id
        completion = json.loads((source / "completion.json").read_text())
        if completion.get("status") != "PASS":
            raise ValueError("source workflow is incomplete")
        summary_path = source / "evaluation" / "joint_summary.json"
        summary = json.loads(summary_path.read_text())
        phase2_id = _identifier(summary["source_phase2_result"])
        rows = summary["runs"]
        keys = [(r["seed"], r["condition_mode"], r["operator_seed"]) for r in rows]
        if not keys or len(set(keys)) != len(keys) or len(rows) != summary["formal_run_count"]:
            raise ValueError("source matrix contains missing or duplicate cells")
        expected = {
            (s, m, i)
            for s in {k[0] for k in keys}
            for m in {k[1] for k in keys}
            for i in {k[2] for k in keys}
        }
        if set(keys) != expected or {k[1] for k in keys} != {"known", "latent_inferred"}:
            raise ValueError("source must contain a complete paired known/latent matrix")
        handoff_path = root / "runs" / "v0_9" / phase2_id / "v0_8_handoff_audit.json"
        handoff = json.loads(handoff_path.read_text())
        by_seed = {int(item["backbone_seed"]): item for item in handoff["seeds"]}
        for row in rows:
            checkpoint = relocate_run_path(row["checkpoint"], root)
            if not checkpoint.is_relative_to((root / "runs" / "v0_9" / args.source_id).resolve()):
                raise ValueError("checkpoint does not belong to the declared source session")
            item = by_seed[row["seed"]]
            for key in ("context_checkpoint", "backbone_checkpoint"):
                relocate_run_path(item[key], root)
            for suffix in (
                f"data/controlled_cylinder_seed_{row['seed']}.pt",
                f"seeds/seed_{row['seed']}/cache/adaptive_cache.pt",
            ):
                relocate_run_path(f"runs/v0_9/{phase2_id}/{suffix}", root)
        metadata.update(
            source_summary_sha256=file_sha256(summary_path),
            source_phase2_id=phase2_id,
            source_handoff_sha256=file_sha256(handoff_path),
            expected_cells=len(rows),
            torch_version=torch.__version__,
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        )
        _write_json(raw / "configs" / "audit_request.json", metadata)
        announce(f"preflight: PASS cells={len(rows)}")
        stage = "targeted_software_tests"
        if not args.skip_tests:
            announce(f"{stage}: START")
            command = [sys.executable, "-m", "pytest", "-q", "tests/test_v0_9_bottleneck_audit.py"]
            with (raw / "logs" / "pytest.log").open("w") as stream:
                process = subprocess.Popen(
                    command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                if process.wait():
                    raise RuntimeError("targeted bottleneck audit tests failed")
            announce(f"{stage}: PASS")
        else:
            announce(f"{stage}: SKIPPED (explicit request)")
        manifests: dict[int, list[dict[str, Any]]] = {}
        for row in sorted(rows, key=lambda r: (r["seed"], r["condition_mode"], r["operator_seed"])):
            active_cell = {k: row[k] for k in ("seed", "condition_mode", "operator_seed")}
            label = f"seed_{row['seed']}/{row['condition_mode']}/init_{row['operator_seed']}"
            stage = f"same_sample_replay/{label}"
            announce(f"{stage}: START")
            cell_output = raw / "cells" / label
            result = audit_checkpoint(
                root=root,
                row=row,
                phase2_id=phase2_id,
                handoff=by_seed[row["seed"]],
                output=cell_output,
                device=args.device,
            )
            manifest = json.loads((cell_output / "window_manifest.json").read_text())
            previous = manifests.setdefault(row["seed"], manifest)
            if previous != manifest:
                raise ValueError("source cells do not use identical per-seed audit windows")
            completed.append(result)
            _write_json(
                raw / "evaluation" / "progress.json",
                {**metadata, "completed_cells": len(completed)},
            )
            announce(f"{stage}: PASS windows={result['window_count']}")
        stage = "aggregation_and_report"
        announce(f"{stage}: START")
        report = {
            **metadata,
            "status": "PASS",
            "completed_cells": len(completed),
            "runs": completed,
        }
        _write_json(compact / "audit.json", report)
        _write_json(raw / "evaluation" / "audit.json", report)
        write_audit_report(compact / "report.md", completed, source_id=args.source_id)
        _write_json(compact / "completion.json", {k: v for k, v in report.items() if k != "runs"})
        announce(f"{stage}: PASS")
        announce(f"COMPLETE (diagnostic only) report={compact / 'report.md'}")
        return report
    except Exception as error:
        failure = {
            **metadata,
            "status": "FAILED_INCOMPLETE",
            "failed_stage": stage,
            "failed_cell": active_cell,
            "completed_cells": len(completed),
            "exception_summary": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "preserved_runs": str(raw),
        }
        _write_json(compact / "failure.json", failure)
        _write_json(compact / "completion.json", failure)
        _write_json(compact / "partial_audit.json", {**failure, "runs": completed})
        (compact / "report.md").write_text(
            f"# V0.9 Stage-1 audit: FAILED_INCOMPLETE\n\nStage: `{stage}`.\n\n"
            f"{failure['exception_summary']}\n\nCompleted cells: {len(completed)}. "
            "No scientific conclusion; no source artifacts modified or retraining performed.\n\n"
            "See `failure.json` and `partial_audit.json`; raw logs/windows remain under "
            f"`{raw}`.\n",
            encoding="utf-8",
        )
        announce(f"{stage}: FAIL {error}; report={compact / 'report.md'}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", type=_identifier, required=True)
    parser.add_argument(
        "--validation-id",
        type=_identifier,
        default="v09-stage1-audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--skip-tests", action="store_true", help="skip only new targeted software tests"
    )
    run_audit(parser.parse_args())


if __name__ == "__main__":
    main()
