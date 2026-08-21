#!/usr/bin/env python3
"""Re-evaluate a completed V0.9 session without repeating any training."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from eval.evaluate_v0_9 import evaluate_v0_9  # noqa: E402
from jka_model.adaptive import aggregate_v0_9_results  # noqa: E402
from jka_model.config import load_config  # noqa: E402
from jka_model.utils import get_git_commit  # noqa: E402


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required V0.9 reassessment artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def reassess_existing(validation_id: str, *, device: str = "cuda") -> dict[str, Any]:
    """Reopen only formal checkpoints and regenerate locked-test evidence/reporting."""
    session = ROOT / "runs" / "v0_9" / validation_id
    compact = ROOT / "gpu_validation" / "v0_9" / "results" / validation_id
    if not session.is_dir():
        raise FileNotFoundError(f"V0.9 raw session does not exist: {session}")
    compact.mkdir(parents=True, exist_ok=True)
    handoff = _read_json(session / "v0_8_handoff_audit.json")
    seed_artifacts = {
        int(item["backbone_seed"]): item for item in handoff.get("seeds", [])
    }
    if len(seed_artifacts) != 3:
        raise ValueError("V0.9 reassessment requires three audited handoff seeds")
    formal_runs = sorted(session.glob("seeds/seed_*/formal/*/init_*"))
    if len(formal_runs) != 18:
        raise ValueError("V0.9 reassessment requires all 18 formal runs")

    log_dir = session / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "reassessment.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8") as log_stream:
        sys.stdout = _Tee(original_stdout, log_stream)  # type: ignore[assignment]
        sys.stderr = _Tee(original_stderr, log_stream)  # type: ignore[assignment]
        try:
            print(
                f"[V0.9][reassess] locked-test reevaluation: START id={validation_id}",
                flush=True,
            )
            reevaluated = 0
            for run_dir in formal_runs:
                seed = int(run_dir.parents[2].name.removeprefix("seed_"))
                mode = run_dir.parent.name
                initialization = int(run_dir.name.removeprefix("init_"))
                config_path = run_dir / "config" / "resolved_config.yaml"
                resolved = load_config(config_path)
                if resolved.cylinder_wake_2d is None or not resolved.cylinder_wake_2d.dataset_path:
                    raise ValueError(f"formal run lacks observable dataset provenance: {run_dir}")
                artifacts = seed_artifacts[seed]
                print(
                    f"[V0.9][reassess] seed={seed} mode={mode} "
                    f"init={initialization}: START",
                    flush=True,
                )
                evaluate_v0_9(
                    resolved,
                    checkpoint=run_dir / "checkpoints" / "best_scientific_gate.pt",
                    context_checkpoint=artifacts["context_checkpoint"],
                    adaptive_cache=(
                        session / "seeds" / f"seed_{seed}" / "cache" / "adaptive_cache.pt"
                    ),
                    backbone_checkpoint=artifacts["backbone_checkpoint"],
                    physical_dataset=resolved.cylinder_wake_2d.dataset_path,
                    output_dir=run_dir,
                    device=device,
                )
                reevaluated += 1
                print(
                    f"[V0.9][reassess] seed={seed} mode={mode} "
                    f"init={initialization}: PASS",
                    flush=True,
                )
            print("[V0.9][reassess] locked-test reevaluation: PASS", flush=True)
            print("[V0.9][reassess] aggregation/report: START", flush=True)
            decision = aggregate_v0_9_results(session, compact)
            if not bool(decision.get("compact_audit", {}).get("complete")):
                raise RuntimeError("V0.9 reassessment compact audit is incomplete")
            print("[V0.9][reassess] aggregation/report: PASS", flush=True)

            completion_path = session / "completion.json"
            completion = _read_json(completion_path)
            completion.update(
                {
                    "scientific_status": decision["low_rank_operator_adaptation"],
                    "v1_0_readiness": decision["v1_0_readiness"],
                    "reassessment_status": "PASS",
                    "reassessment_git_commit": get_git_commit(ROOT),
                    "reassessment_time": datetime.now(timezone.utc).isoformat(),
                    "reassessment_evaluation_count": reevaluated,
                    "reassessment_training_count": 0,
                }
            )
            rendered = json.dumps(completion, indent=2, sort_keys=True) + "\n"
            completion_path.write_text(rendered, encoding="utf-8")
            (compact / "completion.json").write_text(rendered, encoding="utf-8")
            (compact / "reassessment.json").write_text(
                json.dumps(
                    {
                        "validation_id": validation_id,
                        "status": "PASS",
                        "training_count": 0,
                        "evaluation_count": reevaluated,
                        "decision_schema_version": decision["schema_version"],
                        "log": str(log_path),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"V0.9 REASSESSMENT COMPLETE id={validation_id} training=0 "
                f"science={decision['low_rank_operator_adaptation']} "
                f"report={compact / 'report.md'}",
                flush=True,
            )
            return decision
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate an existing V0.9 raw run without repeating training."
    )
    parser.add_argument("--validation-id", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    reassess_existing(args.validation_id, device=args.device)


if __name__ == "__main__":
    main()
