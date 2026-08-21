#!/usr/bin/env python3
"""Re-evaluate existing V0.8 checkpoints after decision/report-only changes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluate_v0_8 import evaluate_v0_8  # noqa: E402
from jka_model.context import aggregate_v0_8_results  # noqa: E402
from jka_model.utils import get_git_commit  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required reassessment artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def reassess_existing(validation_id: str, *, device: str = "cuda") -> dict[str, Any]:
    session = ROOT / "runs" / "v0_8" / validation_id
    compact = ROOT / "gpu_validation" / "v0_8" / "results" / validation_id
    if not session.is_dir():
        raise FileNotFoundError(f"V0.8 raw session does not exist: {session}")
    completion_path = session / "completion.json"
    completion = _read_json(completion_path)
    family = str(completion.get("final_context_family") or "")
    if not family:
        raise ValueError("existing V0.8 completion does not identify a context family")
    seed_dirs = sorted(session.glob("seeds/seed_*"))
    if len(seed_dirs) != 3:
        raise ValueError("V0.8 reassessment requires exactly three backbone seed directories")

    print(
        f"[V0.8][reassess] G6 locked-test reevaluation: START id={validation_id} family={family}",
        flush=True,
    )
    reevaluated = 0
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.removeprefix("seed_"))
        backbone = Path(_read_json(seed_dir / "backbone_acceptance.json")["checkpoint"])
        cache = seed_dir / "cache" / "residual_cache.pt"
        candidates = sorted((seed_dir / "candidates" / family).glob("init_*"))
        if len(candidates) != 3:
            raise ValueError(f"seed {seed} does not contain three selected-family checkpoints")
        for candidate in candidates:
            initialization = int(candidate.name.removeprefix("init_"))
            print(
                f"[V0.8][reassess] seed={seed} init={initialization}: START",
                flush=True,
            )
            evaluate_v0_8(
                candidate / "config" / "resolved_config.yaml",
                checkpoint=candidate / "checkpoints" / "best.pt",
                backbone_checkpoint=backbone,
                residual_cache=cache,
                output_dir=seed_dir / "contexts" / candidate.name,
                device=device,
            )
            reevaluated += 1
            print(
                f"[V0.8][reassess] seed={seed} init={initialization}: PASS",
                flush=True,
            )
    print("[V0.8][reassess] G6 locked-test reevaluation: PASS", flush=True)
    print("[V0.8][reassess] G7 aggregation/report: START", flush=True)
    decision = aggregate_v0_8_results(session, compact)
    if not bool(decision.get("compact_audit", {}).get("complete")):
        raise RuntimeError("V0.8 compact audit artifact set is incomplete")
    print("[V0.8][reassess] G7 aggregation/report: PASS", flush=True)

    completion.update(
        {
            "scientific_status": decision["dynamic_context"],
            "reassessment_status": "PASS",
            "reassessment_git_commit": get_git_commit(ROOT),
            "reassessment_time": datetime.now(timezone.utc).isoformat(),
            "reassessment_evaluation_count": reevaluated,
            "v0_9_operator_adaptation_readiness": decision["v0_9_operator_adaptation_readiness"],
        }
    )
    rendered = json.dumps(completion, indent=2, sort_keys=True) + "\n"
    completion_path.write_text(rendered, encoding="utf-8")
    (compact / "completion.json").write_text(rendered, encoding="utf-8")
    print(
        f"V0.8 REASSESSMENT COMPLETE id={validation_id} "
        f"dynamic_context={decision['dynamic_context']} "
        f"v0_9={decision['v0_9_operator_adaptation_readiness']} "
        f"report={compact / 'report.md'}",
        flush=True,
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate an existing V0.8 raw run without retraining G1-G5."
    )
    parser.add_argument("--validation-id", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    reassess_existing(args.validation_id, device=args.device)


if __name__ == "__main__":
    main()
