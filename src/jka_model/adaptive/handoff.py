"""Strict V0.8-to-V0.9 readiness and artifact handoff audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class V08SeedHandoff:
    backbone_seed: int
    backbone_checkpoint: Path
    context_checkpoint: Path
    context_init_seed: int


@dataclass(frozen=True, slots=True)
class V08Handoff:
    validation_id: str
    raw_run: Path
    compact_result: Path
    route: str
    context_family: str
    seeds: tuple[V08SeedHandoff, ...]
    decision: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping JSON: {path}")
    return payload


def audit_v0_8_handoff(
    validation_id: str,
    *,
    runs_root: str | Path = "runs/v0_8",
    results_root: str | Path = "gpu_validation/v0_8/results",
) -> V08Handoff:
    raw = Path(runs_root) / validation_id
    compact = Path(results_root) / validation_id
    if not raw.is_dir() or not compact.is_dir():
        raise ValueError("V0.9 requires both raw and compact V0.8 artifacts")
    completion = _read_json(compact / "completion.json")
    decision = _read_json(compact / "evaluation" / "v0_8_scientific_decision.json")
    if completion.get("status") != "PASS" or not completion.get("all_required_stages_completed"):
        raise ValueError("V0.8 handoff is incomplete")
    if not bool(decision.get("v0_9_ready")):
        raise ValueError("V0.9 BLOCKED BY V0.8 READINESS")
    if float(decision.get("joint_v0_9_support_fraction", 0.0)) != 1.0:
        raise ValueError("V0.9 requires 3/3 jointly passing V0.8 backbone/data seeds")
    route = str(decision.get("v0_7_route_on_new_problem"))
    family = str(decision.get("context_family", "")).lower()
    if route not in {"R2", "R3"} or family not in {
        "instantaneous",
        "instantaneous_matched",
        "history_mlp",
        "attention",
    }:
        raise ValueError("V0.8 handoff lacks a locked R2/R3 context family")
    nested = decision.get("nested_seed_support")
    if not isinstance(nested, dict) or len(nested) != 3:
        raise ValueError("V0.8 strict readiness lacks three nested backbone records")
    handoffs: list[V08SeedHandoff] = []
    for seed_text, support in sorted(nested.items(), key=lambda item: int(item[0])):
        seed = int(seed_text)
        if not isinstance(support, dict) or not bool(support.get("v0_9_supported")):
            raise ValueError(f"V0.8 backbone seed {seed} does not jointly pass readiness")
        acceptance = _read_json(raw / "seeds" / f"seed_{seed}" / "backbone_acceptance.json")
        backbone = Path(str(acceptance.get("checkpoint", "")))
        if not backbone.is_file():
            candidates = sorted(
                (raw / "seeds" / f"seed_{seed}" / "backbone_runs").glob(
                    "*/checkpoints/best_forecast_post_warmup.pt"
                )
            )
            if len(candidates) != 1:
                raise ValueError(f"cannot resolve V0.8 backbone checkpoint for seed {seed}")
            backbone = candidates[0]
        context_runs = sorted(
            (raw / "seeds" / f"seed_{seed}" / "candidates" / family).glob("init_*")
        )
        scored: list[tuple[float, int, Path]] = []
        for run in context_runs:
            summary_path = run / "evaluation" / "training_summary.json"
            checkpoint_path = run / "checkpoints" / "best.pt"
            if not summary_path.is_file() or not checkpoint_path.is_file():
                continue
            summary = _read_json(summary_path)
            score = float(summary["validation"]["residual_standardized_mse"])
            initialization = int(run.name.removeprefix("init_"))
            scored.append((score, initialization, checkpoint_path))
        if not scored:
            raise ValueError(f"cannot resolve V0.8 context checkpoint for seed {seed}")
        _, initialization, context = min(scored)
        handoffs.append(V08SeedHandoff(seed, backbone.resolve(), context.resolve(), initialization))
    return V08Handoff(
        validation_id=validation_id,
        raw_run=raw.resolve(),
        compact_result=compact.resolve(),
        route=route,
        context_family=family,
        seeds=tuple(handoffs),
        decision=decision,
    )
