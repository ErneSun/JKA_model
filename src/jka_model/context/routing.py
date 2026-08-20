"""Evidence-owned V0.7 to V0.8 context-family routing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ContextRoute:
    residual_route: str
    context_family: str | None
    history_length: int | None
    source: Path
    source_payload: dict[str, Any]


def select_context_family(route: str) -> str | None:
    """Map only the updated V0.7 R1--R3 contract; R0 is deliberately invalid."""
    if route == "R2":
        return "instantaneous"
    if route == "R3":
        return "attention"
    if route in {"R1", "INCONCLUSIVE"}:
        return None
    if route == "R0":
        raise ValueError("R0 is obsolete: reclassify the V0.7 evidence under the R1-R3 router")
    raise ValueError(f"unknown V0.7 residual route: {route!r}")


def load_v0_7_route(path: str | Path) -> ContextRoute:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V0.7 route result must be a JSON object")
    route = str(payload.get("residual_route", ""))
    family = select_context_family(route)
    history = payload.get("locked_history_steps")
    history_length = None if history is None else int(history)
    if route == "R3" and (history_length is None or history_length < 2):
        raise ValueError("R3 requires a locked causal history length >= 2")
    if route == "R2":
        history_length = 1
    return ContextRoute(route, family, history_length, source, payload)
