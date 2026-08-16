"""Exact optimizer-step-indexed EMA state for V0.6."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jka_model.config import EMAConfig
from jka_model.models import FieldJEPAKoopmanModel, ema_update_target


@dataclass(slots=True)
class EMATracker:
    config: EMAConfig
    total_updates: int
    update_count: int = 0
    current_tau: float | None = None

    def __post_init__(self) -> None:
        if self.total_updates < 1 or not 0 <= self.update_count <= self.total_updates:
            raise ValueError("EMA update counts are invalid")
        if self.current_tau is not None and not 0 <= self.current_tau <= 1:
            raise ValueError("current EMA tau must lie in [0,1]")

    def next_tau(self) -> float:
        if self.config.schedule == "constant" or self.total_updates == 1:
            return self.config.start_tau
        fraction = min(self.update_count / (self.total_updates - 1), 1.0)
        return self.config.start_tau + fraction * (self.config.end_tau - self.config.start_tau)

    def update_after_optimizer(self, model: FieldJEPAKoopmanModel) -> float:
        """Perform exactly one target update after one successful optimizer update."""
        if self.update_count >= self.total_updates:
            raise RuntimeError("EMA update count exceeds configured optimizer updates")
        tau = self.next_tau()
        ema_update_target(model, tau)
        self.update_count += 1
        self.current_tau = tau
        return tau

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "total_updates": self.total_updates,
            "update_count": self.update_count,
            "current_tau": self.current_tau,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> EMATracker:
        required = {"config", "total_updates", "update_count", "current_tau"}
        missing = required - set(state)
        if missing:
            raise ValueError(f"EMA state missing fields: {', '.join(sorted(missing))}")
        config_raw = state["config"]
        if not isinstance(config_raw, Mapping):
            raise ValueError("EMA config state must be a mapping")
        return cls(
            config=EMAConfig.from_dict(config_raw),
            total_updates=int(state["total_updates"]),
            update_count=int(state["update_count"]),
            current_tau=(None if state["current_tau"] is None else float(state["current_tau"])),
        )
