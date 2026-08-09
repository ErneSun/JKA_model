"""Trajectory-safe conversion from full records to model-ready windows."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from jka_model.contracts import ProblemBatch
from jka_model.data.datasets import TrajectoryRecord
from jka_model.data.normalization import ChannelStandardizer


class TrajectoryWindowDataset(Dataset[ProblemBatch]):
    """Enumerate windows wholly contained within individual trajectories."""

    def __init__(
        self,
        records: Sequence[TrajectoryRecord],
        *,
        history: int,
        horizon: int,
        normalizer: ChannelStandardizer,
    ) -> None:
        if history < 2 or horizon < 1:
            raise ValueError("history must be at least 2 and horizon must be positive")
        if not normalizer.is_fitted:
            raise ValueError("window dataset requires a fitted normalizer")
        self.records = tuple(records)
        if not self.records:
            raise ValueError("window dataset requires at least one trajectory")
        self.history = history
        self.horizon = horizon
        self.normalizer = normalizer
        self._references: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            first_t = history - 1
            last_t = record.num_steps - horizon
            if last_t < first_t:
                raise ValueError(
                    f"trajectory {record.trajectory_id!r} is too short for history/horizon"
                )
            self._references.extend(
                (record_index, t) for t in range(first_t, last_t + 1)
            )

    def __len__(self) -> int:
        return len(self._references)

    def __getitem__(self, index: int) -> ProblemBatch:
        record_index, t = self._references[index]
        record = self.records[record_index]
        context_start = t - self.history + 1
        future_stop = t + self.horizon + 1
        context_raw = record.states_raw[context_start : t + 1]
        future_raw = record.states_raw[t + 1 : future_stop]
        history_actions = (
            None if record.actions is None else record.actions[context_start:t]
        )
        future_actions = None if record.actions is None else record.actions[t : t + self.horizon]
        return ProblemBatch(
            context_states_raw=context_raw.unsqueeze(0),
            future_states_raw=future_raw.unsqueeze(0),
            context_states_model=self.normalizer.transform(context_raw).unsqueeze(0),
            future_states_model=self.normalizer.transform(future_raw).unsqueeze(0),
            history_actions=(None if history_actions is None else history_actions.unsqueeze(0)),
            future_actions=(None if future_actions is None else future_actions.unsqueeze(0)),
            history_dts=record.dts[context_start:t].unsqueeze(0),
            future_dts=record.dts[t : t + self.horizon].unsqueeze(0),
            mu_static=(None if record.mu_static is None else record.mu_static.unsqueeze(0)),
            coordinates=(
                None if record.coordinates is None else record.coordinates.unsqueeze(0)
            ),
            cell_weights=(
                None if record.cell_weights is None else record.cell_weights.unsqueeze(0)
            ),
            valid_mask=(None if record.valid_mask is None else record.valid_mask.unsqueeze(0)),
            trajectory_id=[record.trajectory_id],
        )


def _cat_optional(values: Sequence[Tensor | None], name: str) -> Tensor | None:
    present = [value is not None for value in values]
    if any(present) and not all(present):
        raise ValueError(f"cannot collate mixed presence for {name}")
    tensors = [value for value in values if value is not None]
    return None if not tensors else torch.cat(tensors, dim=0)


def collate_problem_batches(items: Sequence[ProblemBatch]) -> ProblemBatch:
    """Collate B=1 window items while preserving canonical field names."""
    if not items:
        raise ValueError("cannot collate an empty batch")
    identifiers: list[object] = []
    for item in items:
        if isinstance(item.trajectory_id, (list, tuple)):
            identifiers.extend(item.trajectory_id)
        else:
            identifiers.append(item.trajectory_id)
    return ProblemBatch(
        context_states_raw=torch.cat([item.context_states_raw for item in items], dim=0),
        future_states_raw=torch.cat([item.future_states_raw for item in items], dim=0),
        context_states_model=torch.cat([item.context_states_model for item in items], dim=0),
        future_states_model=torch.cat([item.future_states_model for item in items], dim=0),
        history_dts=torch.cat([item.history_dts for item in items], dim=0),
        future_dts=torch.cat([item.future_dts for item in items], dim=0),
        history_actions=_cat_optional([item.history_actions for item in items], "history_actions"),
        future_actions=_cat_optional([item.future_actions for item in items], "future_actions"),
        mu_static=_cat_optional([item.mu_static for item in items], "mu_static"),
        coordinates=_cat_optional([item.coordinates for item in items], "coordinates"),
        cell_weights=_cat_optional([item.cell_weights for item in items], "cell_weights"),
        valid_mask=_cat_optional([item.valid_mask for item in items], "valid_mask"),
        trajectory_id=identifiers,
    )
