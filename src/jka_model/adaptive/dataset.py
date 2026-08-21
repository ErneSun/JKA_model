"""Causal V0.9 windows over the frozen-backbone adaptive cache."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from jka_model.adaptive.cache import AdaptiveCache


class AdaptiveWindowDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        cache: AdaptiveCache,
        split: str,
        history: int,
        *,
        shuffle_older_history: bool = False,
        shuffle_seed: int = 0,
    ) -> None:
        if split not in {"train", "validation", "test"} or history < 1:
            raise ValueError("invalid V0.9 split/history")
        self.trajectories = cache.select(split)
        if not self.trajectories:
            raise ValueError(f"V0.9 cache has no {split} trajectories")
        self.history = history
        self.shuffle_older_history = shuffle_older_history
        self.locations = [
            (trajectory_index, target_index)
            for trajectory_index, trajectory in enumerate(self.trajectories)
            for target_index in range(history - 1, trajectory.dts.shape[0])
        ]
        generator = torch.Generator().manual_seed(shuffle_seed)
        self.permutation = torch.randperm(len(self.locations), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, index: int) -> dict[str, object]:
        trajectory_index, target_index = self.locations[index]
        trajectory = self.trajectories[trajectory_index]
        start = target_index - self.history + 1
        history_z = trajectory.latents[start : target_index + 1].clone()
        history_dts = trajectory.dts[start:target_index].clone()
        previous_available = target_index >= self.history
        previous_target = target_index - 1 if previous_available else target_index
        previous_start = previous_target - self.history + 1
        previous_history_z = trajectory.latents[
            previous_start : previous_target + 1
        ].clone()
        previous_history_dts = trajectory.dts[previous_start:previous_target].clone()
        if self.shuffle_older_history and self.history > 1:
            source_index, source_target = self.locations[self.permutation[index]]
            source = self.trajectories[source_index]
            source_start = source_target - self.history + 1
            history_z[:-1] = source.latents[source_start:source_target]
            history_dts = source.dts[source_start:source_target].clone()
        return {
            "history_z": history_z,
            "history_dts": history_dts,
            "next_dt": trajectory.dts[target_index : target_index + 1],
            "context_parameters": trajectory.context_parameters,
            "condition": trajectory.conditions[target_index],
            "target_next": trajectory.latents[target_index + 1],
            "nominal_residual": trajectory.nominal_residuals[target_index],
            "schedule_type": trajectory.schedule_type,
            "trajectory_id": trajectory.trajectory_id,
            "target_index": target_index,
            "relative_transition_index": target_index - trajectory.transition_index,
            "previous_history_z": previous_history_z,
            "previous_history_dts": previous_history_dts,
            "previous_next_dt": trajectory.dts[previous_target : previous_target + 1],
            "previous_condition": trajectory.conditions[previous_target],
            "smoothness_eligible": bool(
                previous_available and trajectory.schedule_type == "smooth"
            ),
        }


class AdaptiveRolloutDataset(Dataset[dict[str, object]]):
    """Causal closed-loop training windows with future truth used only as targets."""

    def __init__(
        self,
        cache: AdaptiveCache,
        split: str,
        history: int,
        horizon: int,
        *,
        stride: int = 1,
    ) -> None:
        if (
            split not in {"train", "validation", "test"}
            or history < 1
            or horizon < 1
            or stride < 1
        ):
            raise ValueError("invalid V0.9 rollout split/history/horizon")
        self.trajectories = cache.select(split)
        if not self.trajectories:
            raise ValueError(f"V0.9 cache has no {split} trajectories")
        self.history = history
        self.horizon = horizon
        self.stride = stride
        self.locations = [
            (trajectory_index, target_index)
            for trajectory_index, trajectory in enumerate(self.trajectories)
            for target_index in range(
                history - 1, trajectory.dts.shape[0] - horizon + 1, stride
            )
        ]
        if not self.locations:
            raise ValueError("V0.9 trajectories are too short for the rollout curriculum")

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, index: int) -> dict[str, object]:
        trajectory_index, target_index = self.locations[index]
        trajectory = self.trajectories[trajectory_index]
        start = target_index - self.history + 1
        stop = target_index + self.horizon
        return {
            "history_z": trajectory.latents[start : target_index + 1].clone(),
            "history_dts": trajectory.dts[start:target_index].clone(),
            "future_dts": trajectory.dts[target_index:stop].clone(),
            "future_conditions": trajectory.conditions[target_index:stop].clone(),
            "target_latents": trajectory.latents[target_index + 1 : stop + 1].clone(),
            "context_parameters": trajectory.context_parameters,
            "schedule_type": trajectory.schedule_type,
            "trajectory_id": trajectory.trajectory_id,
            "target_index": target_index,
            "relative_transition_index": target_index - trajectory.transition_index,
        }
