"""Leakage-free fixed-history windows over cached V0.7 residual targets."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from jka_model.residual.cache import ResidualCache, ResidualTrajectory


class ResidualWindowDataset(Dataset[dict[str, object]]):
    """Windows never cross trajectory or split boundaries."""

    def __init__(
        self,
        cache: ResidualCache,
        split: str,
        history: int,
        *,
        shuffle_history: bool = False,
        shuffle_seed: int = 0,
    ) -> None:
        if split not in {"train", "validation", "test"} or history < 1:
            raise ValueError("invalid residual split or history")
        self.trajectories = cache.select(split)
        if not self.trajectories:
            raise ValueError(f"residual cache has no {split} trajectories")
        self.history = history
        self.shuffle_history = shuffle_history
        self.locations: list[tuple[int, int]] = []
        for trajectory_index, trajectory in enumerate(self.trajectories):
            for target_index in range(history - 1, trajectory.residuals.shape[0]):
                self.locations.append((trajectory_index, target_index))
        if not self.locations:
            raise ValueError("residual trajectories are too short for requested history")
        # A fixed permutation makes shuffled-history a reproducible matched control.
        step = max(1, (2 * shuffle_seed + 1) % len(self.locations))
        while len(self.locations) > 1 and self._gcd(step, len(self.locations)) != 1:
            step += 1
        self.history_source = [
            self.locations[(index * step + 1) % len(self.locations)]
            for index in range(len(self.locations))
        ]

    @staticmethod
    def _gcd(left: int, right: int) -> int:
        while right:
            left, right = right, left % right
        return left

    def __len__(self) -> int:
        return len(self.locations)

    def _older_history(self, trajectory: ResidualTrajectory, target_index: int):
        start = target_index - self.history + 1
        return trajectory.latents[start:target_index], trajectory.dts[start:target_index]

    def __getitem__(self, index: int) -> dict[str, object]:
        trajectory_index, target_index = self.locations[index]
        trajectory = self.trajectories[trajectory_index]
        older_z, older_dt = self._older_history(trajectory, target_index)
        if self.shuffle_history:
            source_trajectory_index, source_target_index = self.history_source[index]
            source = self.trajectories[source_trajectory_index]
            older_z, older_dt = self._older_history(source, source_target_index)
        history_z = torch.cat((older_z, trajectory.latents[target_index : target_index + 1]))
        return {
            "history_z": history_z,
            "history_dts": older_dt,
            "next_dt": trajectory.dts[target_index : target_index + 1],
            "parameters": trajectory.parameters,
            "target": trajectory.residuals[target_index],
            "trajectory_id": trajectory.trajectory_id,
            "target_index": target_index,
        }
