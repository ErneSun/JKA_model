"""Leakage-free V0.8 windows over the frozen V0.7 residual cache."""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor
from torch.utils.data import Dataset

from jka_model.residual import ResidualCache


def residual_training_scales(cache: ResidualCache) -> tuple[Tensor, Tensor, str]:
    """Per-coordinate residual RMS and scalar adequacy RMS from train trajectories only."""
    residuals = torch.cat([item.residuals.float() for item in cache.select("train")])
    if residuals.numel() == 0:
        raise ValueError("context cache has no training residuals")
    floor = max(float(residuals.square().mean().sqrt()) * 1e-3, torch.finfo(torch.float32).eps)
    residual_scale = residuals.square().mean(dim=0).sqrt().clamp_min(floor)
    adequacy = residuals.square().mean(dim=-1).sqrt()
    adequacy_scale = adequacy.square().mean().sqrt().clamp_min(floor)
    digest = hashlib.sha256()
    digest.update(residual_scale.double().contiguous().numpy().tobytes())
    digest.update(adequacy_scale.double().reshape(1).numpy().tobytes())
    return residual_scale, adequacy_scale, digest.hexdigest()


class ContextWindowDataset(Dataset[dict[str, object]]):
    """A target sees current/past latents and known dt/parameters, never future latent state."""

    def __init__(
        self,
        cache: ResidualCache,
        split: str,
        history: int,
        *,
        shuffle_older_history: bool = False,
        shuffle_seed: int = 0,
    ) -> None:
        if split not in {"train", "validation", "test"} or history < 1:
            raise ValueError("invalid V0.8 split/history")
        self.trajectories = cache.select(split)
        if not self.trajectories:
            raise ValueError(f"context cache has no {split} trajectories")
        self.history = history
        self.shuffle_older_history = shuffle_older_history
        self.locations = [
            (trajectory_index, target_index)
            for trajectory_index, trajectory in enumerate(self.trajectories)
            for target_index in range(history - 1, trajectory.residuals.shape[0])
        ]
        if not self.locations:
            raise ValueError("V0.8 trajectories are too short for requested history")
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
        if self.shuffle_older_history and self.history > 1:
            source_trajectory_index, source_target_index = self.locations[self.permutation[index]]
            source = self.trajectories[source_trajectory_index]
            source_start = source_target_index - self.history + 1
            history_z[:-1] = source.latents[source_start:source_target_index]
            history_dts = source.dts[source_start:source_target_index].clone()
        target = trajectory.residuals[target_index]
        return {
            "history_z": history_z,
            "history_dts": history_dts,
            "next_dt": trajectory.dts[target_index : target_index + 1],
            "parameters": trajectory.parameters,
            "target_residual": target,
            "target_adequacy": target.square().mean().sqrt().reshape(1),
            "trajectory_id": trajectory.trajectory_id,
            "target_index": target_index,
        }
