"""Fingerprint-safe V0.9 latent cache with causal per-transition conditions."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from jka_model.data import ChannelStandardizer, SplitManifest
from jka_model.data.datasets import TrajectoryDataset
from jka_model.models import FieldJEPAKoopmanModel


def _bytes(value: Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


@dataclass(frozen=True, slots=True)
class AdaptiveTrajectory:
    trajectory_id: str
    split: str
    schedule_type: str
    transition_index: int
    latents: Tensor
    dts: Tensor
    context_parameters: Tensor
    conditions: Tensor
    nominal_residuals: Tensor

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("invalid V0.9 cache split")
        if self.schedule_type not in {"smooth", "abrupt"}:
            raise ValueError("invalid V0.9 schedule type")
        if self.latents.ndim != 2 or self.nominal_residuals.shape != self.latents[:-1].shape:
            raise ValueError("invalid V0.9 latent/residual alignment")
        if self.dts.shape != (self.latents.shape[0] - 1,):
            raise ValueError("invalid V0.9 dt alignment")
        if self.conditions.shape != (self.dts.shape[0], 2):
            raise ValueError("V0.9 conditions must align as [transition,Re/U]")
        if self.context_parameters.ndim != 1:
            raise ValueError("V0.9 context parameters must be a vector")
        if not 0 < self.transition_index < self.dts.shape[0]:
            raise ValueError("invalid V0.9 transition index")
        for value in (
            self.latents,
            self.dts,
            self.context_parameters,
            self.conditions,
            self.nominal_residuals,
        ):
            if not torch.isfinite(value).all():
                raise ValueError("V0.9 cache tensors must be finite")


@dataclass(frozen=True, slots=True)
class AdaptiveCache:
    trajectories: tuple[AdaptiveTrajectory, ...]
    backbone_checkpoint_sha256: str
    backbone_config_hash: str
    context_checkpoint_sha256: str
    data_fingerprint: str
    split_manifest: dict[str, Any]
    normalizer_state: dict[str, Any]
    nominal_generator: Tensor
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.trajectories or self.schema_version != 1:
            raise ValueError("invalid V0.9 adaptive cache")
        dimensions = {item.latents.shape[1] for item in self.trajectories}
        if len(dimensions) != 1:
            raise ValueError("V0.9 trajectories must share one latent dimension")
        if self.nominal_generator.shape != (self.latent_dim, self.latent_dim):
            raise ValueError("V0.9 cache nominal generator shape mismatch")

    @property
    def latent_dim(self) -> int:
        return self.trajectories[0].latents.shape[1]

    @property
    def context_parameter_dim(self) -> int:
        return self.trajectories[0].context_parameters.numel()

    def select(self, split: str) -> tuple[AdaptiveTrajectory, ...]:
        return tuple(item for item in self.trajectories if item.split == split)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "backbone_checkpoint_sha256": self.backbone_checkpoint_sha256,
                    "backbone_config_hash": self.backbone_config_hash,
                    "context_checkpoint_sha256": self.context_checkpoint_sha256,
                    "data_fingerprint": self.data_fingerprint,
                    "split_manifest": self.split_manifest,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        for item in self.trajectories:
            digest.update(item.trajectory_id.encode())
            digest.update(item.split.encode())
            digest.update(item.schedule_type.encode())
            digest.update(str(item.transition_index).encode())
            for value in (
                item.latents,
                item.dts,
                item.context_parameters,
                item.conditions,
                item.nominal_residuals,
            ):
                digest.update(_bytes(value))
        digest.update(_bytes(self.nominal_generator))
        return digest.hexdigest()

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backbone_checkpoint_sha256": self.backbone_checkpoint_sha256,
            "backbone_config_hash": self.backbone_config_hash,
            "context_checkpoint_sha256": self.context_checkpoint_sha256,
            "data_fingerprint": self.data_fingerprint,
            "split_manifest": self.split_manifest,
            "normalizer_state": self.normalizer_state,
            "nominal_generator": self.nominal_generator,
            "cache_fingerprint": self.fingerprint,
            "trajectories": [
                {
                    "trajectory_id": item.trajectory_id,
                    "split": item.split,
                    "schedule_type": item.schedule_type,
                    "transition_index": item.transition_index,
                    "latents": item.latents,
                    "dts": item.dts,
                    "context_parameters": item.context_parameters,
                    "conditions": item.conditions,
                    "nominal_residuals": item.nominal_residuals,
                }
                for item in self.trajectories
            ],
        }


@torch.no_grad()
def build_adaptive_cache(
    model: FieldJEPAKoopmanModel,
    records: TrajectoryDataset,
    normalizer: ChannelStandardizer,
    manifest: SplitManifest,
    *,
    backbone_checkpoint_sha256: str,
    backbone_config_hash: str,
    context_checkpoint_sha256: str,
    data_fingerprint: str,
) -> AdaptiveCache:
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("V0.9 cache construction requires a completely frozen backbone")
    device = model.koopman_core.A.device
    split_lookup = {
        trajectory_id: split
        for split in ("train", "validation", "test")
        for trajectory_id in getattr(manifest, split)
    }
    cached: list[AdaptiveTrajectory] = []
    for record in records:
        if record.trajectory_id not in split_lookup:
            raise ValueError("V0.9 split manifest is incomplete")
        conditions = torch.as_tensor(record.metadata.get("condition_series"), dtype=torch.float32)
        schedule_type = str(record.metadata.get("schedule_type"))
        transition_index = int(record.metadata.get("transition_index", -1))
        if record.mu_static is None:
            raise ValueError("V0.9 context checkpoint requires nominal static parameters")
        fields = normalizer.transform(record.states_raw.to(device=device, dtype=torch.float32))
        latents = model.encode(fields)
        nominal = model.koopman_core.step(latents[:-1], record.dts.to(device=device))
        residuals = (latents[1:] - nominal).detach()
        cached.append(
            AdaptiveTrajectory(
                trajectory_id=record.trajectory_id,
                split=split_lookup[record.trajectory_id],
                schedule_type=schedule_type,
                transition_index=transition_index,
                latents=latents.detach().cpu(),
                dts=record.dts.detach().float().cpu(),
                context_parameters=record.mu_static.detach().float().cpu(),
                conditions=conditions,
                nominal_residuals=residuals.float().cpu(),
            )
        )
    return AdaptiveCache(
        trajectories=tuple(cached),
        backbone_checkpoint_sha256=backbone_checkpoint_sha256,
        backbone_config_hash=backbone_config_hash,
        context_checkpoint_sha256=context_checkpoint_sha256,
        data_fingerprint=data_fingerprint,
        split_manifest=manifest.to_dict(),
        normalizer_state=dict(normalizer.state_dict()),
        nominal_generator=model.koopman_core.A.detach().float().cpu(),
    )


def save_adaptive_cache(cache: AdaptiveCache, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(cache.to_payload(), temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_adaptive_cache(path: str | Path) -> AdaptiveCache:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("invalid V0.9 adaptive cache payload")
    cache = AdaptiveCache(
        trajectories=tuple(AdaptiveTrajectory(**item) for item in payload["trajectories"]),
        backbone_checkpoint_sha256=str(payload["backbone_checkpoint_sha256"]),
        backbone_config_hash=str(payload["backbone_config_hash"]),
        context_checkpoint_sha256=str(payload["context_checkpoint_sha256"]),
        data_fingerprint=str(payload["data_fingerprint"]),
        split_manifest=dict(payload["split_manifest"]),
        normalizer_state=dict(payload["normalizer_state"]),
        nominal_generator=torch.as_tensor(payload["nominal_generator"]).float(),
        schema_version=int(payload["schema_version"]),
    )
    if payload.get("cache_fingerprint") != cache.fingerprint:
        raise ValueError("V0.9 adaptive cache fingerprint mismatch")
    return cache


def adaptive_training_scales(cache: AdaptiveCache) -> tuple[Tensor, Tensor, Tensor]:
    train = cache.select("train")
    if not train:
        raise ValueError("V0.9 cache has no training trajectories")
    residuals = torch.cat([item.nominal_residuals for item in train])
    floor = max(float(residuals.square().mean().sqrt()) * 1e-3, torch.finfo(torch.float32).eps)
    residual_scale = residuals.square().mean(dim=0).sqrt().clamp_min(floor)
    conditions = torch.cat([item.conditions for item in train])
    condition_mean = conditions.mean(dim=0)
    condition_std = conditions.std(dim=0).clamp_min(1e-6)
    return residual_scale, condition_mean, condition_std
