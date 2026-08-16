"""Frozen-online residual targets and fingerprinted latent caches for V0.7."""

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


def _tensor_bytes(value: Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


@dataclass(frozen=True, slots=True)
class ResidualTrajectory:
    trajectory_id: str
    split: str
    latents: Tensor
    dts: Tensor
    parameters: Tensor
    residuals: Tensor

    def __post_init__(self) -> None:
        if self.latents.ndim != 2 or self.residuals.ndim != 2:
            raise ValueError("latents and residuals must have shape [time, latent_dim]")
        if self.dts.ndim != 1 or self.parameters.ndim != 1:
            raise ValueError("dts and parameters must be vectors")
        if self.latents.shape[0] != self.dts.shape[0] + 1:
            raise ValueError("a residual trajectory requires one more latent state than dt")
        if self.residuals.shape != self.latents[:-1].shape:
            raise ValueError("residual shape must match one-step latent transitions")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("invalid residual trajectory split")
        for value in (self.latents, self.dts, self.parameters, self.residuals):
            if not torch.isfinite(value).all():
                raise ValueError("residual cache tensors must be finite")


@dataclass(frozen=True, slots=True)
class ResidualCache:
    trajectories: tuple[ResidualTrajectory, ...]
    backbone_checkpoint_sha256: str
    backbone_config_hash: str
    data_fingerprint: str
    split_manifest: dict[str, Any]
    normalizer_state: dict[str, Any]
    target_semantics: str = "stopgrad(E_online(U_next)-exp(A*dt)E_online(U_current))"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise ValueError("residual cache cannot be empty")
        if self.schema_version != 1:
            raise ValueError("unsupported residual cache schema")
        dimensions = {item.latents.shape[1] for item in self.trajectories}
        if len(dimensions) != 1:
            raise ValueError("all residual trajectories must share latent_dim")

    @property
    def latent_dim(self) -> int:
        return self.trajectories[0].latents.shape[1]

    @property
    def parameter_dim(self) -> int:
        return self.trajectories[0].parameters.numel()

    @property
    def split_fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.split_manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def normalizer_fingerprint(self) -> str:
        digest = hashlib.sha256()
        metadata = {
            key: value
            for key, value in self.normalizer_state.items()
            if not isinstance(value, Tensor)
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        for key in sorted(self.normalizer_state):
            value = self.normalizer_state[key]
            if isinstance(value, Tensor):
                digest.update(key.encode())
                digest.update(_tensor_bytes(value))
        return digest.hexdigest()

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        metadata = {
            "schema_version": self.schema_version,
            "backbone_checkpoint_sha256": self.backbone_checkpoint_sha256,
            "backbone_config_hash": self.backbone_config_hash,
            "data_fingerprint": self.data_fingerprint,
            "split_manifest": self.split_manifest,
            "target_semantics": self.target_semantics,
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        normalizer_metadata = {
            key: value
            for key, value in self.normalizer_state.items()
            if not isinstance(value, Tensor)
        }
        digest.update(
            json.dumps(normalizer_metadata, sort_keys=True, separators=(",", ":")).encode()
        )
        for key in sorted(self.normalizer_state):
            value = self.normalizer_state[key]
            if isinstance(value, Tensor):
                digest.update(key.encode())
                digest.update(_tensor_bytes(value))
        for item in self.trajectories:
            digest.update(item.trajectory_id.encode())
            digest.update(item.split.encode())
            for value in (item.latents, item.dts, item.parameters, item.residuals):
                digest.update(_tensor_bytes(value))
        return digest.hexdigest()

    def select(self, split: str) -> tuple[ResidualTrajectory, ...]:
        return tuple(item for item in self.trajectories if item.split == split)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backbone_checkpoint_sha256": self.backbone_checkpoint_sha256,
            "backbone_config_hash": self.backbone_config_hash,
            "data_fingerprint": self.data_fingerprint,
            "split_manifest": self.split_manifest,
            "normalizer_state": self.normalizer_state,
            "target_semantics": self.target_semantics,
            "split_fingerprint": self.split_fingerprint,
            "normalizer_fingerprint": self.normalizer_fingerprint,
            "cache_fingerprint": self.fingerprint,
            "trajectories": [
                {
                    "trajectory_id": item.trajectory_id,
                    "split": item.split,
                    "latents": item.latents,
                    "dts": item.dts,
                    "parameters": item.parameters,
                    "residuals": item.residuals,
                }
                for item in self.trajectories
            ],
        }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def build_residual_cache(
    model: FieldJEPAKoopmanModel,
    records: TrajectoryDataset,
    normalizer: ChannelStandardizer,
    manifest: SplitManifest,
    *,
    backbone_checkpoint_sha256: str,
    backbone_config_hash: str,
    data_fingerprint: str,
    dtype: torch.dtype = torch.float32,
) -> ResidualCache:
    """Compute targets using only the frozen V0.6 online encoder and Koopman core."""
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("residual cache supports float32 or float64")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("the complete V0.6 backbone must be frozen before cache construction")
    model.eval()
    device = model.koopman_core.A.device
    split_lookup = {
        trajectory_id: split
        for split in ("train", "validation", "test")
        for trajectory_id in getattr(manifest, split)
    }
    cached: list[ResidualTrajectory] = []
    for record in records:
        if record.trajectory_id not in split_lookup:
            raise ValueError(f"trajectory absent from split manifest: {record.trajectory_id}")
        fields = normalizer.transform(record.states_raw.to(device=device, dtype=torch.float32))
        # Deliberately use online encode; the EMA target is forbidden by the V0.7 contract.
        latents = model.encode(fields)
        predicted = model.koopman_core.step(latents[:-1], record.dts.to(device=device))
        residuals = (latents[1:] - predicted).detach()
        cached.append(
            ResidualTrajectory(
                trajectory_id=record.trajectory_id,
                split=split_lookup[record.trajectory_id],
                latents=latents.detach().to(device="cpu", dtype=dtype),
                dts=record.dts.detach().to(device="cpu", dtype=dtype),
                parameters=record.mu_static.detach().to(device="cpu", dtype=dtype),
                residuals=residuals.to(device="cpu", dtype=dtype),
            )
        )
    return ResidualCache(
        trajectories=tuple(cached),
        backbone_checkpoint_sha256=backbone_checkpoint_sha256,
        backbone_config_hash=backbone_config_hash,
        data_fingerprint=data_fingerprint,
        split_manifest=manifest.to_dict(),
        normalizer_state=dict(normalizer.state_dict()),
    )


def save_residual_cache(cache: ResidualCache, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(cache.to_payload(), temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_residual_cache(path: str | Path) -> ResidualCache:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("invalid residual cache payload")
    cache = ResidualCache(
        trajectories=tuple(ResidualTrajectory(**item) for item in payload["trajectories"]),
        backbone_checkpoint_sha256=str(payload["backbone_checkpoint_sha256"]),
        backbone_config_hash=str(payload["backbone_config_hash"]),
        data_fingerprint=str(payload["data_fingerprint"]),
        split_manifest=dict(payload["split_manifest"]),
        normalizer_state=dict(payload["normalizer_state"]),
        target_semantics=str(payload["target_semantics"]),
        schema_version=int(payload["schema_version"]),
    )
    if payload.get("cache_fingerprint") != cache.fingerprint:
        raise ValueError("residual cache fingerprint mismatch")
    return cache
