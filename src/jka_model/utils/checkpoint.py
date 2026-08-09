"""Versioned checkpoint schema with compatibility guards."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from jka_model.config import ProjectConfig, stable_config_hash
from jka_model.constants import (
    ARCHITECTURE_REVISION,
    CHECKPOINT_SCHEMA_VERSION,
    PROJECT_VERSION,
)
from jka_model.contracts import ProblemSpec
from jka_model.training import TrainStage
from jka_model.utils.seed import RNGState


@dataclass(slots=True)
class Checkpoint:
    """Complete epoch-boundary resume envelope.

    Model and optimizer fields remain optional so fixed-generator V0.3 diagnostics and
    historical V0.1/V0.2 artifacts can use the same schema. Trainable V0.3 identification
    stores the Koopman generator and optimizer state in these fields.
    """

    train_stage: TrainStage
    epoch: int
    global_step: int
    architecture_revision: str = ARCHITECTURE_REVISION
    project_version: str = PROJECT_VERSION
    online_model_state: Mapping[str, Any] | None = None
    target_model_state: Mapping[str, Any] | None = None
    optimizer_state: Mapping[str, Any] | None = None
    scheduler_state: Mapping[str, Any] | None = None
    amp_scaler_state: Mapping[str, Any] | None = None
    rng_state: RNGState | None = None
    normalizer_state: Mapping[str, Any] | None = None
    problem_spec: ProblemSpec | None = None
    config: ProjectConfig | None = None
    config_hash: str | None = None
    data_fingerprint: str | None = None
    split_manifest: Any = None
    physics_constraint_spec: Any = None
    git_commit: str | None = None
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.epoch < 0 or self.global_step < 0:
            raise ValueError("checkpoint epoch and global_step must be non-negative")
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema {self.schema_version} is incompatible with runtime "
                f"schema {CHECKPOINT_SCHEMA_VERSION}"
            )
        if self.architecture_revision != ARCHITECTURE_REVISION:
            raise ValueError(
                f"checkpoint architecture revision {self.architecture_revision!r} is incompatible "
                f"with runtime revision {ARCHITECTURE_REVISION!r}"
            )
        if self.project_version != PROJECT_VERSION:
            raise ValueError(
                f"checkpoint project version {self.project_version!r} is incompatible with "
                f"runtime version {PROJECT_VERSION!r}"
            )
        if self.config is not None:
            resolved_hash = stable_config_hash(self.config)
            if self.config_hash is None:
                self.config_hash = resolved_hash
            elif self.config_hash != resolved_hash:
                raise ValueError("checkpoint config_hash does not match the resolved config")

    def to_payload(self) -> dict[str, Any]:
        """Return a plain torch-saveable schema payload."""
        return {
            "schema_version": self.schema_version,
            "architecture_revision": self.architecture_revision,
            "project_version": self.project_version,
            "train_stage": self.train_stage.value,
            "online_model_state": self.online_model_state,
            "target_model_state": self.target_model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "amp_scaler_state": self.amp_scaler_state,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "rng_state": None if self.rng_state is None else self.rng_state.to_checkpoint_dict(),
            "normalizer_state": self.normalizer_state,
            "problem_spec": None if self.problem_spec is None else self.problem_spec.to_dict(),
            "config": None if self.config is None else self.config.to_dict(),
            "config_hash": self.config_hash,
            "data_fingerprint": self.data_fingerprint,
            "split_manifest": self.split_manifest,
            "physics_constraint_spec": self.physics_constraint_spec,
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Checkpoint:
        required = {
            "schema_version",
            "architecture_revision",
            "project_version",
            "train_stage",
            "online_model_state",
            "target_model_state",
            "optimizer_state",
            "scheduler_state",
            "epoch",
            "global_step",
            "rng_state",
            "normalizer_state",
            "problem_spec",
            "config",
            "config_hash",
            "data_fingerprint",
            "split_manifest",
            "physics_constraint_spec",
            "git_commit",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"checkpoint is missing field(s): {', '.join(sorted(missing))}")
        if int(payload["schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema {payload['schema_version']} is incompatible with runtime "
                f"schema {CHECKPOINT_SCHEMA_VERSION}"
            )
        if str(payload["architecture_revision"]) != ARCHITECTURE_REVISION:
            raise ValueError(
                f"checkpoint architecture revision {payload['architecture_revision']!r} is "
                f"incompatible with runtime revision {ARCHITECTURE_REVISION!r}"
            )
        if str(payload["project_version"]) != PROJECT_VERSION:
            raise ValueError(
                f"checkpoint project version {payload['project_version']!r} is incompatible with "
                f"runtime version {PROJECT_VERSION!r}"
            )
        rng_payload = payload["rng_state"]
        problem_payload = payload["problem_spec"]
        config_payload = payload["config"]
        return cls(
            schema_version=int(payload["schema_version"]),
            architecture_revision=str(payload["architecture_revision"]),
            project_version=str(payload["project_version"]),
            train_stage=TrainStage(str(payload["train_stage"])),
            online_model_state=payload["online_model_state"],
            target_model_state=payload["target_model_state"],
            optimizer_state=payload["optimizer_state"],
            scheduler_state=payload["scheduler_state"],
            amp_scaler_state=payload.get("amp_scaler_state"),
            epoch=int(payload["epoch"]),
            global_step=int(payload["global_step"]),
            rng_state=(None if rng_payload is None else RNGState.from_checkpoint_dict(rng_payload)),
            normalizer_state=payload["normalizer_state"],
            problem_spec=(
                None if problem_payload is None else ProblemSpec.from_dict(problem_payload)
            ),
            config=(None if config_payload is None else ProjectConfig.from_dict(config_payload)),
            config_hash=(None if payload["config_hash"] is None else str(payload["config_hash"])),
            data_fingerprint=(
                None if payload["data_fingerprint"] is None else str(payload["data_fingerprint"])
            ),
            split_manifest=payload["split_manifest"],
            physics_constraint_spec=payload["physics_constraint_spec"],
            git_commit=None if payload["git_commit"] is None else str(payload["git_commit"]),
        )


def save_checkpoint(checkpoint: Checkpoint, path: str | Path) -> None:
    """Atomically save a checkpoint payload."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(checkpoint.to_payload(), temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Checkpoint:
    """Load a trusted local checkpoint and enforce schema/revision compatibility."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older supported torch.
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return Checkpoint.from_payload(payload)
