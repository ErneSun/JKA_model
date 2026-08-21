"""Exact-resume schema-9 V0.9 adaptive-operator checkpoint."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from jka_model.config import ProjectConfig, stable_config_hash
from jka_model.constants import ARCHITECTURE_REVISION, CHECKPOINT_SCHEMA_VERSION, PROJECT_VERSION
from jka_model.training import TrainStage

REQUIRED_FIELDS = {
    "schema_version",
    "architecture_revision",
    "project_version",
    "train_stage",
    "epoch",
    "global_step",
    "optimizer_update_step",
    "condition_mode",
    "rank",
    "flow_data_seed",
    "backbone_seed",
    "context_init_seed",
    "operator_init_seed",
    "adaptive_state",
    "best_adaptive_state",
    "optimizer_state",
    "scheduler_state",
    "amp_scaler_state",
    "rng_state",
    "config",
    "config_hash",
    "backbone_checkpoint_sha256",
    "context_checkpoint_sha256",
    "adaptive_cache_fingerprint",
    "residual_training_scale",
    "condition_mean",
    "condition_std",
    "best_validation_score",
    "epochs_without_improvement",
    "git_commit",
    "runtime",
}


def validate_adaptive_checkpoint(payload: Mapping[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(payload)
    if missing:
        raise ValueError(f"V0.9 checkpoint missing field(s): {sorted(missing)!r}")
    if (int(payload["schema_version"]), str(payload["project_version"])) != (
        CHECKPOINT_SCHEMA_VERSION,
        PROJECT_VERSION,
    ):
        raise ValueError("V0.9 checkpoint schema/project mismatch")
    if str(payload["architecture_revision"]) != ARCHITECTURE_REVISION:
        raise ValueError("V0.9 checkpoint architecture revision mismatch")
    if str(payload["train_stage"]) != TrainStage.ADAPTIVE.value:
        raise ValueError("V0.9 checkpoint must use adaptive train stage")
    config_payload = payload["config"]
    if not isinstance(config_payload, Mapping):
        raise ValueError("V0.9 checkpoint lacks a resolved config")
    # Hash the serialized payload before default-filling it.  This keeps old,
    # valid checkpoints loadable when a later schema adds optional fields while
    # still detecting any mutation of the checkpoint's original config.
    if payload["config_hash"] != stable_config_hash(config_payload):
        raise ValueError("V0.9 checkpoint config hash mismatch")
    config = ProjectConfig.from_dict(config_payload)
    if config.v0_9_adaptive is None or config.v0_9_training is None:
        raise ValueError("V0.9 checkpoint config lacks adaptive sections")
    if int(payload["rank"]) != config.v0_9_adaptive.rank:
        raise ValueError("V0.9 checkpoint rank mismatch")
    if str(payload["condition_mode"]) != config.v0_9_adaptive.condition_mode:
        raise ValueError("V0.9 checkpoint condition-mode mismatch")
    if int(payload["operator_init_seed"]) != config.v0_9_training.operator_initialization_seed:
        raise ValueError("V0.9 checkpoint operator seed mismatch")
    if payload["adaptive_state"] is None or payload["best_adaptive_state"] is None:
        raise ValueError("V0.9 checkpoint lacks adaptive model state")
    if float(payload["best_validation_score"]) < 0:
        raise ValueError("V0.9 checkpoint has invalid validation score")


def save_adaptive_checkpoint(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_adaptive_checkpoint(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_adaptive_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("V0.9 checkpoint payload must be a mapping")
    validate_adaptive_checkpoint(payload)
    loaded = dict(payload)
    resolved = ProjectConfig.from_dict(payload["config"])
    loaded["config"] = resolved.to_dict()
    loaded["config_hash"] = stable_config_hash(resolved)
    return loaded
