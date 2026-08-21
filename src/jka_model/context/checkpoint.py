"""Standalone schema-8 V0.8 context checkpoint with exact-resume state."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from jka_model.config import ProjectConfig, stable_config_hash
from jka_model.constants import (
    ARCHITECTURE_REVISION,
    CHECKPOINT_SCHEMA_VERSION,
    PROJECT_VERSION,
    V0_8_CHECKPOINT_SCHEMA_VERSION,
    V0_8_PROJECT_VERSION,
)
from jka_model.training import TrainStage

REQUIRED_FIELDS = {
    "schema_version",
    "architecture_revision",
    "project_version",
    "train_stage",
    "epoch",
    "global_step",
    "optimizer_update_step",
    "context_family",
    "residual_route",
    "history_length_steps",
    "context_dim",
    "flow_data_seed",
    "backbone_seed",
    "context_init_seed",
    "context_state",
    "optimizer_state",
    "scheduler_state",
    "amp_scaler_state",
    "rng_state",
    "config",
    "config_hash",
    "backbone_checkpoint_sha256",
    "residual_cache_fingerprint",
    "residual_scale_fingerprint",
    "residual_training_scale",
    "adequacy_training_scale",
    "split_fingerprint",
    "normalizer_fingerprint",
    "git_commit",
    "best_context_state",
    "best_validation_loss",
    "epochs_without_improvement",
    "runtime",
}


def validate_context_checkpoint(payload: Mapping[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(payload)
    if missing:
        raise ValueError(f"V0.8 checkpoint missing field(s): {sorted(missing)!r}")
    version_pair = (int(payload["schema_version"]), str(payload["project_version"]))
    if version_pair not in {
        (V0_8_CHECKPOINT_SCHEMA_VERSION, V0_8_PROJECT_VERSION),
        (CHECKPOINT_SCHEMA_VERSION, PROJECT_VERSION),
    }:
        raise ValueError("V0.8 checkpoint schema/project version mismatch")
    if str(payload["architecture_revision"]) != ARCHITECTURE_REVISION:
        raise ValueError("V0.8 checkpoint architecture revision mismatch")
    if str(payload["train_stage"]) != TrainStage.CONTEXT.value:
        raise ValueError("V0.8 checkpoint must use context train stage")
    config_payload = payload["config"]
    if not isinstance(config_payload, Mapping):
        raise ValueError("V0.8 checkpoint lacks resolved config")
    config = ProjectConfig.from_dict(config_payload)
    if payload["config_hash"] != stable_config_hash(config):
        raise ValueError("V0.8 checkpoint config hash mismatch")
    if config.v0_8_context is None or config.v0_8_training is None:
        raise ValueError("V0.8 checkpoint config lacks context sections")
    if int(payload["context_dim"]) != config.v0_8_context.context_dim:
        raise ValueError("V0.8 checkpoint context dimension mismatch")
    if int(payload["context_init_seed"]) != config.v0_8_training.context_initialization_seed:
        raise ValueError("V0.8 checkpoint context seed mismatch")
    if str(payload["residual_route"]) not in {"R2", "R3"}:
        raise ValueError("V0.8 trained checkpoint requires R2 or R3")
    if str(payload["context_family"]) not in {
        "instantaneous",
        "instantaneous_matched",
        "attention",
        "history_mlp",
    }:
        raise ValueError("V0.8 checkpoint has unsupported context family")
    if payload["context_state"] is None or payload["optimizer_state"] is None:
        raise ValueError("V0.8 checkpoint lacks model or optimizer state")
    if payload["best_context_state"] is None:
        raise ValueError("V0.8 checkpoint lacks validation-selected state")
    if float(payload["best_validation_loss"]) < 0:
        raise ValueError("V0.8 checkpoint has invalid best validation loss")
    if int(payload["epochs_without_improvement"]) < 0:
        raise ValueError("V0.8 checkpoint has invalid early-stop state")


def save_context_checkpoint(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_context_checkpoint(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_context_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("V0.8 checkpoint payload must be a mapping")
    validate_context_checkpoint(payload)
    return dict(payload)
