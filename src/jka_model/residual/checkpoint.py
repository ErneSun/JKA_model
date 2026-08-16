"""Standalone schema-7 V0.7 backbone-plus-closure checkpoint."""

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
    "closure_variant",
    "backbone_state",
    "closure_state",
    "optimizer_state",
    "scheduler_state",
    "amp_scaler_state",
    "rng_state",
    "normalizer_state",
    "problem_spec",
    "config",
    "config_hash",
    "data_fingerprint",
    "split_manifest",
    "backbone_checkpoint_sha256",
    "cache_fingerprint",
    "git_commit",
}


def validate_residual_checkpoint(payload: Mapping[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(payload)
    if missing:
        raise ValueError(f"V0.7 checkpoint missing field(s): {sorted(missing)!r}")
    if int(payload["schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("V0.7 checkpoint schema mismatch")
    if str(payload["project_version"]) != PROJECT_VERSION:
        raise ValueError("V0.7 checkpoint project version mismatch")
    if str(payload["architecture_revision"]) != ARCHITECTURE_REVISION:
        raise ValueError("V0.7 checkpoint architecture revision mismatch")
    if str(payload["train_stage"]) != TrainStage.RESIDUAL.value:
        raise ValueError("V0.7 checkpoint must use residual train stage")
    config = payload["config"]
    if not isinstance(config, Mapping):
        raise ValueError("V0.7 checkpoint lacks resolved config")
    resolved = ProjectConfig.from_dict(config)
    if payload["config_hash"] != stable_config_hash(resolved):
        raise ValueError("V0.7 checkpoint config hash mismatch")
    if payload["backbone_state"] is None or payload["closure_state"] is None:
        raise ValueError("V0.7 standalone checkpoint lacks backbone or closure state")


def save_residual_checkpoint(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_residual_checkpoint(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_residual_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("V0.7 checkpoint payload must be a mapping")
    validate_residual_checkpoint(payload)
    return dict(payload)
