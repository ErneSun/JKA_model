"""Standalone schema-7 V0.7 backbone-plus-closure checkpoint."""

from __future__ import annotations

import hashlib
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
    V0_7_CHECKPOINT_SCHEMA_VERSION,
    V0_7_PROJECT_VERSION,
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
    "closure_variant",
    "backbone_data_seed",
    "closure_init_seed",
    "history_length_steps",
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
    "residual_training_scale",
    "residual_scale_fingerprint",
    "git_commit",
}


def validate_residual_checkpoint(payload: Mapping[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(payload)
    if missing:
        raise ValueError(f"V0.7 checkpoint missing field(s): {sorted(missing)!r}")
    version_pair = (int(payload["schema_version"]), str(payload["project_version"]))
    if version_pair not in {
        (V0_7_CHECKPOINT_SCHEMA_VERSION, V0_7_PROJECT_VERSION),
        (V0_8_CHECKPOINT_SCHEMA_VERSION, V0_8_PROJECT_VERSION),
        (CHECKPOINT_SCHEMA_VERSION, PROJECT_VERSION),
    }:
        raise ValueError("V0.7 checkpoint schema/project version mismatch")
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
    if resolved.project_version != str(payload["project_version"]):
        raise ValueError("V0.7 checkpoint resolved project version mismatch")
    if resolved.residual_training is None or resolved.residual_closure is None:
        raise ValueError("V0.7 checkpoint config lacks residual sections")
    if int(payload["backbone_data_seed"]) != resolved.training.seed:
        raise ValueError("V0.7 checkpoint backbone/data seed mismatch")
    if int(payload["closure_init_seed"]) != resolved.residual_training.initialization_seed:
        raise ValueError("V0.7 checkpoint closure initialization seed mismatch")
    if int(payload["history_length_steps"]) != resolved.residual_closure.history:
        raise ValueError("V0.7 checkpoint history length mismatch")
    scale = torch.as_tensor(payload["residual_training_scale"], dtype=torch.float64)
    if scale.ndim != 1 or scale.numel() != resolved.koopman.state_dim or torch.any(scale <= 0):
        raise ValueError("V0.7 checkpoint residual training scale is invalid")
    fingerprint = hashlib.sha256(scale.contiguous().numpy().tobytes()).hexdigest()
    if payload["residual_scale_fingerprint"] != fingerprint:
        raise ValueError("V0.7 checkpoint residual scale fingerprint mismatch")
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
