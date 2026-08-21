"""Prepare the frozen-backbone V0.9 adaptive cache from controlled trajectories."""

from __future__ import annotations

from pathlib import Path

import torch

from jka_model.adaptive import build_adaptive_cache, save_adaptive_cache
from jka_model.config import ProjectConfig, load_config
from jka_model.context.checkpoint import load_context_checkpoint
from jka_model.data import (
    ChannelStandardizer,
    SplitManifest,
    data_fingerprint,
    load_cylinder_wake_dataset,
    make_split_manifest,
)
from jka_model.residual.cache import file_sha256
from jka_model.utils import load_checkpoint
from train.train_v0_6 import initialize_v0_6_model


def _stratified_schedule_manifest(config: ProjectConfig, records: object) -> SplitManifest:
    groups: dict[str, list[object]] = {"smooth": [], "abrupt": []}
    for record in records:  # type: ignore[union-attr]
        schedule_type = str(record.metadata.get("schedule_type"))
        if schedule_type not in groups:
            raise ValueError("V0.9 record lacks a registered schedule type")
        groups[schedule_type].append(record)
    if any(len(group) < 3 for group in groups.values()):
        raise ValueError("V0.9 requires at least three trajectories per schedule type")
    manifests = [make_split_manifest(group, config.data.split) for group in groups.values()]
    return SplitManifest(
        train=tuple(identifier for manifest in manifests for identifier in manifest.train),
        validation=tuple(
            identifier for manifest in manifests for identifier in manifest.validation
        ),
        test=tuple(identifier for manifest in manifests for identifier in manifest.test),
        seed=config.data.split.seed,
        ratios=(config.data.split.train, config.data.split.validation, config.data.split.test),
    )


def prepare_v0_9_cache(
    config: ProjectConfig | str | Path,
    *,
    backbone_checkpoint: str | Path,
    context_checkpoint: str | Path,
    physical_dataset: str | Path,
    destination: str | Path,
    device: str | torch.device | None = None,
) -> Path:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    if resolved.v0_9_condition is None or resolved.cylinder_wake_2d is None:
        raise ValueError("prepare_v0_9_cache requires the complete V0.9 physical contract")
    selected = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
    )
    saved = load_checkpoint(backbone_checkpoint, map_location="cpu")
    if saved.online_model_state is None or saved.target_model_state is None:
        raise ValueError("V0.9 backbone checkpoint lacks online/target state")
    if saved.normalizer_state is None or saved.config_hash is None:
        raise ValueError("V0.9 backbone checkpoint lacks normalizer/config provenance")
    context = load_context_checkpoint(context_checkpoint)
    backbone_sha = file_sha256(backbone_checkpoint)
    if context["backbone_checkpoint_sha256"] != backbone_sha:
        raise ValueError("V0.9 context does not belong to the selected backbone")
    model = initialize_v0_6_model(resolved, device=selected)
    model.load_online_state_dict(saved.online_model_state)
    model.target_encoder.load_state_dict(saved.target_model_state, strict=True)
    model.requires_grad_(False)
    model.eval()
    normalizer = ChannelStandardizer(eps=resolved.data.normalization.eps)
    normalizer.load_state_dict(saved.normalizer_state)
    dataset = load_cylinder_wake_dataset(physical_dataset, resolved.cylinder_wake_2d)
    manifest = _stratified_schedule_manifest(resolved, dataset.records)
    cache = build_adaptive_cache(
        model,
        dataset.records,
        normalizer,
        manifest,
        backbone_checkpoint_sha256=backbone_sha,
        backbone_config_hash=saved.config_hash,
        context_checkpoint_sha256=file_sha256(context_checkpoint),
        data_fingerprint=data_fingerprint(dataset.records, dataset.problem_spec),
    )
    output = Path(destination)
    save_adaptive_cache(cache, output)
    manifest.save(output.with_name("split_manifest.json"))
    return output
