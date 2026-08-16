"""Build and audit a fingerprinted V0.7 residual cache from a frozen V0.6 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from jka_model.config import ProjectConfig, load_config
from jka_model.data import ChannelStandardizer, SplitManifest, data_fingerprint
from jka_model.problems import create_problem_adapter
from jka_model.residual import build_residual_cache, residual_statistics, save_residual_cache
from jka_model.residual.cache import file_sha256
from jka_model.utils import load_checkpoint
from train.train_v0_7 import _require_v0_7, load_frozen_v0_6_backbone


def prepare_v0_7_cache(
    config: ProjectConfig | str | Path,
    *,
    backbone_checkpoint: str | Path,
    destination: str | Path,
    diagnostics_path: str | Path,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    _require_v0_7(resolved)
    assert resolved.residual_closure
    selected = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
    )
    print(f"[V0.7][residual-cache] START device={selected}", flush=True)
    shell, saved = load_frozen_v0_6_backbone(backbone_checkpoint, resolved, selected)
    source = load_checkpoint(backbone_checkpoint, map_location="cpu")
    if source.normalizer_state is None or not isinstance(source.split_manifest, dict):
        raise ValueError("V0.6 checkpoint lacks normalizer or split manifest")
    adapter = create_problem_adapter(resolved)
    records = adapter.build_dataset(seed=resolved.training.seed)
    spec = adapter.build_problem_spec()
    manifest = SplitManifest.from_dict(source.split_manifest)
    fingerprint = data_fingerprint(records, spec)
    if source.data_fingerprint != fingerprint:
        raise ValueError("regenerated V0.6 data fingerprint mismatch")
    normalizer = ChannelStandardizer(eps=resolved.data.normalization.eps)
    normalizer.load_state_dict(source.normalizer_state)
    dtype = torch.float32 if resolved.residual_closure.cache_dtype == "float32" else torch.float64
    cache = build_residual_cache(
        shell,
        records,
        normalizer,
        manifest,
        backbone_checkpoint_sha256=file_sha256(backbone_checkpoint),
        backbone_config_hash=str(saved.config_hash),
        data_fingerprint=fingerprint,
        dtype=dtype,
    )
    save_residual_cache(cache, destination)
    diagnostics = {
        "cache_fingerprint": cache.fingerprint,
        "backbone_checkpoint_sha256": cache.backbone_checkpoint_sha256,
        "target_semantics": cache.target_semantics,
        "target_encoder_used": False,
        "splits": {
            split: residual_statistics(cache, split, resolved.residual_closure.max_acf_lag)
            for split in ("train", "validation", "test")
        },
    }
    diagnostics_destination = Path(diagnostics_path)
    diagnostics_destination.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_destination.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[V0.7][residual-cache] PASS fingerprint={cache.fingerprint} "
        f"train_rms={diagnostics['splits']['train']['rms']:.6g}",
        flush=True,
    )
    return diagnostics
