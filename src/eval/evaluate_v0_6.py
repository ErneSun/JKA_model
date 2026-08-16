"""Canonical V0.6 evaluation: online model for rollout, target for diagnostics only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from eval.evaluate_v0_5 import evaluate_v0_5
from jka_model.config import ProjectConfig, load_config
from jka_model.data import ChannelStandardizer, SplitManifest, select_split
from jka_model.evaluation import (
    latent_statistics,
    latent_tracking_distance,
    near_identity_diagnostic,
)
from jka_model.models import normalized_parameter_distance
from jka_model.problems import create_problem_adapter
from jka_model.utils import load_checkpoint
from train.train_v0_6 import initialize_v0_6_model


def evaluate_v0_6(
    config: ProjectConfig | str | Path,
    *,
    checkpoint: str | Path,
    device: str | torch.device | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate held-out online rollouts and report target/collapse diagnostics."""
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    if resolved.jepa_loss is None or resolved.ema is None or resolved.v0_6_evaluation is None:
        raise ValueError("evaluate_v0_6 requires complete V0.6 config sections")
    selected = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
    )
    saved = load_checkpoint(checkpoint, map_location="cpu")
    if saved.config_hash != resolved.stable_hash:
        raise ValueError("V0.6 evaluation checkpoint/config mismatch")
    if saved.online_model_state is None or saved.target_model_state is None:
        raise ValueError("V0.6 checkpoint lacks online or target model state")
    # Reuse the already-validated V0.5 held-out evaluator. Its model contains only the
    # online encoder, Koopman core and decoder, which enforces the V0.6 inference contract.
    result = evaluate_v0_5(resolved, checkpoint=checkpoint, device=selected, run_dir=None)
    model = initialize_v0_6_model(resolved, device=selected)
    model.load_online_state_dict(saved.online_model_state)
    model.target_encoder.load_state_dict(saved.target_model_state, strict=True)
    model.target_encoder.requires_grad_(False)
    model.eval()
    adapter = create_problem_adapter(resolved)
    records = adapter.build_dataset(seed=resolved.training.seed)
    if not isinstance(saved.split_manifest, dict):
        raise ValueError("V0.6 checkpoint lacks split manifest")
    test_records = select_split(records, SplitManifest.from_dict(saved.split_manifest), "test")
    if saved.normalizer_state is None:
        raise ValueError("V0.6 checkpoint lacks normalizer state")
    normalizer = ChannelStandardizer(eps=resolved.data.normalization.eps)
    normalizer.load_state_dict(saved.normalizer_state)
    online_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for record in test_records:
            fields = normalizer.transform(record.states_raw.to(selected, torch.float32))
            online_parts.append(model.encode(fields).cpu())
            target_parts.append(model.encode_target(fields).cpu())
    online_latent = torch.cat(online_parts)
    target_latent = torch.cat(target_parts)
    online_statistics = latent_statistics(online_latent)
    target_statistics = latent_statistics(target_latent)
    tracking = {
        "online": online_statistics,
        "target": target_statistics,
        "latent_distance": latent_tracking_distance(online_latent, target_latent),
        "parameter_distance": normalized_parameter_distance(model),
    }
    all_dts = torch.cat([record.dts for record in test_records]).to(selected)
    result.update(
        {
            "phase": "v0.6",
            "inference_modules": ["online_encoder", "koopman_core", "training_decoder"],
            "target_used_for_rollout": False,
            "jepa_enabled": resolved.jepa_loss.enabled,
            "tracking": tracking,
            "near_identity": near_identity_diagnostic(model.koopman_core.A.detach(), all_dts),
            "ema_state": saved.ema_state,
            "optimizer_update_step": saved.optimizer_update_step,
            "collapse_threshold": resolved.v0_6_evaluation.min_latent_std,
            "collapse_gate": (
                online_statistics["min_dimension_std"]
                >= resolved.v0_6_evaluation.min_latent_std
                and target_statistics["min_dimension_std"]
                >= resolved.v0_6_evaluation.min_latent_std
            ),
            "scientific_acceptance": "PENDING_GPU",
        }
    )
    if run_dir is not None:
        root = Path(run_dir)
        destination = root / "evaluation"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (destination / "final_metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (destination / "jepa_diagnostics.json").write_text(
            json.dumps(
                {
                    "tracking": tracking,
                    "near_identity": result["near_identity"],
                    "collapse_gate": result["collapse_gate"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        reports = root / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "test_record.md").write_text(
            "# V0.6 held-out test record\n\n"
            "Rollout used the online encoder, continuous Koopman core and decoder only.\n\n"
            f"- long rollout RMSE: {result['rollout']['long']['rmse']:.8g}\n"
            f"- long mass drift: {result['rollout']['long']['mass_drift']:.8g}\n"
            f"- online latent minimum std: {online_statistics['min_dimension_std']:.8g}\n"
            f"- target latent minimum std: {target_statistics['min_dimension_std']:.8g}\n"
            f"- collapse gate: `{result['collapse_gate']}`\n"
            "- scientific acceptance: `PENDING_GPU`\n",
            encoding="utf-8",
        )
    return result
