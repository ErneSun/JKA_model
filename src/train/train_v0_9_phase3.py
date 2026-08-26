"""Phase-3 joint representation training with raw-field online re-encoding."""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from jka_model.adaptive import (
    AdaptiveKoopmanModel,
    FactorizedAdaptiveOperator,
    adaptive_training_scales,
    load_adaptive_cache,
    phase2_condition_scales,
)
from jka_model.config import ProjectConfig, load_config, save_config
from jka_model.context import build_dynamic_context_model
from jka_model.context.checkpoint import load_context_checkpoint
from jka_model.data import ChannelStandardizer, load_cylinder_wake_dataset
from jka_model.manifold import (
    RawFieldAdaptiveRolloutDataset,
    assert_online_reencoding_required,
    configure_phase3_route,
    joint_markov_objective,
    move_raw_field_batch,
)
from jka_model.models import FieldJEPAKoopmanModel
from jka_model.residual.cache import file_sha256
from jka_model.utils import get_git_commit, load_checkpoint, set_global_seed


@dataclass(frozen=True, slots=True)
class Phase3JointTrainingResult:
    run_dir: Path
    checkpoint: Path
    completed_epochs: int
    trainable_parameters: int
    validation_metrics: dict[str, float]
    locked_test_metrics: dict[str, float]


@torch.no_grad()
def _frozen_target_latents(
    backbone: FieldJEPAKoopmanModel,
    normalizer: ChannelStandardizer,
    records: Any,
    device: torch.device,
    *,
    chunk_size: int = 16,
) -> dict[str, torch.Tensor]:
    """Cache only immutable JEPA teacher targets; online histories remain raw fields."""
    targets: dict[str, torch.Tensor] = {}
    backbone.target_encoder.eval()
    for record in records:
        encoded: list[torch.Tensor] = []
        for chunk in record.states_raw.split(chunk_size):
            model_states = normalizer.transform(chunk.to(device=device, dtype=torch.float32))
            encoded.append(backbone.encode_target(model_states).float().cpu())
        targets[record.trajectory_id] = torch.cat(encoded, dim=0)
    return targets


def _save_payload(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_joint_models(
    config: ProjectConfig,
    *,
    cache: Any,
    context_checkpoint: str | Path,
    backbone_checkpoint: str | Path,
    device: torch.device,
) -> tuple[
    FieldJEPAKoopmanModel,
    nn.Module,
    AdaptiveKoopmanModel,
    ChannelStandardizer,
    dict[str, Any],
]:
    from train.train_v0_6 import initialize_v0_6_model

    phase3 = config.v0_9_phase3
    phase2 = config.v0_9_phase2
    adaptive_config = config.v0_9_adaptive
    context_config = config.v0_8_context
    if phase3 is None or phase2 is None or adaptive_config is None or context_config is None:
        raise ValueError("Phase-3 joint model requires complete V0.8/V0.9 configuration")
    if file_sha256(backbone_checkpoint) != cache.backbone_checkpoint_sha256:
        raise ValueError("Phase-3 backbone/cache fingerprint mismatch")
    if file_sha256(context_checkpoint) != cache.context_checkpoint_sha256:
        raise ValueError("Phase-3 context/cache fingerprint mismatch")

    saved_backbone = load_checkpoint(backbone_checkpoint, map_location="cpu")
    if (
        saved_backbone.online_model_state is None
        or saved_backbone.target_model_state is None
        or saved_backbone.normalizer_state is None
    ):
        raise ValueError("Phase-3 backbone lacks JEPA/normalizer state")
    backbone = initialize_v0_6_model(config, device=device)
    backbone.load_online_state_dict(saved_backbone.online_model_state)
    backbone.target_encoder.load_state_dict(saved_backbone.target_model_state, strict=True)
    normalizer = ChannelStandardizer(eps=config.data.normalization.eps)
    normalizer.load_state_dict(saved_backbone.normalizer_state)
    if not normalizer.matches_state_dict(cache.normalizer_state):
        raise ValueError("Phase-3 backbone/cache normalizer mismatch")
    reference_encoder = copy.deepcopy(backbone.online_encoder).to(device)
    reference_encoder.requires_grad_(False)
    reference_encoder.eval()

    context_payload = load_context_checkpoint(context_checkpoint)
    history = int(context_payload["history_length_steps"])
    dynamic = build_dynamic_context_model(
        context_config,
        family=str(context_payload["context_family"]),
        latent_dim=cache.latent_dim,
        parameter_dim=cache.context_parameter_dim,
        history=history,
    )
    dynamic.load_state_dict(context_payload["best_context_state"], strict=True)
    operator = FactorizedAdaptiveOperator(
        cache.nominal_generator,
        context_config.context_dim,
        adaptive_config,
        phase2,
    )
    adaptive_model = AdaptiveKoopmanModel(dynamic.context_encoder, operator).to(device)
    assert_online_reencoding_required("joint", uses_frozen_latent_cache=False)
    configure_phase3_route(
        "joint",
        backbone=backbone,
        context_encoder=adaptive_model.context_encoder,
        operator=adaptive_model.operator_adapter,
        physical_decoder=None,
        joint_backbone_allowlist=phase3.joint_backbone_allowlist,
    )
    backbone.target_encoder.requires_grad_(False)
    backbone.target_encoder.eval()
    if backbone.koopman_core.A.requires_grad:
        raise RuntimeError("Phase-3 joint route must keep inherited A0 frozen")
    return backbone, reference_encoder, adaptive_model, normalizer, context_payload


@torch.no_grad()
def _evaluate(
    backbone: FieldJEPAKoopmanModel,
    reference_encoder: nn.Module,
    adaptive_model: AdaptiveKoopmanModel,
    loader: DataLoader,
    normalizer: ChannelStandardizer,
    residual_scale: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    config: ProjectConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    backbone.eval()
    adaptive_model.eval()
    totals: dict[str, float] = {}
    count = 0
    maximum = {"representation_drift", "physical_manifold_violation", "burden_max"}
    for raw in loader:
        batch = move_raw_field_batch(raw, device)
        objective = joint_markov_objective(
            backbone,
            reference_encoder,
            adaptive_model,
            batch,
            normalizer,
            residual_scale,
            condition_mean,
            condition_std,
            config,
            epoch=epoch,
            validation=True,
        )
        batch_size = batch["history_raw"].shape[0]
        for name, value in {"total": objective.total, **objective.terms}.items():
            scalar = float(value)
            if name in maximum:
                totals[name] = max(totals.get(name, float("-inf")), scalar)
            else:
                totals[name] = totals.get(name, 0.0) + scalar * batch_size
        count += batch_size
    if count == 0:
        raise RuntimeError("Phase-3 validation dataset is empty")
    return {
        name: value if name in maximum else value / count for name, value in totals.items()
    }


def train_v0_9_phase3_joint(
    config: ProjectConfig | str | Path,
    *,
    context_checkpoint: str | Path,
    adaptive_cache: str | Path,
    backbone_checkpoint: str | Path,
    physical_dataset: str | Path,
    run_dir: str | Path,
    frozen_target_cache: str | Path | None = None,
    device: str | torch.device = "cuda",
) -> Phase3JointTrainingResult:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    phase3 = resolved.v0_9_phase3
    training = resolved.v0_9_training
    cylinder = resolved.cylinder_wake_2d
    if phase3 is None or training is None or cylinder is None or not phase3.enabled:
        raise ValueError("Phase-3 joint training requires enabled complete configuration")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    destination = Path(run_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("config", "logs", "checkpoints", "evaluation", "metadata"):
        (destination / name).mkdir()
    save_config(resolved, destination / "config" / "resolved_config.yaml")
    print(
        f"[V0.9][phase3:joint:{resolved.v0_9_adaptive.condition_mode}] START "
        f"seed={resolved.training.seed} init={training.operator_initialization_seed} "
        f"epochs={training.epochs} device={selected}",
        flush=True,
    )
    set_global_seed(
        training.operator_initialization_seed,
        deterministic=resolved.training.deterministic,
    )
    cache = load_adaptive_cache(adaptive_cache)
    backbone, reference_encoder, adaptive_model, normalizer, context_payload = (
        _build_joint_models(
            resolved,
            cache=cache,
            context_checkpoint=context_checkpoint,
            backbone_checkpoint=backbone_checkpoint,
            device=selected,
        )
    )
    dataset = load_cylinder_wake_dataset(physical_dataset, cylinder)
    target_cache_path = None if frozen_target_cache is None else Path(frozen_target_cache)
    if target_cache_path is not None and target_cache_path.is_file():
        try:
            target_payload = torch.load(target_cache_path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover
            target_payload = torch.load(target_cache_path, map_location="cpu")
        if (
            not isinstance(target_payload, dict)
            or target_payload.get("schema_version") != "v0.9-phase3-target-1"
            or target_payload.get("backbone_checkpoint_sha256")
            != file_sha256(backbone_checkpoint)
        ):
            raise ValueError("Phase-3 frozen JEPA target cache provenance mismatch")
        frozen_targets = target_payload["targets"]
    else:
        frozen_targets = _frozen_target_latents(
            backbone, normalizer, dataset.records, selected
        )
        if target_cache_path is not None:
            target_cache_path.parent.mkdir(parents=True, exist_ok=True)
            _save_payload(
                {
                    "schema_version": "v0.9-phase3-target-1",
                    "backbone_checkpoint_sha256": file_sha256(backbone_checkpoint),
                    "targets": frozen_targets,
                },
                target_cache_path,
            )
    history = int(context_payload["history_length_steps"])
    maximum_horizon = max(
        *training.rollout_horizons,
        *training.active_observable_horizons,
    )
    datasets = {
        split: RawFieldAdaptiveRolloutDataset(
            cache,
            dataset.records,
            split,
            history,
            maximum_horizon,
            stride=phase3.raw_field_rollout_stride,
            frozen_target_latents=frozen_targets,
        )
        for split in ("train", "validation", "test")
    }
    loaders = {
        split: DataLoader(
            value,
            batch_size=phase3.raw_field_batch_size,
            shuffle=split == "train",
            num_workers=0,
        )
        for split, value in datasets.items()
    }
    residual_scale_cpu, condition_mean_cpu, condition_std_cpu = adaptive_training_scales(cache)
    condition_mean_cpu, condition_std_cpu = phase2_condition_scales(cache)
    residual_scale = residual_scale_cpu.to(selected)
    condition_mean = condition_mean_cpu.to(selected)
    condition_std = condition_std_cpu.to(selected)

    representation_parameters = [
        parameter for parameter in backbone.parameters() if parameter.requires_grad
    ]
    adaptive_parameters = [
        parameter for parameter in adaptive_model.parameters() if parameter.requires_grad
    ]
    if not representation_parameters or not adaptive_parameters:
        raise RuntimeError("Phase-3 joint route has an empty declared parameter group")
    if {id(value) for value in representation_parameters} & {
        id(value) for value in adaptive_parameters
    }:
        raise RuntimeError("Phase-3 optimizer parameter groups overlap")
    optimizer = AdamW(
        [
            {"params": representation_parameters, "lr": phase3.representation_learning_rate},
            {"params": adaptive_parameters, "lr": phase3.operator_learning_rate},
        ],
        weight_decay=training.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=max(1, training.epochs // 2), gamma=0.5)
    amp_enabled = selected.type == "cuda" and training.precision != "fp32"
    amp_dtype = torch.float16 if training.precision == "amp_fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype is torch.float16)
    nominal_core = backbone.koopman_core.A.detach().clone()
    nominal_adapter = adaptive_model.operator_adapter.nominal_generator.detach().clone()
    best_key = (float("inf"), float("inf"), float("inf"))
    best_states: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_metrics: dict[str, float] = {}
    stale = 0
    completed = 0
    history_rows: list[dict[str, Any]] = []
    maturity_fraction = max(
        max(training.rollout_start_fractions),
        training.physics_start_fraction + training.physics_ramp_duration_fraction,
    )
    earliest_stop_epoch = math.ceil(maturity_fraction * max(training.epochs - 1, 1)) + 1
    for epoch in range(training.epochs):
        backbone.train()
        adaptive_model.train()
        for raw in loaders["train"]:
            batch = move_raw_field_batch(raw, selected)
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast(
                device_type=selected.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            )
            with autocast:
                objective = joint_markov_objective(
                    backbone,
                    reference_encoder,
                    adaptive_model,
                    batch,
                    normalizer,
                    residual_scale,
                    condition_mean,
                    condition_std,
                    resolved,
                    epoch=epoch,
                    validation=False,
                )
            scaler.scale(objective.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [*representation_parameters, *adaptive_parameters],
                training.gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        if not torch.equal(backbone.koopman_core.A.detach(), nominal_core) or not torch.equal(
            adaptive_model.operator_adapter.nominal_generator.detach(), nominal_adapter
        ):
            raise RuntimeError("Phase-3 joint training changed the frozen nominal generator A0")
        metrics = _evaluate(
            backbone,
            reference_encoder,
            adaptive_model,
            loaders["validation"],
            normalizer,
            residual_scale,
            condition_mean,
            condition_std,
            resolved,
            selected,
            epoch,
        )
        drift_violation = max(
            metrics["representation_drift"] - phase3.max_normalized_representation_drift,
            0.0,
        )
        key = (
            1.0 if drift_violation > 0 else 0.0,
            metrics["physical_manifold_violation"] + drift_violation,
            metrics["total"],
        )
        history_rows.append({"epoch": epoch + 1, **metrics})
        if key < best_key:
            best_key = key
            best_states = (
                copy.deepcopy(backbone.state_dict()),
                copy.deepcopy(adaptive_model.state_dict()),
            )
            best_metrics = dict(metrics)
            stale = 0
        else:
            stale += 1
        completed = epoch + 1
        if completed >= earliest_stop_epoch and stale >= training.patience:
            break
    if best_states is None:
        raise RuntimeError("Phase-3 joint training selected no finite checkpoint")
    backbone.load_state_dict(best_states[0], strict=True)
    adaptive_model.load_state_dict(best_states[1], strict=True)
    locked_test_metrics = _evaluate(
        backbone,
        reference_encoder,
        adaptive_model,
        loaders["test"],
        normalizer,
        residual_scale,
        condition_mean,
        condition_std,
        resolved,
        selected,
        max(completed - 1, 0),
    )
    (destination / "logs" / "history.json").write_text(
        json.dumps(history_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trainable_count = sum(
        parameter.numel() for parameter in [*representation_parameters, *adaptive_parameters]
    )
    checkpoint = destination / "checkpoints" / "best_joint_markov.pt"
    _save_payload(
        {
            "schema_version": "v0.9-phase3-joint-1",
            "route": "joint",
            "config": resolved.to_dict(),
            "config_hash": resolved.stable_hash,
            "backbone_state": backbone.state_dict(),
            "adaptive_state": adaptive_model.state_dict(),
            "reference_encoder_state": reference_encoder.state_dict(),
            "normalizer_state": normalizer.state_dict(),
            "source_backbone_sha256": file_sha256(backbone_checkpoint),
            "source_context_sha256": file_sha256(context_checkpoint),
            # The operator is initialized from the same seed as the frozen route;
            # loading a trained Phase-2 operator here would give joint an extra
            # optimization budget and invalidate the matched comparison.
            "source_phase2_checkpoint_sha256": None,
            "adaptive_cache_fingerprint": cache.fingerprint,
            "frozen_target_cache_sha256": (
                None if target_cache_path is None else file_sha256(target_cache_path)
            ),
            "condition_mean": condition_mean_cpu,
            "condition_std": condition_std_cpu,
            "residual_scale": residual_scale_cpu,
            "completed_epochs": completed,
            "trainable_parameters": trainable_count,
            "best_validation_metrics": best_metrics,
            "locked_test_metrics": locked_test_metrics,
            "nominal_generator_unchanged": True,
            "git_commit": get_git_commit(Path.cwd()),
        },
        checkpoint,
    )
    (destination / "evaluation" / "validation_metrics.json").write_text(
        json.dumps(best_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "evaluation" / "locked_test_metrics.json").write_text(
        json.dumps(locked_test_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[V0.9][phase3:joint:{resolved.v0_9_adaptive.condition_mode}] PASS "
        f"epochs={completed} val_total={best_metrics['total']:.6g} "
        f"drift={best_metrics['representation_drift']:.6g}",
        flush=True,
    )
    return Phase3JointTrainingResult(
        destination,
        checkpoint,
        completed,
        trainable_count,
        best_metrics,
        locked_test_metrics,
    )
