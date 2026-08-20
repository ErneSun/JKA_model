"""Canonical closure-only V0.7 residual training."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from jka_model.config import ProjectConfig, load_config, save_config
from jka_model.constants import (
    ARCHITECTURE_REVISION,
    CHECKPOINT_SCHEMA_VERSION,
    V0_7_CHECKPOINT_SCHEMA_VERSION,
    V0_7_PROJECT_VERSION,
)
from jka_model.residual import (
    ResidualCache,
    ResidualKoopmanModel,
    ResidualWindowDataset,
    build_closure,
    closure_metrics,
    load_residual_cache,
)
from jka_model.residual.cache import file_sha256
from jka_model.residual.checkpoint import load_residual_checkpoint, save_residual_checkpoint
from jka_model.training import (
    TrainStage,
    assert_optimizer_matches_trainable_params,
    configure_train_stage,
)
from jka_model.utils import (
    RNGState,
    capture_rng_state,
    get_git_commit,
    load_checkpoint,
    restore_rng_state,
    set_global_seed,
)
from train.train_v0_6 import initialize_v0_6_model


@dataclass(frozen=True, slots=True)
class V07TrainingResult:
    run_dir: Path
    latest_checkpoint: Path
    best_checkpoint: Path
    variant: str
    start_epoch: int
    completed_epochs: int
    initial_validation_mse: float
    best_validation_mse: float
    test_metrics: dict[str, Any]


def _require_v0_7(config: ProjectConfig) -> None:
    required = (
        config.jepa_loss,
        config.ema,
        config.v0_6_evaluation,
        config.residual_closure,
        config.residual_training,
        config.memory_sweep,
        config.v0_7_evaluation,
    )
    if any(section is None for section in required):
        raise ValueError("train_v0_7 requires complete V0.6 inheritance and V0.7 sections")
    if config.training.stage is not TrainStage.RESIDUAL:
        raise ValueError("V0.7 training.stage must be residual")


def _backbone_contract(config: ProjectConfig) -> dict[str, Any]:
    payload = config.to_dict()
    names = (
        "architecture",
        "data",
        "koopman",
        "advection_diffusion_2d",
        "cylinder_wake_2d",
        "field_autoencoder",
        "field_loss",
        "v0_5_evaluation",
        "jepa_loss",
        "ema",
        "v0_6_evaluation",
    )
    return {name: payload[name] for name in names}


def load_frozen_v0_6_backbone(
    checkpoint: str | Path, config: ProjectConfig, device: torch.device
) -> tuple[ResidualKoopmanModel, Any]:
    """Load exact V0.6 online/core/decoder/target state, then freeze every parameter."""
    saved = load_checkpoint(checkpoint, map_location="cpu")
    if saved.train_stage is not TrainStage.JEPA:
        raise ValueError("V0.7 initialization requires a V0.6 JEPA-stage checkpoint")
    if saved.config is None or _backbone_contract(saved.config) != _backbone_contract(config):
        raise ValueError("V0.7 config does not inherit the V0.6 backbone contract exactly")
    if saved.online_model_state is None or saved.target_model_state is None:
        raise ValueError("V0.6 checkpoint lacks online or target state")
    backbone = initialize_v0_6_model(config, device=device)
    backbone.load_online_state_dict(saved.online_model_state)
    backbone.target_encoder.load_state_dict(saved.target_model_state, strict=True)
    backbone.requires_grad_(False)
    assert config.residual_closure and config.koopman
    parameter_dim = config.data.parameter_dim if config.residual_closure.include_parameters else 0
    placeholder = build_closure(
        "zero",
        latent_dim=config.koopman.state_dim,
        history=config.residual_closure.history,
        parameter_dim=parameter_dim,
        hidden_dim=config.residual_closure.hidden_dim,
        depth=config.residual_closure.depth,
    )
    return ResidualKoopmanModel(backbone, placeholder).to(device), saved


def _batch_to_device(batch: dict[str, Any], device: torch.device, include_parameters: bool):
    parameters = batch["parameters"].to(device=device, dtype=torch.float32)
    if not include_parameters:
        parameters = parameters[:, :0]
    return (
        batch["history_z"].to(device=device, dtype=torch.float32),
        batch["history_dts"].to(device=device, dtype=torch.float32),
        batch["next_dt"].to(device=device, dtype=torch.float32),
        parameters,
        batch["target"].to(device=device, dtype=torch.float32),
    )


@torch.no_grad()
def _evaluate_loader(
    model: ResidualKoopmanModel,
    loader: DataLoader,
    device: torch.device,
    include_parameters: bool,
    residual_scale: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        history_z, history_dts, next_dt, parameters, target = _batch_to_device(
            batch, device, include_parameters
        )
        predictions.append(model.residual_head(history_z, history_dts, next_dt, parameters).cpu())
        targets.append(target.cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return closure_metrics(prediction, target, residual_scale)


def _training_residual_scale(cache: ResidualCache) -> torch.Tensor:
    residuals = torch.cat([item.residuals.float() for item in cache.select("train")])
    scale = residuals.square().mean(dim=0).sqrt()
    relative_floor = float(residuals.square().mean().sqrt()) * 1e-3
    return scale.clamp_min(max(relative_floor, torch.finfo(scale.dtype).eps))


def _residual_scale_fingerprint(scale: torch.Tensor) -> str:
    value = scale.detach().cpu().contiguous().to(torch.float64)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _loaders(cache: ResidualCache, config: ProjectConfig, variant: str):
    assert config.residual_closure and config.residual_training
    kwargs = {
        "history": config.residual_closure.history,
        "shuffle_history": variant == "shuffled_history",
        "shuffle_seed": config.residual_training.initialization_seed,
    }
    train = ResidualWindowDataset(cache, "train", **kwargs)
    validation = ResidualWindowDataset(cache, "validation", **kwargs)
    test = ResidualWindowDataset(cache, "test", **kwargs)
    batch_size = config.residual_training.batch_size
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def _physical_history_summary(cache: ResidualCache, history: int) -> dict[str, float]:
    if history == 1:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    spans = torch.cat(
        [
            trajectory.dts[index - history + 1 : index].sum().reshape(1)
            for trajectory in cache.trajectories
            for index in range(history - 1, trajectory.residuals.shape[0])
        ]
    ).double()
    return {"mean": float(spans.mean()), "min": float(spans.min()), "max": float(spans.max())}


def train_v0_7(
    config: ProjectConfig | str | Path,
    *,
    backbone_checkpoint: str | Path,
    cache_path: str | Path,
    variant: str,
    run_dir: str | Path,
    device: str | torch.device | None = None,
    resume_from: str | Path | None = None,
) -> V07TrainingResult:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    _require_v0_7(resolved)
    assert resolved.residual_closure and resolved.residual_training and resolved.koopman
    checkpoint_schema = (
        V0_7_CHECKPOINT_SCHEMA_VERSION
        if resolved.project_version == V0_7_PROJECT_VERSION
        else CHECKPOINT_SCHEMA_VERSION
    )
    checkpoint_project = resolved.project_version
    if variant not in resolved.residual_closure.variants:
        raise ValueError(f"closure variant {variant!r} is not enabled")
    selected = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
    )
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    destination = Path(run_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("config", "metadata", "logs", "checkpoints", "evaluation", "reports"):
        (destination / name).mkdir()
    print(
        f"[V0.7][train:{variant}] START device={selected} "
        f"epochs={resolved.residual_training.epochs}",
        flush=True,
    )
    closure_seed = resolved.residual_training.initialization_seed
    set_global_seed(closure_seed, deterministic=resolved.training.deterministic)
    cache = load_residual_cache(cache_path)
    residual_scale_cpu = _training_residual_scale(cache)
    residual_scale_fingerprint = _residual_scale_fingerprint(residual_scale_cpu)
    residual_scale = residual_scale_cpu.to(selected)
    source_sha = file_sha256(backbone_checkpoint)
    if cache.backbone_checkpoint_sha256 != source_sha:
        raise ValueError("residual cache was built from a different V0.6 checkpoint")
    shell, saved_backbone = load_frozen_v0_6_backbone(backbone_checkpoint, resolved, selected)
    if cache.backbone_config_hash != saved_backbone.config_hash:
        raise ValueError("residual cache/backbone config hash mismatch")
    parameter_dim = (
        resolved.data.parameter_dim if resolved.residual_closure.include_parameters else 0
    )
    closure = build_closure(
        variant,
        latent_dim=resolved.koopman.state_dim,
        history=resolved.residual_closure.history,
        parameter_dim=parameter_dim,
        hidden_dim=resolved.residual_closure.hidden_dim,
        depth=resolved.residual_closure.depth,
    ).to(selected)
    model = ResidualKoopmanModel(initialize_v0_6_model(resolved, device=selected), closure).to(
        selected
    )
    model.load_backbone_state_dict(shell.backbone_state_dict())
    configure_train_stage(model, TrainStage.RESIDUAL)
    train_loader, validation_loader, test_loader = _loaders(cache, resolved, variant)
    optimizer = None
    scheduler = None
    if variant != "zero":
        optimizer = AdamW(
            model.residual_head.parameters(),
            lr=resolved.residual_training.learning_rate,
            weight_decay=resolved.residual_training.weight_decay,
        )
        assert_optimizer_matches_trainable_params(model, optimizer)
        scheduler = StepLR(
            optimizer, step_size=max(1, resolved.residual_training.epochs // 2), gamma=0.5
        )
    amp_enabled = selected.type == "cuda" and resolved.residual_training.precision != "fp32"
    amp_dtype = (
        torch.float16 if resolved.residual_training.precision == "amp_fp16" else torch.bfloat16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype is torch.float16)
    start_epoch = global_step = 0
    best_mse = float("inf")
    best_state: dict[str, Any] | None = None
    if resume_from is not None:
        resumed = load_residual_checkpoint(resume_from)
        if resumed["config_hash"] != resolved.stable_hash or resumed["closure_variant"] != variant:
            raise ValueError("V0.7 resume config/variant mismatch")
        if resumed["cache_fingerprint"] != cache.fingerprint:
            raise ValueError("V0.7 resume residual cache mismatch")
        if resumed["residual_scale_fingerprint"] != residual_scale_fingerprint:
            raise ValueError("V0.7 resume residual scale mismatch")
        model.load_backbone_state_dict(resumed["backbone_state"])
        model.residual_head.load_state_dict(resumed["closure_state"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(resumed["optimizer_state"])
            assert scheduler is not None
            scheduler.load_state_dict(resumed["scheduler_state"])
            if resumed["amp_scaler_state"] is not None:
                scaler.load_state_dict(resumed["amp_scaler_state"])
        restore_rng_state(RNGState.from_checkpoint_dict(resumed["rng_state"]))
        start_epoch = int(resumed["epoch"])
        global_step = int(resumed["global_step"])
    save_config(resolved, destination / "config" / "resolved_config.yaml")
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    (destination / "metadata" / "provenance.json").write_text(
        json.dumps(
            {
                "project_version": checkpoint_project,
                "architecture_revision": ARCHITECTURE_REVISION,
                "train_stage": TrainStage.RESIDUAL.value,
                "variant": variant,
                "backbone_checkpoint": str(Path(backbone_checkpoint).resolve()),
                "backbone_checkpoint_sha256": source_sha,
                "backbone_config_hash": saved_backbone.config_hash,
                "cache": str(Path(cache_path).resolve()),
                "cache_fingerprint": cache.fingerprint,
                "data_fingerprint": cache.data_fingerprint,
                "split_fingerprint": cache.split_fingerprint,
                "normalizer_fingerprint": cache.normalizer_fingerprint,
                "closure_family": type(model.residual_head).__name__,
                "history_length_steps": resolved.residual_closure.history,
                "history_length_physical_time": _physical_history_summary(
                    cache, resolved.residual_closure.history
                ),
                "parameter_matched_control": variant == "instantaneous",
                "history_shuffled": variant == "shuffled_history",
                "history_shuffle": variant == "shuffled_history",
                "backbone_data_seed": resolved.training.seed,
                "closure_initialization_seed": closure_seed,
                "closure_init_seed": closure_seed,
                "residual_training_scale": residual_scale_cpu.tolist(),
                "residual_scale_fingerprint": residual_scale_fingerprint,
                "residual_loss": "per_dimension_train_rms_standardized_mse",
                "frozen_backbone": True,
                "target_encoder_used_for_residual": False,
                "trainable_parameters": trainable_parameters,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    initial = _evaluate_loader(
        model,
        validation_loader,
        selected,
        resolved.residual_closure.include_parameters,
        residual_scale_cpu,
    )
    initial_mse = float(initial["standardized_mse"])
    history_path = destination / "logs" / "epoch_metrics.csv"
    latest_path = destination / "checkpoints" / "latest.pt"
    best_path = destination / "checkpoints" / "best.pt"
    detailed = selected.type != "cuda"
    stale = 0
    completed = start_epoch
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch",
                "global_step",
                "train_standardized_mse",
                "validation_standardized_mse",
                "validation_raw_mse",
                "seconds",
            ],
        )
        writer.writeheader()
        epoch_limit = start_epoch if variant == "zero" else resolved.residual_training.epochs
        for epoch in range(start_epoch, epoch_limit):
            assert optimizer is not None and scheduler is not None
            started = time.perf_counter()
            model.train()
            total = count = 0
            for batch in train_loader:
                history_z, history_dts, next_dt, parameters, target = _batch_to_device(
                    batch, selected, resolved.residual_closure.include_parameters
                )
                optimizer.zero_grad(set_to_none=True)
                context = torch.autocast("cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
                with context:
                    prediction = model.residual_head(history_z, history_dts, next_dt, parameters)
                    loss = ((prediction - target) / residual_scale).square().mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.residual_head.parameters(), resolved.residual_training.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                size = target.shape[0]
                total += float(loss.detach()) * size
                count += size
                global_step += 1
            scheduler.step()
            validation = _evaluate_loader(
                model,
                validation_loader,
                selected,
                resolved.residual_closure.include_parameters,
                residual_scale_cpu,
            )
            validation_mse = float(validation["standardized_mse"])
            completed = epoch + 1
            writer.writerow(
                {
                    "epoch": completed,
                    "global_step": global_step,
                    "train_standardized_mse": total / count,
                    "validation_standardized_mse": validation_mse,
                    "validation_raw_mse": validation["mse"],
                    "seconds": time.perf_counter() - started,
                }
            )
            stream.flush()
            if validation_mse < best_mse:
                best_mse = validation_mse
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.residual_head.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            payload = {
                "schema_version": checkpoint_schema,
                "architecture_revision": ARCHITECTURE_REVISION,
                "project_version": checkpoint_project,
                "train_stage": TrainStage.RESIDUAL.value,
                "epoch": completed,
                "global_step": global_step,
                "closure_variant": variant,
                "backbone_data_seed": resolved.training.seed,
                "closure_init_seed": closure_seed,
                "history_length_steps": resolved.residual_closure.history,
                "backbone_state": model.backbone_state_dict(),
                "closure_state": model.residual_head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "amp_scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                "rng_state": capture_rng_state().to_checkpoint_dict(),
                "normalizer_state": cache.normalizer_state,
                "problem_spec": None
                if saved_backbone.problem_spec is None
                else saved_backbone.problem_spec.to_dict(),
                "config": resolved.to_dict(),
                "config_hash": resolved.stable_hash,
                "data_fingerprint": cache.data_fingerprint,
                "split_manifest": cache.split_manifest,
                "backbone_checkpoint_sha256": source_sha,
                "cache_fingerprint": cache.fingerprint,
                "residual_training_scale": residual_scale_cpu.tolist(),
                "residual_scale_fingerprint": residual_scale_fingerprint,
                "git_commit": get_git_commit(Path.cwd()),
            }
            save_residual_checkpoint(payload, latest_path)
            if best_state is not None and validation_mse == best_mse:
                payload["closure_state"] = best_state
                save_residual_checkpoint(payload, best_path)
            if detailed:
                print(
                    f"[V0.7][train:{variant}] epoch={completed} "
                    f"train_standardized_mse={total / count:.6g} "
                    f"val_standardized_mse={validation_mse:.6g}",
                    flush=True,
                )
            if stale >= resolved.residual_training.patience:
                break
    if variant == "zero":
        best_mse = initial_mse
        payload = {
            "schema_version": checkpoint_schema,
            "architecture_revision": ARCHITECTURE_REVISION,
            "project_version": checkpoint_project,
            "train_stage": TrainStage.RESIDUAL.value,
            "epoch": 0,
            "global_step": 0,
            "closure_variant": variant,
            "backbone_data_seed": resolved.training.seed,
            "closure_init_seed": closure_seed,
            "history_length_steps": resolved.residual_closure.history,
            "backbone_state": model.backbone_state_dict(),
            "closure_state": model.residual_head.state_dict(),
            "optimizer_state": None,
            "scheduler_state": None,
            "amp_scaler_state": None,
            "rng_state": capture_rng_state().to_checkpoint_dict(),
            "normalizer_state": cache.normalizer_state,
            "problem_spec": None
            if saved_backbone.problem_spec is None
            else saved_backbone.problem_spec.to_dict(),
            "config": resolved.to_dict(),
            "config_hash": resolved.stable_hash,
            "data_fingerprint": cache.data_fingerprint,
            "split_manifest": cache.split_manifest,
            "backbone_checkpoint_sha256": source_sha,
            "cache_fingerprint": cache.fingerprint,
            "residual_training_scale": residual_scale_cpu.tolist(),
            "residual_scale_fingerprint": residual_scale_fingerprint,
            "git_commit": get_git_commit(Path.cwd()),
        }
        save_residual_checkpoint(payload, latest_path)
        save_residual_checkpoint(payload, best_path)
    else:
        model.residual_head.load_state_dict(load_residual_checkpoint(best_path)["closure_state"])
    test_metrics = _evaluate_loader(
        model,
        test_loader,
        selected,
        resolved.residual_closure.include_parameters,
        residual_scale_cpu,
    )
    (destination / "evaluation" / "teacher_forced_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[V0.7][train:{variant}] PASS best_val_standardized_mse={best_mse:.6g} "
        f"test_r2={test_metrics['r2']:.6g} run={destination}",
        flush=True,
    )
    return V07TrainingResult(
        run_dir=destination,
        latest_checkpoint=latest_path,
        best_checkpoint=best_path,
        variant=variant,
        start_epoch=start_epoch,
        completed_epochs=completed,
        initial_validation_mse=initial_mse,
        best_validation_mse=best_mse,
        test_metrics=test_metrics,
    )
