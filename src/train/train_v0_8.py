"""Canonical V0.8 context-only residual-supervision training."""

from __future__ import annotations

import csv
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from jka_model.config import ProjectConfig, load_config, save_config
from jka_model.constants import ARCHITECTURE_REVISION, CHECKPOINT_SCHEMA_VERSION, PROJECT_VERSION
from jka_model.context import (
    ContextWindowDataset,
    build_dynamic_context_model,
    context_prediction_metrics,
    load_v0_7_route,
    residual_training_scales,
)
from jka_model.context.checkpoint import load_context_checkpoint, save_context_checkpoint
from jka_model.residual import load_residual_cache
from jka_model.residual.cache import file_sha256
from jka_model.training import (
    TrainStage,
    assert_optimizer_matches_trainable_params,
    configure_train_stage,
)
from jka_model.utils import (
    RNGState,
    capture_rng_state,
    get_git_commit,
    restore_rng_state,
    set_global_seed,
)


@dataclass(frozen=True, slots=True)
class V08TrainingResult:
    run_dir: Path
    status: str
    residual_route: str
    context_family: str | None
    latest_checkpoint: Path | None
    best_checkpoint: Path | None
    start_epoch: int
    completed_epochs: int
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]


def _require_v0_8(config: ProjectConfig) -> None:
    required = (
        config.koopman,
        config.v0_8_routing,
        config.v0_8_context,
        config.v0_8_training,
        config.v0_8_evaluation,
    )
    if any(section is None for section in required):
        raise ValueError("train_v0_8 requires the complete V0.8 configuration")
    if config.training.stage is not TrainStage.CONTEXT:
        raise ValueError("V0.8 training.stage must be context")


def _batch(batch: dict[str, Any], device: torch.device, include_parameters: bool):
    parameters = batch["parameters"].to(device=device, dtype=torch.float32)
    if not include_parameters:
        parameters = parameters[:, :0]
    return (
        batch["history_z"].to(device=device, dtype=torch.float32),
        batch["history_dts"].to(device=device, dtype=torch.float32),
        batch["next_dt"].to(device=device, dtype=torch.float32),
        parameters,
        batch["target_residual"].to(device=device, dtype=torch.float32),
        batch["target_adequacy"].to(device=device, dtype=torch.float32),
    )


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    include_parameters: bool,
    residual_scale: torch.Tensor,
    adequacy_scale: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    residual_predictions: list[torch.Tensor] = []
    residual_targets: list[torch.Tensor] = []
    adequacy_predictions: list[torch.Tensor] = []
    adequacy_targets: list[torch.Tensor] = []
    for raw in loader:
        history_z, history_dts, next_dt, parameters, target_r, target_q = _batch(
            raw, device, include_parameters
        )
        _, prediction_r, prediction_q = model(history_z, history_dts, next_dt, parameters)
        residual_predictions.append(prediction_r.cpu())
        residual_targets.append(target_r.cpu())
        adequacy_predictions.append(prediction_q.cpu())
        adequacy_targets.append(target_q.cpu())
    return context_prediction_metrics(
        torch.cat(residual_predictions),
        torch.cat(residual_targets),
        torch.cat(adequacy_predictions),
        torch.cat(adequacy_targets),
        residual_scale.cpu(),
        adequacy_scale.cpu(),
    )


def train_v0_8(
    config: ProjectConfig | str | Path,
    *,
    backbone_checkpoint: str | Path,
    residual_cache: str | Path,
    v0_7_route_result: str | Path | None,
    run_dir: str | Path,
    device: str | torch.device | None = None,
    resume_from: str | Path | None = None,
) -> V08TrainingResult:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    _require_v0_8(resolved)
    assert resolved.koopman and resolved.v0_8_context and resolved.v0_8_training
    assert resolved.v0_8_routing
    route_path = v0_7_route_result or resolved.v0_8_routing.v0_7_result
    if not route_path:
        raise ValueError("V0.8 requires an explicit V0.7 route result")
    route = load_v0_7_route(route_path)
    destination = Path(run_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("config", "metadata", "logs", "checkpoints", "evaluation", "reports"):
        (destination / name).mkdir()
    save_config(resolved, destination / "config" / "resolved_config.yaml")
    (destination / "evaluation" / "v0_8_problem_route.json").write_text(
        json.dumps(route.source_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if route.context_family is None:
        status = "DIAGNOSTIC_ONLY" if route.residual_route == "R1" else "INCONCLUSIVE"
        report = {
            "status": status,
            "residual_route": route.residual_route,
            "context_family": None,
            "reason": "R1 requires diagnosis; INCONCLUSIVE forbids architecture escalation",
        }
        (destination / "reports" / "route_stop.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[V0.8][train] PASS status={status} route={route.residual_route}", flush=True)
        return V08TrainingResult(
            destination, status, route.residual_route, None, None, None, 0, 0, {}, {}
        )
    family = (
        route.context_family
        if resolved.v0_8_context.family == "auto"
        else resolved.v0_8_context.family
    )
    if route.residual_route == "R2" and family != "instantaneous":
        raise ValueError("R2 formal route permits only instantaneous context")
    if route.residual_route == "R3" and family not in {
        "attention",
        "history_mlp",
        "instantaneous",
        "instantaneous_matched",
    }:
        raise ValueError("R3 permits Attention plus its registered temporal/current controls")
    history = int(route.history_length or resolved.v0_8_context.history_length)
    selected = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
    )
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(
        f"[V0.8][train:{family}] START device={selected} route={route.residual_route} "
        f"H={history} epochs={resolved.v0_8_training.epochs}",
        flush=True,
    )
    set_global_seed(
        resolved.v0_8_training.context_initialization_seed,
        deterministic=resolved.training.deterministic,
    )
    cache = load_residual_cache(residual_cache)
    backbone_sha = file_sha256(backbone_checkpoint)
    if cache.backbone_checkpoint_sha256 != backbone_sha:
        raise ValueError("V0.8 cache/backbone checkpoint fingerprint mismatch")
    residual_scale_cpu, adequacy_scale_cpu, scale_fingerprint = residual_training_scales(cache)
    residual_scale = residual_scale_cpu.to(selected)
    adequacy_scale = adequacy_scale_cpu.to(selected)
    parameter_dim = cache.parameter_dim if resolved.v0_8_context.include_parameters else 0
    model = build_dynamic_context_model(
        resolved.v0_8_context,
        family=family,
        latent_dim=cache.latent_dim,
        parameter_dim=parameter_dim,
        history=history,
    ).to(selected)
    configure_train_stage(model, TrainStage.CONTEXT)
    optimizer = AdamW(
        model.parameters(),
        lr=resolved.v0_8_training.learning_rate,
        weight_decay=resolved.v0_8_training.weight_decay,
    )
    assert_optimizer_matches_trainable_params(model, optimizer)
    scheduler = StepLR(optimizer, step_size=max(1, resolved.v0_8_training.epochs // 2), gamma=0.5)
    amp_enabled = selected.type == "cuda" and resolved.v0_8_training.precision != "fp32"
    amp_dtype = torch.float16 if resolved.v0_8_training.precision == "amp_fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype is torch.float16)
    # Test trajectories are deliberately not instantiated in training. They are opened
    # only by evaluate_v0_8 after the validation-selected family/H/checkpoint is locked.
    datasets = {
        split: ContextWindowDataset(cache, split, history) for split in ("train", "validation")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=resolved.v0_8_training.batch_size,
            shuffle=split == "train",
            num_workers=0,
        )
        for split, dataset in datasets.items()
    }
    start_epoch = global_step = optimizer_update_step = 0
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    if resume_from is not None:
        saved = load_context_checkpoint(resume_from)
        if saved["config_hash"] != resolved.stable_hash:
            raise ValueError("V0.8 resume config mismatch")
        if saved["residual_cache_fingerprint"] != cache.fingerprint:
            raise ValueError("V0.8 resume cache mismatch")
        model.load_state_dict(saved["context_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        if saved["amp_scaler_state"] is not None:
            scaler.load_state_dict(saved["amp_scaler_state"])
        restore_rng_state(RNGState.from_checkpoint_dict(saved["rng_state"]))
        start_epoch = int(saved["epoch"])
        global_step = int(saved["global_step"])
        optimizer_update_step = int(saved["optimizer_update_step"])
        best_loss = float(saved["best_validation_loss"])
        best_state = dict(saved["best_context_state"])
        epochs_without_improvement = int(saved["epochs_without_improvement"])
    latest = destination / "checkpoints" / "latest.pt"
    best = destination / "checkpoints" / "best.pt"

    def checkpoint_payload(epoch: int) -> dict[str, Any]:
        if best_state is None:
            raise RuntimeError("cannot checkpoint before validation selects a state")
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "architecture_revision": ARCHITECTURE_REVISION,
            "project_version": PROJECT_VERSION,
            "train_stage": TrainStage.CONTEXT.value,
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_update_step": optimizer_update_step,
            "context_family": family,
            "residual_route": route.residual_route,
            "history_length_steps": history,
            "context_dim": resolved.v0_8_context.context_dim,
            "flow_data_seed": resolved.training.seed,
            "backbone_seed": resolved.training.seed,
            "context_init_seed": resolved.v0_8_training.context_initialization_seed,
            "context_state": model.state_dict(),
            "best_context_state": best_state,
            "best_validation_loss": best_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "amp_scaler_state": scaler.state_dict() if amp_enabled else None,
            "rng_state": capture_rng_state().to_checkpoint_dict(),
            "config": resolved.to_dict(),
            "config_hash": resolved.stable_hash,
            "backbone_checkpoint_sha256": backbone_sha,
            "residual_cache_fingerprint": cache.fingerprint,
            "residual_scale_fingerprint": scale_fingerprint,
            "residual_training_scale": residual_scale_cpu,
            "adequacy_training_scale": adequacy_scale_cpu,
            "split_fingerprint": cache.split_fingerprint,
            "normalizer_fingerprint": cache.normalizer_fingerprint,
            "git_commit": get_git_commit(Path.cwd()),
            "runtime": {
                "device": str(selected),
                "dtype": str(next(model.parameters()).dtype),
                "precision": resolved.v0_8_training.precision,
                "torch_version": torch.__version__,
            },
        }

    log_path = destination / "logs" / "epoch_metrics.csv"
    completed_epochs = start_epoch
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        for epoch in range(start_epoch, resolved.v0_8_training.epochs):
            model.train()
            total = count = 0
            for raw in loaders["train"]:
                history_z, history_dts, next_dt, parameters, target_r, target_q = _batch(
                    raw, selected, resolved.v0_8_context.include_parameters
                )
                optimizer.zero_grad(set_to_none=True)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast:
                    _, prediction_r, prediction_q = model(
                        history_z, history_dts, next_dt, parameters
                    )
                    residual_loss = ((prediction_r - target_r) / residual_scale).square().mean()
                    adequacy_loss = ((prediction_q - target_q) / adequacy_scale).square().mean()
                    loss = residual_loss + resolved.v0_8_training.lambda_adequacy * adequacy_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), resolved.v0_8_training.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach()) * target_r.shape[0]
                count += target_r.shape[0]
                global_step += 1
                optimizer_update_step += 1
            scheduler.step()
            validation = _evaluate(
                model,
                loaders["validation"],
                selected,
                resolved.v0_8_context.include_parameters,
                residual_scale,
                adequacy_scale,
            )
            validation_loss = validation["residual_standardized_mse"] + (
                resolved.v0_8_training.lambda_adequacy * validation["adequacy_standardized_mse"]
            )
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": total / count,
                    "validation_loss": validation_loss,
                }
            )
            stream.flush()
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
                save_context_checkpoint(checkpoint_payload(epoch + 1), best)
            else:
                epochs_without_improvement += 1
            completed_epochs = epoch + 1
            save_context_checkpoint(checkpoint_payload(completed_epochs), latest)
            if epochs_without_improvement >= resolved.v0_8_training.patience:
                break
    if best_state is None:
        raise RuntimeError("V0.8 training did not produce a validation checkpoint")
    if not latest.exists():
        save_context_checkpoint(checkpoint_payload(completed_epochs), latest)
    model.load_state_dict(best_state)
    if not best.exists():
        save_context_checkpoint(checkpoint_payload(completed_epochs), best)
    validation_metrics = _evaluate(
        model,
        loaders["validation"],
        selected,
        resolved.v0_8_context.include_parameters,
        residual_scale,
        adequacy_scale,
    )
    test_metrics: dict[str, float] = {}
    summary = {
        "status": "PASS",
        "residual_route": route.residual_route,
        "context_family": family,
        "history_length_steps": history,
        "completed_epochs": completed_epochs,
        "validation": validation_metrics,
        "test_locked_confirmation": "NOT_OPENED_DURING_TRAINING",
    }
    (destination / "evaluation" / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[V0.8][train:{family}] PASS validation_nrmse="
        f"{validation_metrics['residual_nrmse']:.6g} test=LOCKED",
        flush=True,
    )
    return V08TrainingResult(
        destination,
        "PASS",
        route.residual_route,
        family,
        latest,
        best,
        start_epoch,
        completed_epochs,
        validation_metrics,
        test_metrics,
    )
