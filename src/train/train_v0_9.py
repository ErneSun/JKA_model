"""Canonical V0.9 operator-only training over a frozen V0.8 context."""

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

from jka_model.adaptive import (
    AdaptiveKoopmanModel,
    AdaptiveWindowDataset,
    LowRankAdaptiveOperator,
    adaptive_training_scales,
    load_adaptive_cache,
    load_adaptive_checkpoint,
    operator_burden,
    save_adaptive_checkpoint,
    symmetric_abscissa_proxy,
)
from jka_model.config import ProjectConfig, load_config, save_config
from jka_model.constants import ARCHITECTURE_REVISION, CHECKPOINT_SCHEMA_VERSION, PROJECT_VERSION
from jka_model.context import build_dynamic_context_model
from jka_model.context.checkpoint import load_context_checkpoint
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
class V09TrainingResult:
    run_dir: Path
    latest_checkpoint: Path
    best_checkpoint: Path
    condition_mode: str
    rank: int
    start_epoch: int
    completed_epochs: int
    validation_metrics: dict[str, float]


def _require_v0_9(config: ProjectConfig) -> None:
    required = (
        config.koopman,
        config.v0_8_context,
        config.v0_9_condition,
        config.v0_9_adaptive,
        config.v0_9_training,
        config.v0_9_evaluation,
    )
    if any(section is None for section in required):
        raise ValueError("train_v0_9 requires the complete V0.8 handoff and V0.9 contract")
    if config.training.stage is not TrainStage.ADAPTIVE:
        raise ValueError("V0.9 training.stage must be adaptive")


def _build_model(
    config: ProjectConfig,
    cache: Any,
    context_payload: dict[str, Any],
    device: torch.device,
) -> AdaptiveKoopmanModel:
    assert config.v0_8_context and config.v0_9_adaptive
    family = str(context_payload["context_family"])
    history = int(context_payload["history_length_steps"])
    dynamic = build_dynamic_context_model(
        config.v0_8_context,
        family=family,
        latent_dim=cache.latent_dim,
        parameter_dim=cache.context_parameter_dim,
        history=history,
    )
    dynamic.load_state_dict(context_payload["best_context_state"], strict=True)
    adapter = LowRankAdaptiveOperator(
        cache.nominal_generator,
        config.v0_8_context.context_dim,
        config.v0_9_adaptive,
    )
    model = AdaptiveKoopmanModel(dynamic.context_encoder, adapter).to(device)
    configure_train_stage(model, TrainStage.ADAPTIVE)
    return model


def _move(raw: dict[str, Any], device: torch.device) -> dict[str, Any]:
    tensor_names = (
        "history_z",
        "history_dts",
        "next_dt",
        "context_parameters",
        "condition",
        "target_next",
        "previous_history_z",
        "previous_history_dts",
        "previous_next_dt",
        "previous_condition",
    )
    result = dict(raw)
    for name in tensor_names:
        result[name] = raw[name].to(device=device, dtype=torch.float32)
    result["smoothness_eligible"] = raw["smoothness_eligible"].to(device=device)
    return result


def _condition(
    value: torch.Tensor,
    *,
    mode: str,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor | None:
    return (value - mean) / std if mode == "known" else None


def _loss_bundle(
    model: AdaptiveKoopmanModel,
    batch: dict[str, Any],
    residual_scale: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    config: ProjectConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert config.v0_9_adaptive and config.v0_9_training
    condition = _condition(
        batch["condition"],
        mode=config.v0_9_adaptive.condition_mode,
        mean=condition_mean,
        std=condition_std,
    )
    prediction, _, eta, delta, adapted = model(
        batch["history_z"],
        batch["history_dts"],
        batch["next_dt"],
        batch["context_parameters"],
        condition,
    )
    forecast = ((prediction - batch["target_next"]) / residual_scale).square().mean()
    burden = operator_burden(delta, model.operator_adapter.nominal_generator).square().mean()
    baseline_proxy = symmetric_abscissa_proxy(model.operator_adapter.nominal_generator).detach()
    stability = torch.relu(symmetric_abscissa_proxy(adapted) - baseline_proxy).square().mean()
    smoothness = prediction.new_zeros(())
    eligible = batch["smoothness_eligible"].bool()
    if bool(eligible.any()):
        previous_condition = _condition(
            batch["previous_condition"],
            mode=config.v0_9_adaptive.condition_mode,
            mean=condition_mean,
            std=condition_std,
        )
        with torch.no_grad():
            previous_context = model.context_encoder(
                batch["previous_history_z"],
                batch["previous_history_dts"],
                batch["previous_next_dt"],
                batch["context_parameters"],
            )
        previous_eta = model.operator_adapter.coordinates(previous_context, previous_condition)
        smoothness = (
            (eta[eligible] - previous_eta[eligible]).square().sum(dim=-1)
            / batch["next_dt"][eligible, 0].clamp_min(1e-12)
        ).mean()
    orthogonality = model.operator_adapter.orthogonality_loss()
    training = config.v0_9_training
    total = (
        forecast
        + training.lambda_operator_burden * (burden + orthogonality)
        + training.lambda_smooth * smoothness
        + training.lambda_stability * stability
    )
    return total, {
        "forecast": forecast,
        "burden": burden,
        "smoothness": smoothness,
        "stability": stability,
        "orthogonality": orthogonality,
    }


@torch.no_grad()
def _evaluate(
    model: AdaptiveKoopmanModel,
    loader: DataLoader,
    residual_scale: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    config: ProjectConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for raw in loader:
        batch = _move(raw, device)
        total, terms = _loss_bundle(
            model, batch, residual_scale, condition_mean, condition_std, config
        )
        batch_count = batch["target_next"].shape[0]
        values = {"total": total, **terms}
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_count
        count += batch_count
    return {name: value / count for name, value in totals.items()}


def train_v0_9(
    config: ProjectConfig | str | Path,
    *,
    context_checkpoint: str | Path,
    adaptive_cache: str | Path,
    run_dir: str | Path,
    device: str | torch.device | None = None,
    resume_from: str | Path | None = None,
) -> V09TrainingResult:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    _require_v0_9(resolved)
    assert resolved.v0_9_adaptive and resolved.v0_9_training and resolved.v0_8_context
    selected = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
    )
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    destination = Path(run_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("config", "metadata", "logs", "checkpoints", "evaluation", "reports"):
        (destination / name).mkdir()
    save_config(resolved, destination / "config" / "resolved_config.yaml")
    print(
        f"[V0.9][train:{resolved.v0_9_adaptive.condition_mode}] START "
        f"device={selected} rank={resolved.v0_9_adaptive.rank} "
        f"epochs={resolved.v0_9_training.epochs}",
        flush=True,
    )
    set_global_seed(
        resolved.v0_9_training.operator_initialization_seed,
        deterministic=resolved.training.deterministic,
    )
    cache = load_adaptive_cache(adaptive_cache)
    context_sha = file_sha256(context_checkpoint)
    if cache.context_checkpoint_sha256 != context_sha:
        raise ValueError("V0.9 cache/context checkpoint fingerprint mismatch")
    context_payload = load_context_checkpoint(context_checkpoint)
    model = _build_model(resolved, cache, context_payload, selected)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=resolved.v0_9_training.learning_rate,
        weight_decay=resolved.v0_9_training.weight_decay,
    )
    assert_optimizer_matches_trainable_params(model, optimizer)
    scheduler = StepLR(optimizer, step_size=max(1, resolved.v0_9_training.epochs // 2), gamma=0.5)
    amp_enabled = selected.type == "cuda" and resolved.v0_9_training.precision != "fp32"
    amp_dtype = (
        torch.float16
        if resolved.v0_9_training.precision == "amp_fp16"
        else torch.bfloat16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype is torch.float16)
    history = int(context_payload["history_length_steps"])
    datasets = {
        split: AdaptiveWindowDataset(cache, split, history)
        for split in ("train", "validation")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=resolved.v0_9_training.batch_size,
            shuffle=split == "train",
            num_workers=0,
        )
        for split, dataset in datasets.items()
    }
    residual_scale_cpu, condition_mean_cpu, condition_std_cpu = adaptive_training_scales(cache)
    residual_scale = residual_scale_cpu.to(selected)
    condition_mean = condition_mean_cpu.to(selected)
    condition_std = condition_std_cpu.to(selected)
    start_epoch = global_step = optimizer_update_step = 0
    best_score = float("inf")
    best_state: dict[str, Any] | None = None
    stale = 0
    if resume_from is not None:
        saved = load_adaptive_checkpoint(resume_from)
        if saved["config_hash"] != resolved.stable_hash:
            raise ValueError("V0.9 resume config mismatch")
        if saved["adaptive_cache_fingerprint"] != cache.fingerprint:
            raise ValueError("V0.9 resume cache mismatch")
        model.operator_adapter.load_state_dict(saved["adaptive_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        if saved["amp_scaler_state"] is not None:
            scaler.load_state_dict(saved["amp_scaler_state"])
        restore_rng_state(RNGState.from_checkpoint_dict(saved["rng_state"]))
        start_epoch = int(saved["epoch"])
        global_step = int(saved["global_step"])
        optimizer_update_step = int(saved["optimizer_update_step"])
        best_score = float(saved["best_validation_score"])
        best_state = dict(saved["best_adaptive_state"])
        stale = int(saved["epochs_without_improvement"])
    latest = destination / "checkpoints" / "latest.pt"
    best = destination / "checkpoints" / "best_scientific_gate.pt"

    def checkpoint_payload(epoch: int) -> dict[str, Any]:
        if best_state is None:
            raise RuntimeError("cannot checkpoint before validation selects a state")
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "architecture_revision": ARCHITECTURE_REVISION,
            "project_version": PROJECT_VERSION,
            "train_stage": TrainStage.ADAPTIVE.value,
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_update_step": optimizer_update_step,
            "condition_mode": resolved.v0_9_adaptive.condition_mode,
            "rank": resolved.v0_9_adaptive.rank,
            "flow_data_seed": resolved.training.seed,
            "backbone_seed": resolved.training.seed,
            "context_init_seed": int(context_payload["context_init_seed"]),
            "operator_init_seed": resolved.v0_9_training.operator_initialization_seed,
            "adaptive_state": model.operator_adapter.state_dict(),
            "best_adaptive_state": best_state,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "amp_scaler_state": scaler.state_dict() if amp_enabled else None,
            "rng_state": capture_rng_state().to_checkpoint_dict(),
            "config": resolved.to_dict(),
            "config_hash": resolved.stable_hash,
            "backbone_checkpoint_sha256": cache.backbone_checkpoint_sha256,
            "context_checkpoint_sha256": context_sha,
            "adaptive_cache_fingerprint": cache.fingerprint,
            "residual_training_scale": residual_scale_cpu,
            "condition_mean": condition_mean_cpu,
            "condition_std": condition_std_cpu,
            "best_validation_score": best_score,
            "epochs_without_improvement": stale,
            "git_commit": get_git_commit(Path.cwd()),
            "runtime": {
                "device": str(selected),
                "precision": resolved.v0_9_training.precision,
                "torch_version": torch.__version__,
            },
        }

    completed = start_epoch
    log_path = destination / "logs" / "epoch_metrics.csv"
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "epoch",
            "train_total",
            "train_forecast",
            "validation_total",
            "validation_forecast",
            "validation_burden",
            "validation_stability",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(start_epoch, resolved.v0_9_training.epochs):
            model.train()
            sums = {"total": 0.0, "forecast": 0.0}
            count = 0
            for raw in loaders["train"]:
                batch = _move(raw, selected)
                optimizer.zero_grad(set_to_none=True)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast:
                    loss, terms = _loss_bundle(
                        model,
                        batch,
                        residual_scale,
                        condition_mean,
                        condition_std,
                        resolved,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), resolved.v0_9_training.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                batch_count = batch["target_next"].shape[0]
                sums["total"] += float(loss.detach()) * batch_count
                sums["forecast"] += float(terms["forecast"].detach()) * batch_count
                count += batch_count
                global_step += 1
                optimizer_update_step += 1
            scheduler.step()
            validation = _evaluate(
                model,
                loaders["validation"],
                residual_scale,
                condition_mean,
                condition_std,
                resolved,
                selected,
            )
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_total": sums["total"] / count,
                    "train_forecast": sums["forecast"] / count,
                    "validation_total": validation["total"],
                    "validation_forecast": validation["forecast"],
                    "validation_burden": validation["burden"],
                    "validation_stability": validation["stability"],
                }
            )
            stream.flush()
            score = validation["total"]
            if score < best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.operator_adapter.state_dict().items()
                }
                stale = 0
                save_adaptive_checkpoint(checkpoint_payload(epoch + 1), best)
            else:
                stale += 1
            completed = epoch + 1
            save_adaptive_checkpoint(checkpoint_payload(completed), latest)
            if stale >= resolved.v0_9_training.patience:
                break
    if best_state is None:
        raise RuntimeError("V0.9 training produced no validation checkpoint")
    model.operator_adapter.load_state_dict(best_state, strict=True)
    validation = _evaluate(
        model,
        loaders["validation"],
        residual_scale,
        condition_mean,
        condition_std,
        resolved,
        selected,
    )
    summary = {
        "status": "PASS",
        "condition_mode": resolved.v0_9_adaptive.condition_mode,
        "rank": resolved.v0_9_adaptive.rank,
        "completed_epochs": completed,
        "validation": validation,
        "test_locked_confirmation": "NOT_OPENED_DURING_TRAINING",
        "claims": {
            "backbone_frozen": True,
            "context_frozen": True,
            "A0_frozen": True,
            "additive_residual_enabled": False,
            "persistent_z_R_present": False,
        },
    }
    (destination / "evaluation" / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[V0.9][train:{resolved.v0_9_adaptive.condition_mode}] PASS "
        f"validation_forecast={validation['forecast']:.6g} test=LOCKED",
        flush=True,
    )
    return V09TrainingResult(
        destination,
        latest,
        best,
        resolved.v0_9_adaptive.condition_mode,
        resolved.v0_9_adaptive.rank,
        start_epoch,
        completed,
        validation,
    )
