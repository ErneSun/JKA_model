"""Canonical V0.6 JEPA-over-physics-constrained-Koopman training."""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from jka_model.config import ProjectConfig, load_config, save_config, stable_config_hash
from jka_model.constants import (
    ARCHITECTURE_REVISION,
    CHECKPOINT_SCHEMA_VERSION,
    PROJECT_VERSION,
    V0_5_CHECKPOINT_SCHEMA_VERSION,
    V0_5_PROJECT_VERSION,
)
from jka_model.data import (
    ChannelStandardizer,
    TrajectoryWindowDataset,
    collate_problem_batches,
    data_fingerprint,
    make_split_manifest,
    select_split,
)
from jka_model.evaluation import model_tracking_diagnostics, near_identity_diagnostic
from jka_model.losses import compute_field_jepa_loss
from jka_model.models import FieldJEPAKoopmanModel
from jka_model.problems import create_problem_adapter
from jka_model.training import (
    TrainStage,
    assert_optimizer_matches_trainable_params,
    configure_train_stage,
)
from jka_model.training.ema import EMATracker
from jka_model.utils import (
    Checkpoint,
    capture_rng_state,
    create_run_directory,
    get_git_commit,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    set_global_seed,
)
from train.train_v0_5 import initialize_v0_5_model


@dataclass(frozen=True, slots=True)
class V06TrainingResult:
    run_dir: Path
    latest_checkpoint: Path
    best_checkpoint: Path
    start_epoch: int
    completed_epochs: int
    global_step: int
    optimizer_update_step: int
    initial_loss: float
    final_loss: float
    evaluation: dict[str, Any]


def _require_v0_6(config: ProjectConfig) -> None:
    required = (
        config.koopman,
        config.field_autoencoder,
        config.field_loss,
        config.v0_5_training,
        config.v0_5_evaluation,
        config.jepa_loss,
        config.ema,
        config.v0_6_evaluation,
    )
    if any(section is None for section in required):
        raise ValueError("train_v0_6 requires complete V0.5 inheritance and V0.6 sections")


def initialize_v0_6_model(
    config: ProjectConfig, *, device: str | torch.device
) -> FieldJEPAKoopmanModel:
    """Build the V0.5 online model and a hard-synchronized frozen target encoder."""
    _require_v0_6(config)
    base = initialize_v0_5_model(config, device=device)
    model = FieldJEPAKoopmanModel(base.encoder, base.core, base.decoder).to(device)
    configure_train_stage(model, TrainStage.JEPA)
    return model


def _torch_load_mapping(path: str | Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return payload


def _v0_5_contract(config: ProjectConfig | Mapping[str, Any]) -> dict[str, Any]:
    payload = config.to_dict() if isinstance(config, ProjectConfig) else dict(config)
    names = (
        "architecture",
        "data",
        "koopman",
        "advection_diffusion_2d",
        "field_autoencoder",
        "field_loss",
        "v0_5_evaluation",
    )
    return {name: payload.get(name) for name in names}


def load_v0_5_initialization(path: str | Path, config: ProjectConfig) -> Mapping[str, Any]:
    """Read a trusted validated V0.5 checkpoint without weakening V0.6 resume guards."""
    payload = _torch_load_mapping(path)
    version_pair = (
        int(payload.get("schema_version", -1)),
        str(payload.get("project_version")),
    )
    allowed = {
        (V0_5_CHECKPOINT_SCHEMA_VERSION, V0_5_PROJECT_VERSION),
        (CHECKPOINT_SCHEMA_VERSION, PROJECT_VERSION),
    }
    if version_pair not in allowed:
        raise ValueError("V0.6 initialization requires a supported V0.5 Koopman checkpoint")
    if str(payload.get("architecture_revision")) != ARCHITECTURE_REVISION:
        raise ValueError("V0.5 initialization architecture revision mismatch")
    if str(payload.get("train_stage")) != TrainStage.KOOPMAN.value:
        raise ValueError("V0.6 initialization checkpoint must be a V0.5 Koopman stage")
    source_config = payload.get("config")
    if not isinstance(source_config, Mapping):
        raise ValueError("V0.5 initialization checkpoint lacks resolved config")
    source_resolved = ProjectConfig.from_dict(source_config)
    expected_hash = (
        stable_config_hash(source_config)
        if version_pair[0] == V0_5_CHECKPOINT_SCHEMA_VERSION
        else source_resolved.stable_hash
    )
    if payload.get("config_hash") != expected_hash:
        raise ValueError("V0.5 initialization checkpoint config hash is inconsistent")
    if _v0_5_contract(config) != _v0_5_contract(source_config):
        raise ValueError("V0.5 initialization scientific/model contract mismatch")
    if payload.get("online_model_state") is None:
        raise ValueError("V0.5 initialization checkpoint lacks online model state")
    return payload


def _batches(dataset: TrajectoryWindowDataset, batch_size: int, shuffle: bool) -> list[Any]:
    indices = torch.randperm(len(dataset)).tolist() if shuffle else list(range(len(dataset)))
    return [
        collate_problem_batches([dataset[index] for index in indices[start : start + batch_size]])
        for start in range(0, len(indices), batch_size)
    ]


def _validation(
    model: FieldJEPAKoopmanModel,
    batches: list[Any],
    normalizer: ChannelStandardizer,
    spec: Any,
    config: ProjectConfig,
    constraints: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float]:
    assert config.field_loss and config.jepa_loss
    totals: dict[str, float] = {}
    count = 0
    model.eval()
    with torch.no_grad():
        for cpu_batch in batches:
            batch = cpu_batch.to(device=device, dtype=torch.float32)
            values = compute_field_jepa_loss(
                model,
                batch,
                normalizer,
                spec,
                config.field_loss,
                config.jepa_loss,
                constraints,
                physics_scale=1.0,
            ).as_scalars()
            size = batch.future_dts.shape[0]
            count += size
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + size * value
    return {name: value / count for name, value in totals.items()}


def _optimizer(model: FieldJEPAKoopmanModel, config: ProjectConfig) -> Adam:
    assert config.v0_5_training
    training = config.v0_5_training
    return Adam(
        [
            {
                "params": list(model.online_encoder.parameters())
                + list(model.training_decoder.parameters()),
                "lr": training.learning_rate,
            },
            {
                "params": model.koopman_core.parameters(),
                "lr": training.learning_rate * training.generator_lr_multiplier,
            },
        ],
        weight_decay=training.weight_decay,
    )


def update_ema_after_optimizer_result(
    model: FieldJEPAKoopmanModel,
    tracker: EMATracker,
    *,
    optimizer_updated: bool,
) -> float | None:
    """Centralize the AMP contract: a skipped optimizer update cannot advance EMA."""
    if not optimizer_updated:
        return tracker.current_tau
    return tracker.update_after_optimizer(model)


def train_v0_6(
    config: ProjectConfig | str | Path,
    *,
    device: str | torch.device | None = None,
    init_from_v0_5: str | Path | None = None,
    resume_from: str | Path | None = None,
    run_name: str | None = None,
) -> V06TrainingResult:
    """Train V0.6; initialization and exact resume are intentionally disjoint paths."""
    if init_from_v0_5 is not None and resume_from is not None:
        raise ValueError("init_from_v0_5 and resume_from are mutually exclusive")
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    _require_v0_6(resolved)
    assert resolved.v0_5_training and resolved.field_loss and resolved.jepa_loss and resolved.ema
    selected = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
    )
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_global_seed(resolved.training.seed, deterministic=resolved.training.deterministic)
    adapter = create_problem_adapter(resolved)
    records = adapter.build_dataset(seed=resolved.training.seed)
    spec = adapter.build_problem_spec()
    constraints = dict(adapter.build_physics_constraints())
    manifest = make_split_manifest(records, resolved.data.split)
    fingerprint = data_fingerprint(records, spec)
    normalizer = ChannelStandardizer(eps=resolved.data.normalization.eps).fit(
        records, manifest, spec
    )
    train_records = select_split(records, manifest, "train")
    validation_records = select_split(records, manifest, "validation")
    if not train_records or not validation_records:
        raise ValueError("V0.6 train and validation splits must both be non-empty")
    train_windows = TrajectoryWindowDataset(
        train_records,
        history=resolved.data.history,
        horizon=resolved.data.horizon,
        normalizer=normalizer,
    )
    validation_windows = TrajectoryWindowDataset(
        validation_records,
        history=resolved.data.history,
        horizon=resolved.data.horizon,
        normalizer=normalizer,
    )
    model = initialize_v0_6_model(resolved, device=selected)
    optimizer = _optimizer(model, resolved)
    scheduler = StepLR(
        optimizer,
        step_size=resolved.v0_5_training.scheduler_step_size,
        gamma=resolved.v0_5_training.scheduler_gamma,
    )
    assert_optimizer_matches_trainable_params(model, optimizer)
    amp_enabled = selected.type == "cuda" and resolved.v0_5_training.precision != "fp32"
    amp_dtype = torch.float16 if resolved.v0_5_training.precision == "amp_fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype is torch.float16)
    updates_per_epoch = math.ceil(len(train_windows) / resolved.v0_5_training.batch_size)
    ema = EMATracker(resolved.ema, updates_per_epoch * resolved.v0_5_training.epochs)
    start_epoch = global_step = optimizer_update_step = 0
    init_label = "SCRATCH_UNVALIDATED"
    if init_from_v0_5 is not None:
        source = load_v0_5_initialization(init_from_v0_5, resolved)
        if source.get("data_fingerprint") != fingerprint:
            raise ValueError("V0.5 initialization data fingerprint mismatch")
        if source.get("split_manifest") != manifest.to_dict():
            raise ValueError("V0.5 initialization split manifest mismatch")
        if source.get("normalizer_state") is None or not normalizer.matches_state_dict(
            source["normalizer_state"]
        ):
            raise ValueError("V0.5 initialization normalizer mismatch")
        normalizer.load_state_dict(source["normalizer_state"])
        model.load_online_state_dict(source["online_model_state"])
        model.hard_sync_target()
        init_label = str(Path(init_from_v0_5).resolve())
    elif resume_from is not None:
        saved = load_checkpoint(resume_from, map_location="cpu")
        if saved.train_stage is not TrainStage.JEPA:
            raise ValueError("V0.6 resume checkpoint must have JEPA train stage")
        if saved.config_hash != resolved.stable_hash or saved.data_fingerprint != fingerprint:
            raise ValueError("V0.6 resume config/data mismatch")
        if saved.split_manifest != manifest.to_dict():
            raise ValueError("V0.6 resume split mismatch")
        if any(
            value is None
            for value in (
                saved.online_model_state,
                saved.target_model_state,
                saved.optimizer_state,
                saved.scheduler_state,
                saved.ema_state,
                saved.rng_state,
            )
        ):
            raise ValueError("V0.6 resume checkpoint is incomplete")
        model.load_online_state_dict(saved.online_model_state)  # type: ignore[arg-type]
        model.target_encoder.load_state_dict(saved.target_model_state, strict=True)  # type: ignore[arg-type]
        model.target_encoder.requires_grad_(False)
        optimizer.load_state_dict(saved.optimizer_state)  # type: ignore[arg-type]
        scheduler.load_state_dict(saved.scheduler_state)  # type: ignore[arg-type]
        if saved.amp_scaler_state is not None:
            scaler.load_state_dict(saved.amp_scaler_state)
        ema = EMATracker.from_state_dict(saved.ema_state)  # type: ignore[arg-type]
        if (
            ema.config != resolved.ema
            or ema.total_updates != updates_per_epoch * resolved.v0_5_training.epochs
        ):
            raise ValueError("V0.6 resume EMA schedule mismatch")
        if saved.optimizer_update_step != ema.update_count:
            raise ValueError("V0.6 resume optimizer/EMA update counts disagree")
        if saved.normalizer_state is None or not normalizer.matches_state_dict(
            saved.normalizer_state
        ):
            raise ValueError("V0.6 resume normalizer mismatch")
        normalizer.load_state_dict(saved.normalizer_state)
        restore_rng_state(saved.rng_state)  # type: ignore[arg-type]
        start_epoch, global_step = saved.epoch, saved.global_step
        optimizer_update_step = saved.optimizer_update_step
        init_label = f"RESUME:{Path(resume_from).resolve()}"

    backend = "gpu" if selected.type == "cuda" else "cpu"
    run = create_run_directory(
        Path(resolved.training.run_root) / "v0_6" / backend,
        seed=resolved.training.seed,
        config_hash=resolved.stable_hash,
        train_stage=TrainStage.JEPA,
        git_commit=get_git_commit(Path.cwd()),
        run_id=run_name,
    )
    for name in (
        "config",
        "metadata",
        "logs",
        "checkpoints",
        "evaluation",
        "plots",
        "reports",
        "profiler",
    ):
        (run.run_dir / name).mkdir(exist_ok=True)
    save_config(resolved, run.run_dir / "config" / "resolved_config.yaml")
    manifest.save(run.run_dir / "metadata" / "split_manifest.json")
    branch = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True, check=False
    ).stdout.strip()
    metadata = {
        **run.to_dict(),
        "project_phase": "v0.6",
        "device": str(selected),
        "precision": resolved.v0_5_training.precision,
        "v0_5_initialization": init_label,
        "data_fingerprint": fingerprint,
        "split_manifest": manifest.to_dict(),
        "parameter_count_total": sum(p.numel() for p in model.parameters()),
        "parameter_count_inference": sum(
            p.numel()
            for module in (model.online_encoder, model.koopman_core, model.training_decoder)
            for p in module.parameters()
        ),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "target_in_optimizer": any(
            id(p) in {id(q) for group in optimizer.param_groups for q in group["params"]}
            for p in model.target_encoder.parameters()
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_branch": branch,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "gpu_validation": "NOT_RUN" if selected.type != "cuda" else "RUNNING",
    }
    (run.run_dir / "metadata" / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fixed_validation = _batches(
        validation_windows, resolved.v0_5_training.batch_size, shuffle=False
    )
    initial = _validation(
        model, fixed_validation, normalizer, spec, resolved, constraints, selected
    )
    initial_loss = initial["total_loss"]
    history_path = run.run_dir / "logs" / "epoch_metrics.csv"
    ema_path = run.run_dir / "logs" / "ema_metrics.csv"
    step_path = run.run_dir / "logs" / "step_metrics.jsonl"
    columns = [
        "epoch",
        "global_step",
        "optimizer_update_step",
        "L_total",
        "L_v0_5",
        "L_K",
        "L_generator",
        "L_multi",
        "L_rec",
        "L_forecast",
        "L_var",
        "L_stability",
        "L_mass",
        "L_operator",
        "L_JEPA_one",
        "L_JEPA_multi",
        "val_rollout",
        "val_jepa",
        "tau",
        "epoch_time",
        "samples_per_sec",
        "peak_gpu_memory",
    ]
    latest_path = run.run_dir / "checkpoints" / "latest.pt"
    best_path = run.run_dir / "checkpoints" / "best_forecast_post_warmup.pt"
    best_value = float("inf")
    final_loss = initial_loss
    with (
        history_path.open("w", newline="", encoding="utf-8") as history_stream,
        ema_path.open("w", newline="", encoding="utf-8") as ema_stream,
        step_path.open("w", encoding="utf-8") as step_stream,
    ):
        writer = csv.DictWriter(history_stream, fieldnames=columns)
        writer.writeheader()
        ema_columns = [
            "optimizer_update_step",
            "tau",
            "parameter_distance",
            "latent_distance",
            "online_min_std",
            "target_min_std",
        ]
        ema_writer = csv.DictWriter(ema_stream, fieldnames=ema_columns)
        ema_writer.writeheader()
        for epoch in range(start_epoch, resolved.v0_5_training.epochs):
            started = time.perf_counter()
            if selected.type == "cuda":
                torch.cuda.reset_peak_memory_stats(selected)
            model.train()
            sums: dict[str, float] = {}
            seen = 0
            scale = (
                1.0
                if resolved.v0_5_training.physics_warmup_epochs == 0
                else min(1.0, epoch / resolved.v0_5_training.physics_warmup_epochs)
            )
            last_tau = ema.current_tau
            for cpu_batch in _batches(train_windows, resolved.v0_5_training.batch_size, True):
                batch = cpu_batch.to(device=selected, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                context = torch.autocast("cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
                with context:
                    losses = compute_field_jepa_loss(
                        model,
                        batch,
                        normalizer,
                        spec,
                        resolved.field_loss,
                        resolved.jepa_loss,
                        constraints,
                        physics_scale=scale,
                    )
                if not torch.isfinite(losses.total):
                    raise FloatingPointError("V0.6 loss became non-finite")
                scaler.scale(losses.total).backward()
                scaler.unscale_(optimizer)
                if not scaler.is_enabled() and any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.requires_grad
                ):
                    raise FloatingPointError("V0.6 gradients became non-finite")
                old_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
                optimizer_updated = not scaler.is_enabled() or new_scale >= old_scale
                global_step += 1
                if optimizer_updated:
                    optimizer_update_step += 1
                    last_tau = update_ema_after_optimizer_result(model, ema, optimizer_updated=True)
                if optimizer_update_step != ema.update_count:
                    raise RuntimeError("optimizer and EMA update counts diverged")
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("V0.6 parameters became non-finite")
                values = losses.as_scalars()
                size = batch.future_dts.shape[0]
                seen += size
                for name, value in values.items():
                    sums[name] = sums.get(name, 0.0) + value * size
                step_stream.write(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "optimizer_update_step": optimizer_update_step,
                            "optimizer_updated": optimizer_updated,
                            "tau": last_tau,
                            **values,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            scheduler.step()
            validation = _validation(
                model, fixed_validation, normalizer, spec, resolved, constraints, selected
            )
            sample = fixed_validation[0].to(device=selected, dtype=torch.float32)
            tracking = model_tracking_diagnostics(model, sample.future_states_model)
            ema_writer.writerow(
                {
                    "optimizer_update_step": optimizer_update_step,
                    "tau": last_tau,
                    "parameter_distance": tracking["parameter_distance"],
                    "latent_distance": tracking["latent_distance"],
                    "online_min_std": tracking["online"]["min_dimension_std"],
                    "target_min_std": tracking["target"]["min_dimension_std"],
                }
            )
            means = {name: value / seen for name, value in sums.items()}
            elapsed = time.perf_counter() - started
            row = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "optimizer_update_step": optimizer_update_step,
                "L_total": means["total_loss"],
                "L_v0_5": means["v0_5_total_loss"],
                "L_K": means["koopman_one_step_loss"],
                "L_generator": means["generator_consistency_loss"],
                "L_multi": means["koopman_multi_step_loss"],
                "L_rec": means["reconstruction_loss"],
                "L_forecast": means["forecast_model_mse"],
                "L_var": means["variance_loss"],
                "L_stability": means["stability_loss"],
                "L_mass": means["mass_loss"],
                "L_operator": means["operator_loss"],
                "L_JEPA_one": means["jepa_one_step_loss"],
                "L_JEPA_multi": means["jepa_multi_step_loss"],
                "val_rollout": validation["forecast_model_mse"],
                "val_jepa": validation["jepa_one_step_loss"] + validation["jepa_multi_step_loss"],
                "tau": last_tau,
                "epoch_time": elapsed,
                "samples_per_sec": seen / elapsed,
                "peak_gpu_memory": torch.cuda.max_memory_allocated(selected)
                if selected.type == "cuda"
                else 0,
            }
            writer.writerow(row)
            for stream in (history_stream, ema_stream, step_stream):
                stream.flush()
            final_loss = float(row["L_total"])
            checkpoint = Checkpoint(
                train_stage=TrainStage.JEPA,
                epoch=epoch + 1,
                global_step=global_step,
                optimizer_update_step=optimizer_update_step,
                online_model_state=model.online_state_dict(),
                target_model_state=model.target_encoder.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                amp_scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
                ema_state=ema.state_dict(),
                rng_state=capture_rng_state(),
                normalizer_state=normalizer.state_dict(),
                problem_spec=spec,
                config=resolved,
                data_fingerprint=fingerprint,
                split_manifest=manifest.to_dict(),
                physics_constraint_spec=[
                    f"{key}:{type(value).__name__}" for key, value in constraints.items()
                ],
                git_commit=run.git_commit,
            )
            save_checkpoint(checkpoint, latest_path)
            save_checkpoint(checkpoint, run.run_dir / "checkpoints" / "last.pt")
            if scale >= 1.0 and validation["forecast_model_mse"] < best_value:
                best_value = validation["forecast_model_mse"]
                save_checkpoint(checkpoint, best_path)
    if start_epoch >= resolved.v0_5_training.epochs:
        raise ValueError("resume checkpoint already reached configured epochs")
    if not best_path.exists():
        save_checkpoint(checkpoint, best_path)
    from eval.evaluate_v0_6 import evaluate_v0_6

    evaluation = evaluate_v0_6(resolved, checkpoint=best_path, device=selected, run_dir=run.run_dir)
    all_dts = torch.cat([record.dts for record in records])
    metadata.update(
        {
            "end_time": datetime.now(timezone.utc).isoformat(),
            "global_step": global_step,
            "optimizer_update_step": optimizer_update_step,
            "ema_update_count": ema.update_count,
            "near_identity": near_identity_diagnostic(model.koopman_core.A.detach(), all_dts),
            "gpu_validation": "MEASURED_NOT_REVIEWED" if selected.type == "cuda" else "NOT_RUN",
        }
    )
    (run.run_dir / "metadata" / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run.run_dir / "reports" / "training_record.md").write_text(
        "# V0.6 training record\n\n"
        f"- V0.5 initialization: `{init_label}`\n"
        f"- epochs: {resolved.v0_5_training.epochs}\n"
        f"- optimizer/EMA updates: {optimizer_update_step}/{ema.update_count}\n"
        f"- initial/final loss: {initial_loss:.8g} / {final_loss:.8g}\n"
        "- scientific acceptance: `PENDING_GPU`\n",
        encoding="utf-8",
    )
    return V06TrainingResult(
        run.run_dir,
        latest_path,
        best_path,
        start_epoch,
        resolved.v0_5_training.epochs,
        global_step,
        optimizer_update_step,
        initial_loss,
        final_loss,
        evaluation,
    )
