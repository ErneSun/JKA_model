"""Canonical V0.5 field-Koopman training function."""

from __future__ import annotations

import csv
import json
import platform
import shutil
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from jka_model.config import ProjectConfig, load_config, save_config
from jka_model.data import (
    ChannelStandardizer,
    TrajectoryWindowDataset,
    collate_problem_batches,
    data_fingerprint,
    make_split_manifest,
    select_split,
)
from jka_model.losses import compute_field_koopman_loss
from jka_model.models import (
    ContinuousKoopmanCore,
    FieldKoopmanAutoencoder,
    KoopmanEncoder2D,
    TrainingDecoder2D,
)
from jka_model.physics import PhysicsConstraint
from jka_model.problems import create_problem_adapter
from jka_model.training import (
    TrainStage,
    assert_optimizer_matches_trainable_params,
    configure_train_stage,
)
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


@dataclass(frozen=True, slots=True)
class V05TrainingResult:
    run_dir: Path
    latest_checkpoint: Path
    best_checkpoint: Path
    start_epoch: int
    completed_epochs: int
    global_step: int
    initial_loss: float
    final_loss: float
    gradient_norms: dict[str, float]
    evaluation: dict[str, Any]


def _require_sections(config: ProjectConfig) -> None:
    required = (
        config.koopman,
        config.field_autoencoder,
        config.field_loss,
        config.v0_5_training,
        config.v0_5_evaluation,
    )
    if any(section is None for section in required):
        raise ValueError("train_v0_5 requires every V0.5 config section")


def initialize_v0_5_model(
    config: ProjectConfig,
    *,
    device: torch.device | str,
) -> FieldKoopmanAutoencoder:
    """Initialize only E_K, continuous A, and D_train from the resolved config."""
    _require_sections(config)
    assert config.koopman and config.field_autoencoder
    assert config.v0_5_training
    architecture = config.field_autoencoder
    spec = create_problem_adapter(config).build_problem_spec()
    if spec.grid.shape is None or len(spec.grid.shape) != 2:
        raise ValueError("V0.5 field autoencoder requires a two-dimensional grid")
    nx, ny = spec.grid.shape
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.training.seed)
        encoder = KoopmanEncoder2D(
            architecture.input_channels,
            architecture.latent_dim,
            architecture.width,
            nx,
            ny,
        )
        decoder = TrainingDecoder2D(
            architecture.latent_dim,
            architecture.input_channels,
            nx,
            ny,
            architecture.width,
            architecture.decoder_hidden_dim,
        )
        for module in (encoder, decoder):
            for layer in module.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    nn.init.normal_(layer.weight, std=config.v0_5_training.init_scale)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    generator_random = torch.randn(
        architecture.latent_dim,
        architecture.latent_dim,
        generator=torch.Generator(device="cpu").manual_seed(config.training.seed + 1),
    )
    # Start from a stable continuous generator while retaining rotational modes.
    generator = config.v0_5_training.init_scale * (
        generator_random - generator_random.T - torch.eye(architecture.latent_dim)
    )
    core = ContinuousKoopmanCore(
        architecture.latent_dim, generator=generator, trainable=True, dtype=torch.float32
    )
    return FieldKoopmanAutoencoder(encoder, core, decoder).to(device=device, dtype=torch.float32)


def _batches(windows: TrajectoryWindowDataset, batch_size: int, *, shuffle: bool) -> list[Any]:
    indices = torch.randperm(len(windows)).tolist() if shuffle else list(range(len(windows)))
    return [
        collate_problem_batches([windows[index] for index in indices[start : start + batch_size]])
        for start in range(0, len(indices), batch_size)
    ]


def _mean_validation(
    model: FieldKoopmanAutoencoder,
    batches: list[Any],
    normalizer: ChannelStandardizer,
    spec: Any,
    loss_config: Any,
    constraints: dict[str, PhysicsConstraint],
    device: torch.device,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    model.eval()
    with torch.no_grad():
        for cpu_batch in batches:
            batch = cpu_batch.to(device=device, dtype=torch.float32)
            values = compute_field_koopman_loss(
                model,
                batch,
                normalizer,
                spec,
                loss_config,
                constraints,
                physics_scale=1.0,
            ).as_scalars()
            size = batch.future_dts.shape[0]
            count += size
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value * size
    return {name: value / count for name, value in totals.items()}


def _gradient_norm(module: nn.Module) -> float:
    gradients = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
    return 0.0 if not gradients else float(torch.stack(gradients).norm())


def train_v0_5(
    config: ProjectConfig | str | Path,
    *,
    device: str | torch.device | None = None,
    resume_from: str | Path | None = None,
    run_name: str | None = None,
    checkpoint_epochs: set[int] | None = None,
) -> V05TrainingResult:
    """Train the complete V0.5 model and emit a reproducible run directory."""
    training_started = datetime.now(timezone.utc).isoformat()
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    _require_sections(resolved)
    assert resolved.field_loss and resolved.v0_5_training
    selected = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
    )
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
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
    if not validation_records:
        raise ValueError("V0.5 validation split must be non-empty")
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
    model = initialize_v0_5_model(resolved, device=selected)
    configure_train_stage(model, TrainStage.KOOPMAN)
    base_parameters = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = Adam(
        [
            {"params": base_parameters, "lr": resolved.v0_5_training.learning_rate},
            {
                "params": model.core.parameters(),
                "lr": resolved.v0_5_training.learning_rate
                * resolved.v0_5_training.generator_lr_multiplier,
            },
        ],
        weight_decay=resolved.v0_5_training.weight_decay,
    )
    scheduler = StepLR(
        optimizer,
        step_size=resolved.v0_5_training.scheduler_step_size,
        gamma=resolved.v0_5_training.scheduler_gamma,
    )
    assert_optimizer_matches_trainable_params(model, optimizer)
    amp_enabled = selected.type == "cuda" and resolved.v0_5_training.precision != "fp32"
    amp_dtype = torch.float16 if resolved.v0_5_training.precision == "amp_fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype is torch.float16)
    start_epoch, global_step = 0, 0
    if resume_from is not None:
        # Load the resume envelope on CPU. Model/optimizer loaders place trainable state on
        # their target devices, while normalizer and RNG metadata must remain device-neutral.
        checkpoint = load_checkpoint(resume_from, map_location="cpu")
        if (
            checkpoint.config_hash != resolved.stable_hash
            or checkpoint.data_fingerprint != fingerprint
        ):
            raise ValueError("resume config/data fingerprint mismatch")
        if checkpoint.split_manifest != manifest.to_dict():
            raise ValueError("resume split manifest mismatch")
        if checkpoint.online_model_state is None or checkpoint.optimizer_state is None:
            raise ValueError("resume checkpoint lacks model/optimizer state")
        model.load_state_dict(checkpoint.online_model_state)
        optimizer.load_state_dict(checkpoint.optimizer_state)
        if checkpoint.scheduler_state is None:
            raise ValueError("resume checkpoint lacks scheduler state")
        scheduler.load_state_dict(checkpoint.scheduler_state)
        if checkpoint.amp_scaler_state is not None:
            scaler.load_state_dict(checkpoint.amp_scaler_state)
        if checkpoint.normalizer_state is None or not normalizer.matches_state_dict(
            checkpoint.normalizer_state
        ):
            raise ValueError("resume normalizer state mismatch")
        if checkpoint.rng_state is None:
            raise ValueError("resume checkpoint lacks RNG state")
        restore_rng_state(checkpoint.rng_state)
        start_epoch, global_step = checkpoint.epoch, checkpoint.global_step

    backend = "gpu" if selected.type == "cuda" else "cpu"
    root = Path(resolved.training.run_root) / "v0_5" / backend
    run = create_run_directory(
        root,
        seed=resolved.training.seed,
        config_hash=resolved.stable_hash,
        train_stage=TrainStage.KOOPMAN,
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
    metadata = {
        **run.to_dict(),
        "device": str(selected),
        "precision": resolved.v0_5_training.precision,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "data_fingerprint": fingerprint,
        "gpu_validation": "NOT_RUN" if selected.type != "cuda" else "RUNNING",
        "start_time": training_started,
    }
    (run.run_dir / "metadata" / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (run.run_dir / "metadata" / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    environment = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": str(selected),
        "precision": resolved.v0_5_training.precision,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
    }
    if selected.type == "cuda":
        properties = torch.cuda.get_device_properties(selected)
        environment.update(
            {"gpu_model": properties.name, "gpu_total_memory": properties.total_memory}
        )
        metadata.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu_model": properties.name,
                "gpu_count": torch.cuda.device_count(),
                "gpu_total_memory": properties.total_memory,
                "amp_mode": resolved.v0_5_training.precision,
            }
        )
    (run.run_dir / "metadata" / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )
    branch = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        ).stdout.strip()
    )
    (run.run_dir / "metadata" / "git_state.json").write_text(
        json.dumps(
            {"commit": run.git_commit, "branch": branch, "dirty": dirty}, indent=2, sort_keys=True
        )
        + "\n"
    )
    metadata.update(
        {
            "git_branch": branch,
            "git_dirty": dirty,
            "split_manifest": manifest.to_dict(),
            "environment": environment,
        }
    )
    for filename in ("metadata.json", "run_manifest.json"):
        (run.run_dir / "metadata" / filename).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    (run.run_dir / "metadata" / "data_manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "split": manifest.to_dict(),
                "problem_spec": spec.to_dict(),
                "trajectory_count": len(records),
                "train_count": len(train_records),
                "validation_count": len(validation_records),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (run.run_dir / "metadata" / "model_summary.txt").write_text(
        f"{model}\n\nparameters={metadata['parameter_count']}\ntrainable={metadata['trainable_parameter_count']}\n"
    )
    history_path = run.run_dir / "logs" / "epoch_metrics.csv"
    columns = [
        "epoch",
        "global_step",
        "learning_rate",
        "generator_learning_rate",
        "L_total",
        "L_K",
        "L_generator",
        "L_multi",
        "L_rec",
        "L_forecast",
        "L_var",
        "L_stability",
        "L_physics",
        "L_mass",
        "L_operator",
        "val_rollout",
        "val_reconstruction",
        "val_mass",
        "val_operator",
        "epoch_time",
        "samples_per_sec",
        "peak_gpu_memory",
    ]
    prior_rows: list[dict[str, str]] = []
    prior_step_lines: list[str] = []
    if resume_from is not None:
        prior_history = Path(resume_from).resolve().parents[1] / "logs" / "epoch_metrics.csv"
        if prior_history.is_file():
            with prior_history.open(encoding="utf-8") as prior_stream:
                prior_rows = [
                    row for row in csv.DictReader(prior_stream) if int(row["epoch"]) <= start_epoch
                ]
        prior_steps = Path(resume_from).resolve().parents[1] / "logs" / "step_metrics.jsonl"
        if prior_steps.is_file():
            prior_step_lines = [
                line
                for line in prior_steps.read_text(encoding="utf-8").splitlines()
                if int(json.loads(line)["epoch"]) <= start_epoch
            ]
    fixed_validation = _batches(
        validation_windows, resolved.v0_5_training.batch_size, shuffle=False
    )
    initial = _mean_validation(
        model,
        fixed_validation,
        normalizer,
        spec,
        resolved.field_loss,
        constraints,
        selected,
    )
    initial_loss = initial["total_loss"]
    best_value = min((float(row["val_rollout"]) for row in prior_rows), default=float("inf"))
    latest_path = run.run_dir / "checkpoints" / "latest.pt"
    best_path = run.run_dir / "checkpoints" / "best_forecast.pt"
    best_physics_path = run.run_dir / "checkpoints" / "best_physics.pt"
    best_post_warmup_path = run.run_dir / "checkpoints" / "best_forecast_post_warmup.pt"
    best_physics_post_warmup_path = (
        run.run_dir / "checkpoints" / "best_physics_post_warmup.pt"
    )
    field_loss_config = resolved.field_loss
    assert field_loss_config is not None

    def physics_selection_value(row: dict[str, Any]) -> float:
        return (
            field_loss_config.lambda_mass * float(row["val_mass"])
            + field_loss_config.lambda_operator * float(row["val_operator"])
        )

    best_physics_value = min(
        (physics_selection_value(row) for row in prior_rows),
        default=float("inf"),
    )
    first_full_physics_epoch = resolved.v0_5_training.physics_warmup_epochs + 1
    post_warmup_rows = [
        row for row in prior_rows if int(row["epoch"]) >= first_full_physics_epoch
    ]
    best_post_warmup_value = min(
        (float(row["val_rollout"]) for row in post_warmup_rows),
        default=float("inf"),
    )
    best_physics_post_warmup_value = min(
        (physics_selection_value(row) for row in post_warmup_rows),
        default=float("inf"),
    )
    if resume_from is not None:
        source_checkpoints = Path(resume_from).resolve().parent
        for name, destination in (
            ("best_forecast.pt", best_path),
            ("best_physics.pt", best_physics_path),
            ("best_forecast_post_warmup.pt", best_post_warmup_path),
            ("best_physics_post_warmup.pt", best_physics_post_warmup_path),
        ):
            source = source_checkpoints / name
            if source.is_file():
                shutil.copy2(source, destination)
    last_gradient_norms = {"encoder": 0.0, "decoder": 0.0, "generator": 0.0}
    step_path = run.run_dir / "logs" / "step_metrics.jsonl"
    with (
        history_path.open("w", newline="", encoding="utf-8") as stream,
        step_path.open("w", encoding="utf-8") as step_stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(prior_rows)
        for line in prior_step_lines:
            step_stream.write(line + "\n")
        for epoch in range(start_epoch, resolved.v0_5_training.epochs):
            started = time.perf_counter()
            if selected.type == "cuda":
                torch.cuda.reset_peak_memory_stats(selected)
            model.train()
            seen = 0
            train_sums: dict[str, float] = {}
            scale = (
                1.0
                if resolved.v0_5_training.physics_warmup_epochs == 0
                else min(1.0, max(0.0, epoch / resolved.v0_5_training.physics_warmup_epochs))
            )
            for cpu_batch in _batches(
                train_windows, resolved.v0_5_training.batch_size, shuffle=True
            ):
                batch = cpu_batch.to(device=selected, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                context = torch.autocast("cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
                with context:
                    losses = compute_field_koopman_loss(
                        model,
                        batch,
                        normalizer,
                        spec,
                        resolved.field_loss,
                        constraints,
                        physics_scale=scale,
                    )
                if not torch.isfinite(losses.total):
                    raise FloatingPointError("V0.5 loss became non-finite")
                scaler.scale(losses.total).backward()
                scaler.unscale_(optimizer)
                gradients = [
                    p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
                ]
                if not gradients or any(not torch.isfinite(value).all() for value in gradients):
                    raise FloatingPointError("V0.5 gradients are missing or non-finite")
                scaler.step(optimizer)
                scaler.update()
                if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise FloatingPointError("V0.5 parameters became non-finite")
                global_step += 1
                size = batch.future_dts.shape[0]
                seen += size
                for name, value in losses.as_scalars().items():
                    train_sums[name] = train_sums.get(name, 0.0) + value * size
                step_stream.write(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "generator_learning_rate": optimizer.param_groups[1]["lr"],
                            **losses.as_scalars(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                step_stream.flush()
            scheduler.step()
            validation = _mean_validation(
                model,
                fixed_validation,
                normalizer,
                spec,
                resolved.field_loss,
                constraints,
                selected,
            )
            elapsed = time.perf_counter() - started
            train_mean = {name: value / seen for name, value in train_sums.items()}
            row = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "generator_learning_rate": optimizer.param_groups[1]["lr"],
                "L_total": train_mean["total_loss"],
                "L_K": train_mean["koopman_one_step_loss"],
                "L_generator": train_mean["generator_consistency_loss"],
                "L_multi": train_mean["koopman_multi_step_loss"],
                "L_rec": train_mean["reconstruction_loss"],
                "L_forecast": train_mean["forecast_model_mse"],
                "L_var": train_mean["variance_loss"],
                "L_stability": train_mean["stability_loss"],
                "L_physics": scale
                * resolved.field_loss.lambda_physics
                * (
                    resolved.field_loss.lambda_mass * train_mean["mass_loss"]
                    + resolved.field_loss.lambda_operator * train_mean["operator_loss"]
                ),
                "L_mass": train_mean["mass_loss"],
                "L_operator": train_mean["operator_loss"],
                "val_rollout": validation["forecast_model_mse"],
                "val_reconstruction": validation["reconstruction_loss"],
                "val_mass": validation["mass_loss"],
                "val_operator": validation["operator_loss"],
                "epoch_time": elapsed,
                "samples_per_sec": seen / elapsed,
                "peak_gpu_memory": torch.cuda.max_memory_allocated(selected)
                if selected.type == "cuda"
                else 0,
            }
            writer.writerow(row)
            stream.flush()
            last_gradient_norms = {
                "encoder": _gradient_norm(model.encoder),
                "decoder": _gradient_norm(model.decoder),
                "generator": 0.0
                if model.core.A.grad is None
                else float(model.core.A.grad.detach().norm()),
            }
            checkpoint = Checkpoint(
                train_stage=TrainStage.KOOPMAN,
                epoch=epoch + 1,
                global_step=global_step,
                online_model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                amp_scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
                rng_state=capture_rng_state(),
                normalizer_state=normalizer.state_dict(),
                problem_spec=spec,
                config=resolved,
                data_fingerprint=fingerprint,
                split_manifest=manifest.to_dict(),
                physics_constraint_spec=[
                    f"{role}:{type(item).__name__}" for role, item in constraints.items()
                ],
                git_commit=run.git_commit,
            )
            save_checkpoint(checkpoint, latest_path)
            save_checkpoint(checkpoint, run.run_dir / "checkpoints" / "last.pt")
            completed_epoch = epoch + 1
            if checkpoint_epochs is None or completed_epoch in checkpoint_epochs:
                save_checkpoint(
                    checkpoint,
                    run.run_dir / "checkpoints" / f"epoch_{completed_epoch:04d}.pt",
                )
            if validation["forecast_model_mse"] < best_value:
                best_value = validation["forecast_model_mse"]
                save_checkpoint(checkpoint, best_path)
            physics_value = physics_selection_value(
                {
                    "val_mass": validation["mass_loss"],
                    "val_operator": validation["operator_loss"],
                }
            )
            if physics_value < best_physics_value:
                best_physics_value = physics_value
                save_checkpoint(checkpoint, best_physics_path)
            if scale >= 1.0:
                if validation["forecast_model_mse"] < best_post_warmup_value:
                    best_post_warmup_value = validation["forecast_model_mse"]
                    save_checkpoint(checkpoint, best_post_warmup_path)
                if physics_value < best_physics_post_warmup_value:
                    best_physics_post_warmup_value = physics_value
                    save_checkpoint(checkpoint, best_physics_post_warmup_path)
    if start_epoch >= resolved.v0_5_training.epochs:
        raise ValueError("resume checkpoint already reached configured epochs")
    if not best_path.exists():
        # A resumed run can inherit a better pre-resume metric without inheriting files.
        # Keep the run self-contained by retaining its final checkpoint as evaluation fallback.
        save_checkpoint(checkpoint, best_path)
    from eval.evaluate_v0_5 import evaluate_v0_5

    evaluation = evaluate_v0_5(resolved, checkpoint=best_path, device=selected, run_dir=run.run_dir)
    final_loss = float(list(csv.DictReader(history_path.open(encoding="utf-8")))[-1]["L_total"])
    metadata["end_time"] = datetime.now(timezone.utc).isoformat()
    metadata["global_step"] = global_step
    metadata["completed_epochs"] = resolved.v0_5_training.epochs
    if selected.type == "cuda":
        metadata["peak_gpu_memory"] = torch.cuda.max_memory_allocated(selected)
        metadata["gpu_validation"] = "MEASURED_NOT_REVIEWED"
    (run.run_dir / "metadata" / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (run.run_dir / "reports" / "training_record.md").write_text(
        "# V0.5 training record\n\n"
        f"- run: `{run.run_id}`\n- device: `{selected}`\n"
        f"- precision: `{resolved.v0_5_training.precision}`\n"
        f"- epochs: {resolved.v0_5_training.epochs}\n- global step: {global_step}\n"
        f"- initial validation loss: {initial_loss:.8g}\n- final training loss: {final_loss:.8g}\n"
    )
    return V05TrainingResult(
        run.run_dir,
        latest_path,
        best_path,
        start_epoch,
        resolved.v0_5_training.epochs,
        global_step,
        initial_loss,
        final_loss,
        last_gradient_norms,
        evaluation,
    )
