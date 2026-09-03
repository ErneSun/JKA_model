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
from torch.utils.data import DataLoader

from jka_model.adaptive import (
    OBSERVER_VARIANTS,
    AdaptiveKoopmanModel,
    FactorizedAdaptiveOperator,
    adaptive_training_scales,
    causal_observer_features,
    classify_observer_admission,
    condition_observer_metrics,
    load_adaptive_cache,
    load_adaptive_checkpoint,
    observer_history_variant,
    phase2_condition_scales,
    symmetric_abscissa_proxy,
)
from jka_model.config import ProjectConfig, load_config, save_config
from jka_model.context import build_dynamic_context_model
from jka_model.context.checkpoint import load_context_checkpoint
from jka_model.data import ChannelStandardizer, load_cylinder_wake_dataset
from jka_model.manifold import (
    MatureCheckpointTracker,
    RawFieldAdaptiveRolloutDataset,
    assert_online_reencoding_required,
    centered_linear_cka,
    configure_phase3_route,
    dynamical_gauge_metrics,
    joint_markov_objective,
    move_raw_field_batch,
    orthogonal_procrustes_nrmse,
    phase3_checkpoint_key,
    representation_effective_rank,
)
from jka_model.models import FieldJEPAKoopmanModel
from jka_model.residual.cache import file_sha256
from jka_model.training.ema import EMATracker
from jka_model.utils import get_git_commit, load_checkpoint, set_global_seed


@dataclass(frozen=True, slots=True)
class Phase3JointTrainingResult:
    route: str
    run_dir: Path
    checkpoint: Path
    completed_epochs: int
    best_epoch: int
    trainable_parameters: int
    validation_metrics: dict[str, float]
    locked_test_metrics: dict[str, float]
    observer_admission: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Phase3FrozenEvaluationResult:
    route: str
    run_dir: Path
    trainable_parameters: int
    locked_test_metrics: dict[str, float]


def _phase3_learning_rate_scales(
    *, epoch: int, epochs: int, representation_start: float, representation_ramp: float
) -> tuple[float, float]:
    """Return representation/operator LR multipliers for staged joint refinement."""
    if not 0 <= epoch < epochs or not 0 <= representation_start <= 1:
        raise ValueError("invalid Phase-3 learning-rate schedule")
    if not 0 <= representation_ramp <= 1 or representation_start + representation_ramp > 1:
        raise ValueError("invalid Phase-3 representation ramp")
    progress = epoch / max(epochs - 1, 1)
    if progress <= representation_start:
        representation = 0.0
    else:
        denominator = representation_ramp or max(1.0 - representation_start, 1.0e-12)
        representation = min(1.0, (progress - representation_start) / denominator)
    decay = 0.5 if epoch >= max(1, epochs // 2) else 1.0
    return representation * decay, decay


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


@torch.no_grad()
def _from_scratch_residual_scale(
    backbone: FieldJEPAKoopmanModel,
    adaptive_model: AdaptiveKoopmanModel,
    normalizer: ChannelStandardizer,
    records: Any,
    training_trajectory_ids: set[str],
    device: torch.device,
) -> torch.Tensor:
    """Fit a fixed latent residual scale from the from-scratch training split only."""
    residuals: list[torch.Tensor] = []
    generator = adaptive_model.operator_adapter.nominal_generator.detach().float()
    for record in records:
        if record.trajectory_id not in training_trajectory_ids:
            continue
        states = normalizer.transform(
            record.states_raw.to(device=device, dtype=torch.float32)
        )
        latents = backbone.encode_target(states).float()
        dts = record.dts.to(device=device, dtype=torch.float32)
        transitions = torch.linalg.matrix_exp(generator.unsqueeze(0) * dts[:, None, None])
        nominal = torch.einsum("bij,bj->bi", transitions, latents[:-1])
        residuals.append((latents[1:] - nominal).cpu())
    if not residuals:
        raise ValueError("from-scratch residual scale has no training trajectories")
    values = torch.cat(residuals)
    floor = max(
        float(values.square().mean().sqrt()) * 1.0e-3,
        torch.finfo(torch.float32).eps,
    )
    return values.square().mean(dim=0).sqrt().clamp_min(floor)


def _save_payload(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_phase3_models(
    config: ProjectConfig,
    *,
    cache: Any,
    context_checkpoint: str | Path,
    backbone_checkpoint: str | Path,
    device: torch.device,
    route: str,
) -> tuple[
    FieldJEPAKoopmanModel,
    nn.Module,
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
    inherited_backbone = initialize_v0_6_model(config, device=device)
    inherited_backbone.load_online_state_dict(saved_backbone.online_model_state)
    inherited_backbone.target_encoder.load_state_dict(
        saved_backbone.target_model_state, strict=True
    )
    normalizer = ChannelStandardizer(eps=config.data.normalization.eps)
    normalizer.load_state_dict(saved_backbone.normalizer_state)
    if not normalizer.matches_state_dict(cache.normalizer_state):
        raise ValueError("Phase-3 backbone/cache normalizer mismatch")
    reference_encoder = copy.deepcopy(inherited_backbone.online_encoder).to(device)
    reference_encoder.requires_grad_(False)
    reference_encoder.eval()
    reference_decoder = copy.deepcopy(inherited_backbone.training_decoder).to(device)
    reference_decoder.requires_grad_(False)
    reference_decoder.eval()

    if route in {"frozen", "joint"}:
        backbone = inherited_backbone
    elif route == "from_scratch":
        backbone = initialize_v0_6_model(config, device=device)
        backbone.hard_sync_target()
    else:
        raise ValueError("invalid Phase-3 model route")

    context_payload = load_context_checkpoint(context_checkpoint)
    history = int(context_payload["history_length_steps"])
    dynamic = build_dynamic_context_model(
        context_config,
        family=str(context_payload["context_family"]),
        latent_dim=cache.latent_dim,
        parameter_dim=cache.context_parameter_dim,
        history=history,
    )
    if route in {"frozen", "joint"}:
        dynamic.load_state_dict(context_payload["best_context_state"], strict=True)
    nominal_generator = (
        cache.nominal_generator
        if route in {"frozen", "joint"}
        else backbone.koopman_core.A.detach()
    )
    operator = FactorizedAdaptiveOperator(
        nominal_generator,
        context_config.context_dim,
        adaptive_config,
        phase2,
        trainable_nominal=route == "from_scratch",
    )
    adaptive_model = AdaptiveKoopmanModel(dynamic.context_encoder, operator).to(device)
    assert_online_reencoding_required(route, uses_frozen_latent_cache=False)
    configure_phase3_route(
        route,
        backbone=backbone,
        context_encoder=adaptive_model.context_encoder,
        operator=adaptive_model.operator_adapter,
        physical_decoder=None,
        joint_backbone_allowlist=phase3.joint_backbone_allowlist,
    )
    backbone.target_encoder.requires_grad_(False)
    backbone.target_encoder.eval()
    # The predictor used by this route is adaptive_model.operator_adapter.  Keep the
    # duplicate shell core frozen; from-scratch trains its own nominal generator there.
    backbone.koopman_core.requires_grad_(False)
    if route == "joint" and backbone.koopman_core.A.requires_grad:
        raise RuntimeError("Phase-3 joint route must keep inherited A0 frozen")
    return (
        backbone,
        reference_encoder,
        reference_decoder,
        adaptive_model,
        normalizer,
        context_payload,
    )


@torch.no_grad()
def _evaluate(
    backbone: FieldJEPAKoopmanModel,
    reference_encoder: nn.Module,
    reference_decoder: nn.Module,
    adaptive_model: AdaptiveKoopmanModel,
    loader: DataLoader,
    normalizer: ChannelStandardizer,
    residual_scale: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    config: ProjectConfig,
    device: torch.device,
    epoch: int,
    route: str,
    observer_admitted: bool,
    observer_controls: dict[str, nn.Module] | None = None,
) -> dict[str, float]:
    backbone.eval()
    adaptive_model.eval()
    totals: dict[str, float] = {}
    count = 0
    maximum = {"representation_drift", "physical_manifold_violation", "burden_max"}
    observer_predictions: list[torch.Tensor] = []
    observer_targets: list[torch.Tensor] = []
    candidate_representations: list[torch.Tensor] = []
    reference_representations: list[torch.Tensor] = []
    for raw in loader:
        batch = move_raw_field_batch(raw, device)
        objective = joint_markov_objective(
            backbone,
            reference_encoder,
            reference_decoder,
            adaptive_model,
            batch,
            normalizer,
            residual_scale,
            condition_mean,
            condition_std,
            config,
            epoch=epoch,
            validation=True,
            route=route,
            observer_admitted=observer_admitted,
        )
        batch_size = batch["history_raw"].shape[0]
        for name, value in {"total": objective.total, **objective.terms}.items():
            scalar = float(value)
            is_maximum = name in maximum or name.startswith(
                ("decoded_divergence", "decoded_boundary", "decoded_outer_boundary")
            )
            if is_maximum:
                totals[name] = max(totals.get(name, float("-inf")), scalar)
            else:
                totals[name] = totals.get(name, 0.0) + scalar * batch_size
        q_hat = objective.adaptive.rollout.get("q_hat")
        if q_hat is not None:
            steps = q_hat.shape[1]
            normalized_target = (
                batch["future_condition_targets"][:, :steps] - condition_mean
            ) / condition_std
            observer_predictions.append(q_hat.reshape(-1, q_hat.shape[-1]).cpu())
            observer_targets.append(
                normalized_target.reshape(-1, normalized_target.shape[-1]).cpu()
            )
        history_model = normalizer.transform(batch["history_raw"][:, -1])
        candidate_representations.append(backbone.encode(history_model).float().cpu())
        reference_representations.append(reference_encoder(history_model).float().cpu())
        count += batch_size
    if count == 0:
        raise RuntimeError("Phase-3 validation dataset is empty")
    result = {
        name: (
            value
            if name in maximum
            or name.startswith(
                ("decoded_divergence", "decoded_boundary", "decoded_outer_boundary")
            )
            else value / count
        )
        for name, value in totals.items()
    }
    if observer_predictions:
        observer = condition_observer_metrics(
            torch.cat(observer_predictions), torch.cat(observer_targets)
        )
        result.update({f"observer_{name}": value for name, value in observer.items()})
    phase3 = config.v0_9_phase3
    phase2 = config.v0_9_phase2
    adaptive = config.v0_9_adaptive
    if (
        phase3 is not None
        and phase2 is not None
        and adaptive is not None
        and phase3.observer_admission_enabled
        and adaptive.condition_mode == "latent_inferred"
    ):
        if observer_controls is None:
            raise RuntimeError("Phase-3.7 latent evaluation requires frozen observer controls")
        control_metrics = _evaluate_observer_controls(
            backbone,
            adaptive_model,
            observer_controls,
            loader,
            normalizer,
            condition_mean,
            condition_std,
            device,
        )
        control_decision = classify_observer_admission(
            control_metrics, phase2, phase3
        )
        result.update(
            {
                "observer_admitted": float(
                    observer_admitted and bool(control_decision["admitted"])
                ),
                "observer_history_gain_vs_instantaneous": float(
                    control_decision["history_gain_vs_instantaneous"]
                ),
                "observer_history_gain_vs_shuffled": float(
                    control_decision["history_gain_vs_shuffled"]
                ),
                "observer_instantaneous_normalized_rmse": float(
                    control_metrics["instantaneous"]["normalized_rmse"]
                ),
                "observer_shuffled_normalized_rmse": float(
                    control_metrics["shuffled_history"]["normalized_rmse"]
                ),
                "observer_mean_normalized_rmse": float(
                    control_metrics["mean"]["normalized_rmse"]
                ),
            }
        )
    candidate = torch.cat(candidate_representations)
    reference = torch.cat(reference_representations)
    result.update(
        {
            "representation_linear_cka": float(centered_linear_cka(candidate, reference)),
            "representation_procrustes_nrmse": float(
                orthogonal_procrustes_nrmse(candidate, reference)
            ),
            "representation_effective_rank": float(
                representation_effective_rank(candidate)
            ),
            "reference_effective_rank": float(
                representation_effective_rank(reference)
            ),
        }
    )
    if config.v0_9_phase3 and config.v0_9_phase3.physics_aligned_latent_enabled:
        gauge = dynamical_gauge_metrics(
            candidate,
            reference,
            adaptive_model.operator_adapter.nominal_generator,
        )
        result.update({name: float(value) for name, value in gauge.items()})
    result["nominal_symmetric_abscissa"] = float(
        symmetric_abscissa_proxy(
            adaptive_model.operator_adapter.nominal_generator.detach().float()
        )
    )
    return result


def _observer_prediction(
    observer: nn.Module,
    context: torch.Tensor,
    features: torch.Tensor,
    *,
    output_limit: float,
) -> torch.Tensor:
    raw = observer(torch.cat((context, features), dim=-1))
    return output_limit * torch.tanh(raw / output_limit)


@torch.no_grad()
def _evaluate_observer_controls(
    backbone: FieldJEPAKoopmanModel,
    adaptive_model: AdaptiveKoopmanModel,
    observers: dict[str, nn.Module],
    loader: DataLoader,
    normalizer: ChannelStandardizer,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    predictions = {name: [] for name in OBSERVER_VARIANTS}
    targets: list[torch.Tensor] = []
    output_limit = float(adaptive_model.operator_adapter.observer_output_limit)
    backbone.eval()
    adaptive_model.context_encoder.eval()
    for module in observers.values():
        module.eval()
    for raw in loader:
        batch = move_raw_field_batch(raw, device)
        history_z = backbone.encode(normalizer.transform(batch["history_raw"])).float()
        target = (batch["future_condition_targets"][:, 0] - condition_mean) / condition_std
        for name in OBSERVER_VARIANTS:
            variant = observer_history_variant(history_z, name)
            context = adaptive_model.context_encoder(
                variant,
                batch["history_dts"],
                batch["future_dts"][:, :1],
                batch["context_parameters"],
            )
            features = causal_observer_features(variant, batch["history_dts"])
            predictions[name].append(
                _observer_prediction(
                    observers[name],
                    context,
                    features,
                    output_limit=output_limit,
                ).cpu()
            )
        targets.append(target.cpu())
    if not targets:
        raise RuntimeError("Phase-3 observer validation dataset is empty")
    target_values = torch.cat(targets)
    metrics = {
        name: condition_observer_metrics(torch.cat(values), target_values)
        for name, values in predictions.items()
    }
    metrics["mean"] = condition_observer_metrics(
        torch.zeros_like(target_values), target_values
    )
    return metrics


def _pretrain_observer_admission(
    backbone: FieldJEPAKoopmanModel,
    adaptive_model: AdaptiveKoopmanModel,
    loaders: dict[str, DataLoader],
    normalizer: ChannelStandardizer,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    config: ProjectConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, nn.Module] | None]:
    """Fit the causal observer independently from representation/operator gradients."""
    phase3 = config.v0_9_phase3
    phase2 = config.v0_9_phase2
    adaptive = config.v0_9_adaptive
    if phase3 is None or phase2 is None or adaptive is None:
        raise ValueError("observer admission requires complete Phase-3 configuration")
    required = adaptive.condition_mode == "latent_inferred"
    if not phase3.observer_admission_enabled:
        return (
            {
                "enabled": False,
                "required": required,
                "admitted": True,
                "reason": "legacy_joint_observer_training",
                "operator_condition_route": "legacy_joint",
            },
            None,
        )
    if not required:
        for parameter in adaptive_model.operator_adapter.condition_observer.parameters():
            parameter.requires_grad_(False)
        return (
            {
                "enabled": True,
                "required": required,
                "admitted": True,
                "reason": "known_condition_does_not_require_observer",
                "operator_condition_route": "known_condition_full",
            },
            None,
        )

    print(
        "[V0.9][phase3:observer] START independent history/instantaneous/shuffled controls",
        flush=True,
    )
    source = adaptive_model.operator_adapter.condition_observer
    observers = {name: copy.deepcopy(source).to(device) for name in OBSERVER_VARIANTS}
    optimizers = {
        name: AdamW(
            module.parameters(),
            lr=phase3.observer_learning_rate,
            weight_decay=config.v0_9_training.weight_decay if config.v0_9_training else 0.0,
        )
        for name, module in observers.items()
    }
    output_limit = float(adaptive_model.operator_adapter.observer_output_limit)
    best_keys: dict[str, tuple[float, float] | None] = {
        name: None for name in OBSERVER_VARIANTS
    }
    best_epochs = {name: 0 for name in OBSERVER_VARIANTS}
    best_states: dict[str, dict[str, Any]] = {}
    backbone.eval()
    adaptive_model.context_encoder.eval()
    for epoch in range(phase3.observer_pretrain_epochs):
        for module in observers.values():
            module.train()
        for raw in loaders["train"]:
            batch = move_raw_field_batch(raw, device)
            with torch.no_grad():
                history_z = backbone.encode(
                    normalizer.transform(batch["history_raw"])
                ).float()
                target = (
                    batch["future_condition_targets"][:, 0] - condition_mean
                ) / condition_std
            for name in OBSERVER_VARIANTS:
                variant = observer_history_variant(history_z, name)
                with torch.no_grad():
                    context = adaptive_model.context_encoder(
                        variant,
                        batch["history_dts"],
                        batch["future_dts"][:, :1],
                        batch["context_parameters"],
                    )
                    features = causal_observer_features(variant, batch["history_dts"])
                optimizers[name].zero_grad(set_to_none=True)
                prediction = _observer_prediction(
                    observers[name],
                    context,
                    features,
                    output_limit=output_limit,
                )
                loss = torch.nn.functional.smooth_l1_loss(prediction, target, beta=1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    observers[name].parameters(),
                    config.v0_9_training.gradient_clip_norm
                    if config.v0_9_training
                    else 1.0,
                )
                optimizers[name].step()
        for module in observers.values():
            module.eval()
        metrics = _evaluate_observer_controls(
            backbone,
            adaptive_model,
            observers,
            loaders["validation"],
            normalizer,
            condition_mean,
            condition_std,
            device,
        )
        for name, module in observers.items():
            values = metrics[name]
            key = (
                float(values["normalized_rmse"]),
                -float(values["minimum_r2"]),
            )
            if not all(math.isfinite(value) for value in key):
                continue
            if best_keys[name] is None or key < best_keys[name]:
                best_keys[name] = key
                best_epochs[name] = epoch + 1
                best_states[name] = copy.deepcopy(module.state_dict())
    if set(best_states) != set(OBSERVER_VARIANTS):
        raise RuntimeError("Phase-3 observer admission selected no finite control model")
    for name, module in observers.items():
        module.load_state_dict(best_states[name], strict=True)
    source.load_state_dict(best_states["history"], strict=True)
    source.requires_grad_(False)
    best_metrics = _evaluate_observer_controls(
        backbone,
        adaptive_model,
        observers,
        loaders["validation"],
        normalizer,
        condition_mean,
        condition_std,
        device,
    )
    decision = classify_observer_admission(best_metrics, phase2, phase3)
    report: dict[str, Any] = {
        "enabled": True,
        "required": True,
        "selection_split": "validation",
        "best_epochs": best_epochs,
        "epochs": phase3.observer_pretrain_epochs,
        "controls": best_metrics,
        **decision,
    }
    report["operator_condition_route"] = (
        "latent_condition_full" if report["admitted"] else "history_only_fallback"
    )
    report["operator_route_frozen_after_initial_admission"] = True
    print(
        "[V0.9][phase3:observer] PASS "
        f"admitted={report['admitted']} "
        f"rmse={best_metrics['history']['normalized_rmse']:.6g} "
        f"history_gain={report['history_gain_vs_instantaneous']:.6g}",
        flush=True,
    )
    return report, observers


def _train_v0_9_phase3_route(
    config: ProjectConfig | str | Path,
    *,
    context_checkpoint: str | Path,
    adaptive_cache: str | Path,
    backbone_checkpoint: str | Path,
    physical_dataset: str | Path,
    run_dir: str | Path,
    frozen_target_cache: str | Path | None = None,
    device: str | torch.device = "cuda",
    route: str,
) -> Phase3JointTrainingResult:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    phase3 = resolved.v0_9_phase3
    phase2 = resolved.v0_9_phase2
    training = resolved.v0_9_training
    evaluation = resolved.v0_9_evaluation
    adaptive_config = resolved.v0_9_adaptive
    cylinder = resolved.cylinder_wake_2d
    required = (
        phase3,
        phase2,
        training,
        evaluation,
        adaptive_config,
        cylinder,
        resolved.ema,
    )
    if any(value is None for value in required) or not phase3.enabled:
        raise ValueError("Phase-3 joint training requires enabled complete configuration")
    assert phase3 and phase2 and training and evaluation and adaptive_config and cylinder
    assert resolved.ema
    if route not in {"joint", "from_scratch"}:
        raise ValueError("Phase-3 training route must be joint or from_scratch")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    destination = Path(run_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("config", "logs", "checkpoints", "evaluation", "metadata"):
        (destination / name).mkdir()
    save_config(resolved, destination / "config" / "resolved_config.yaml")
    print(
        f"[V0.9][phase3:{route}:{resolved.v0_9_adaptive.condition_mode}] START "
        f"seed={resolved.training.seed} init={training.operator_initialization_seed} "
        f"epochs={training.epochs} device={selected}",
        flush=True,
    )
    set_global_seed(
        training.operator_initialization_seed,
        deterministic=resolved.training.deterministic,
    )
    cache = load_adaptive_cache(adaptive_cache)
    (
        backbone,
        reference_encoder,
        reference_decoder,
        adaptive_model,
        normalizer,
        context_payload,
    ) = (
        _build_phase3_models(
            resolved,
            cache=cache,
            context_checkpoint=context_checkpoint,
            backbone_checkpoint=backbone_checkpoint,
            device=selected,
            route=route,
        )
    )
    dataset = load_cylinder_wake_dataset(physical_dataset, cylinder)
    target_cache_path = None if frozen_target_cache is None else Path(frozen_target_cache)
    if route == "from_scratch" and target_cache_path is not None:
        raise ValueError("from-scratch Phase-3 must use its live EMA target encoder")
    frozen_targets: dict[str, torch.Tensor] | None = None
    if route == "joint" and target_cache_path is not None and target_cache_path.is_file():
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
    elif route == "joint":
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
    if route == "from_scratch":
        residual_scale_cpu = _from_scratch_residual_scale(
            backbone,
            adaptive_model,
            normalizer,
            dataset.records,
            {item.trajectory_id for item in cache.select("train")},
            selected,
        )
        residual_scale_source = "from_scratch_initial_training_split"
    else:
        residual_scale_cpu, _, _ = adaptive_training_scales(cache)
        residual_scale_source = "inherited_phase2_training_split"
    condition_mean_cpu, condition_std_cpu = phase2_condition_scales(cache)
    residual_scale = residual_scale_cpu.to(selected)
    condition_mean = condition_mean_cpu.to(selected)
    condition_std = condition_std_cpu.to(selected)

    observer_admission, observer_controls = _pretrain_observer_admission(
        backbone,
        adaptive_model,
        loaders,
        normalizer,
        condition_mean,
        condition_std,
        resolved,
        selected,
    )
    observer_admitted = bool(observer_admission["admitted"])

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
    amp_enabled = selected.type == "cuda" and training.precision != "fp32"
    amp_dtype = torch.float16 if training.precision == "amp_fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype is torch.float16)
    ema = (
        EMATracker(resolved.ema, len(loaders["train"]) * training.epochs)
        if route == "from_scratch"
        else None
    )
    nominal_core = backbone.koopman_core.A.detach().clone()
    nominal_adapter = adaptive_model.operator_adapter.nominal_generator.detach().clone()
    best_states: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_ema_state: dict[str, Any] | None = None
    best_metrics: dict[str, float] = {}
    completed = 0
    history_rows: list[dict[str, Any]] = []
    maturity_fraction = max(
        max(training.rollout_start_fractions),
        training.physics_start_fraction + training.physics_ramp_duration_fraction,
    )
    earliest_stop_epoch = math.ceil(maturity_fraction * max(training.epochs - 1, 1)) + 1
    checkpoint_tracker = MatureCheckpointTracker(
        earliest_epoch=earliest_stop_epoch,
        patience=training.patience,
    )
    for epoch in range(training.epochs):
        if route == "joint":
            representation_lr_scale, operator_lr_scale = _phase3_learning_rate_scales(
                epoch=epoch,
                epochs=training.epochs,
                representation_start=training.physics_start_fraction,
                representation_ramp=training.physics_ramp_duration_fraction,
            )
        else:
            decay = 0.5 if epoch >= max(1, training.epochs // 2) else 1.0
            representation_lr_scale = operator_lr_scale = decay
        optimizer.param_groups[0]["lr"] = (
            phase3.representation_learning_rate * representation_lr_scale
        )
        optimizer.param_groups[1]["lr"] = phase3.operator_learning_rate * operator_lr_scale
        for parameter in representation_parameters:
            parameter.requires_grad_(route != "joint" or representation_lr_scale > 0)
        backbone.train()
        # The JEPA teacher is an immutable target for joint and is EMA-updated only
        # after optimizer steps for from-scratch. It must never enter train mode.
        backbone.target_encoder.eval()
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
                    reference_decoder,
                    adaptive_model,
                    batch,
                    normalizer,
                    residual_scale,
                    condition_mean,
                    condition_std,
                    resolved,
                    epoch=epoch,
                    validation=False,
                    route=route,
                    observer_admitted=observer_admitted,
                )
            scaler.scale(objective.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [*representation_parameters, *adaptive_parameters],
                training.gradient_clip_norm,
            )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_updated = not scaler.is_enabled() or scaler.get_scale() >= scale_before
            if route == "from_scratch" and optimizer_updated:
                adaptive_model.operator_adapter.project_trainable_nominal_stability_()
            if ema is not None and optimizer_updated:
                ema.update_after_optimizer(backbone)
        if route == "joint" and (
            not torch.equal(backbone.koopman_core.A.detach(), nominal_core)
            or not torch.equal(
                adaptive_model.operator_adapter.nominal_generator.detach(), nominal_adapter
            )
        ):
            raise RuntimeError("Phase-3 joint training changed the frozen nominal generator A0")
        metrics = _evaluate(
            backbone,
            reference_encoder,
            reference_decoder,
            adaptive_model,
            loaders["validation"],
            normalizer,
            residual_scale,
            condition_mean,
            condition_std,
            resolved,
            selected,
            epoch,
            route,
            observer_admitted,
            observer_controls,
        )
        key = phase3_checkpoint_key(
            metrics,
            phase3,
            evaluation,
            phase2,
            condition_mode=adaptive_config.condition_mode,
            route=route,
        )
        history_rows.append(
            {
                "epoch": epoch + 1,
                "representation_lr_scale": representation_lr_scale,
                "operator_lr_scale": operator_lr_scale,
                **metrics,
            }
        )
        completed = epoch + 1
        selected_checkpoint, should_stop = checkpoint_tracker.consider(completed, key)
        if selected_checkpoint:
            best_states = (
                copy.deepcopy(backbone.state_dict()),
                copy.deepcopy(adaptive_model.state_dict()),
            )
            best_metrics = dict(metrics)
            best_ema_state = None if ema is None else copy.deepcopy(ema.state_dict())
        if should_stop:
            break
    if best_states is None or checkpoint_tracker.best_epoch is None:
        raise RuntimeError("Phase-3 joint training selected no mature finite checkpoint")
    best_epoch = checkpoint_tracker.best_epoch
    backbone.load_state_dict(best_states[0], strict=True)
    adaptive_model.load_state_dict(best_states[1], strict=True)
    locked_test_metrics = _evaluate(
        backbone,
        reference_encoder,
        reference_decoder,
        adaptive_model,
        loaders["test"],
        normalizer,
        residual_scale,
        condition_mean,
        condition_std,
        resolved,
        selected,
        best_epoch - 1,
        route,
        observer_admitted,
        observer_controls,
    )
    if observer_controls is not None:
        validation_controls = _evaluate_observer_controls(
            backbone,
            adaptive_model,
            observer_controls,
            loaders["validation"],
            normalizer,
            condition_mean,
            condition_std,
            selected,
        )
        locked_controls = _evaluate_observer_controls(
            backbone,
            adaptive_model,
            observer_controls,
            loaders["test"],
            normalizer,
            condition_mean,
            condition_std,
            selected,
        )
        observer_admission["joint_validation"] = {
            "controls": validation_controls,
            **classify_observer_admission(validation_controls, phase2, phase3),
        }
        observer_admission["locked_test"] = {
            "controls": locked_controls,
            **classify_observer_admission(locked_controls, phase2, phase3),
        }
    (destination / "logs" / "history.json").write_text(
        json.dumps(history_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trainable_count = sum(
        parameter.numel() for parameter in [*representation_parameters, *adaptive_parameters]
    )
    checkpoint = destination / "checkpoints" / f"best_{route}_markov.pt"
    _save_payload(
        {
            "schema_version": (
                "v0.9-phase3-route-2"
                if phase3.physics_aligned_latent_enabled
                or phase3.observer_admission_enabled
                else "v0.9-phase3-route-1"
            ),
            "route": route,
            "config": resolved.to_dict(),
            "config_hash": resolved.stable_hash,
            "backbone_state": backbone.state_dict(),
            "adaptive_state": adaptive_model.state_dict(),
            "reference_encoder_state": reference_encoder.state_dict(),
            "reference_decoder_state": reference_decoder.state_dict(),
            "normalizer_state": normalizer.state_dict(),
            "source_backbone_sha256": file_sha256(backbone_checkpoint),
            "source_context_sha256": file_sha256(context_checkpoint),
            # No trainable route starts from a trained Phase-2 adaptive state. Joint
            # retains the inherited A0; from-scratch learns its own nominal generator.
            "source_phase2_checkpoint_sha256": None,
            "adaptive_cache_fingerprint": cache.fingerprint,
            "frozen_target_cache_sha256": (
                None if target_cache_path is None else file_sha256(target_cache_path)
            ),
            "ema_state": best_ema_state,
            "condition_mean": condition_mean_cpu,
            "condition_std": condition_std_cpu,
            "residual_scale": residual_scale_cpu,
            "residual_scale_source": residual_scale_source,
            "completed_epochs": completed,
            "best_epoch": best_epoch,
            "checkpoint_eligible_from_epoch": earliest_stop_epoch,
            "trainable_parameters": trainable_count,
            "best_validation_metrics": best_metrics,
            "locked_test_metrics": locked_test_metrics,
            "observer_admission": observer_admission,
            "observer_control_states": (
                None
                if observer_controls is None
                else {
                    name: module.state_dict()
                    for name, module in observer_controls.items()
                }
            ),
            "nominal_generator_unchanged": route == "joint",
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
    (destination / "evaluation" / "observer_admission.json").write_text(
        json.dumps(observer_admission, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[V0.9][phase3:{route}:{resolved.v0_9_adaptive.condition_mode}] PASS "
        f"epochs={completed} best_epoch={best_epoch} val_total={best_metrics['total']:.6g} "
        f"drift={best_metrics['representation_drift']:.6g}",
        flush=True,
    )
    return Phase3JointTrainingResult(
        route,
        destination,
        checkpoint,
        completed,
        best_epoch,
        trainable_count,
        best_metrics,
        locked_test_metrics,
        observer_admission,
    )


def train_v0_9_phase3_joint(
    config: ProjectConfig | str | Path,
    **kwargs: Any,
) -> Phase3JointTrainingResult:
    return _train_v0_9_phase3_route(config, route="joint", **kwargs)


def train_v0_9_phase3_from_scratch(
    config: ProjectConfig | str | Path,
    **kwargs: Any,
) -> Phase3JointTrainingResult:
    kwargs.pop("frozen_target_cache", None)
    return _train_v0_9_phase3_route(
        config,
        route="from_scratch",
        frozen_target_cache=None,
        **kwargs,
    )


@torch.no_grad()
def evaluate_v0_9_phase3_frozen(
    config: ProjectConfig | str | Path,
    *,
    adaptive_checkpoint: str | Path,
    context_checkpoint: str | Path,
    adaptive_cache: str | Path,
    backbone_checkpoint: str | Path,
    physical_dataset: str | Path,
    run_dir: str | Path,
    frozen_target_cache: str | Path | None = None,
    device: str | torch.device = "cuda",
) -> Phase3FrozenEvaluationResult:
    """Evaluate the inherited frozen route with the exact Phase-3 decoded metrics."""
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    required = (
        resolved.v0_9_phase3,
        resolved.v0_9_phase2,
        resolved.v0_9_training,
        resolved.v0_9_adaptive,
        resolved.cylinder_wake_2d,
    )
    if any(value is None for value in required):
        raise ValueError("Phase-3 frozen evaluation requires a complete configuration")
    assert resolved.v0_9_phase3 and resolved.v0_9_training and resolved.v0_9_adaptive
    assert resolved.cylinder_wake_2d
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    destination = Path(run_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "evaluation").mkdir()
    save_config(resolved, destination / "resolved_config.yaml")
    print(
        f"[V0.9][phase3:frozen:{resolved.v0_9_adaptive.condition_mode}] START ",
        f"seed={resolved.training.seed}",
        flush=True,
    )
    cache = load_adaptive_cache(adaptive_cache)
    (
        backbone,
        reference_encoder,
        reference_decoder,
        adaptive_model,
        normalizer,
        context_payload,
    ) = (
        _build_phase3_models(
            resolved,
            cache=cache,
            context_checkpoint=context_checkpoint,
            backbone_checkpoint=backbone_checkpoint,
            device=selected,
            route="frozen",
        )
    )
    payload = load_adaptive_checkpoint(adaptive_checkpoint)
    if payload["adaptive_cache_fingerprint"] != cache.fingerprint:
        raise ValueError("Phase-3 frozen checkpoint/cache mismatch")
    if payload["backbone_checkpoint_sha256"] != file_sha256(backbone_checkpoint):
        raise ValueError("Phase-3 frozen checkpoint/backbone mismatch")
    if payload["context_checkpoint_sha256"] != file_sha256(context_checkpoint):
        raise ValueError("Phase-3 frozen checkpoint/context mismatch")
    adaptive_model.operator_adapter.load_state_dict(
        payload["best_adaptive_state"], strict=True
    )
    dataset = load_cylinder_wake_dataset(physical_dataset, resolved.cylinder_wake_2d)
    target_cache_path = None if frozen_target_cache is None else Path(frozen_target_cache)
    if target_cache_path is not None and target_cache_path.is_file():
        try:
            target_payload = torch.load(
                target_cache_path, map_location="cpu", weights_only=False
            )
        except TypeError:  # pragma: no cover
            target_payload = torch.load(target_cache_path, map_location="cpu")
        if (
            not isinstance(target_payload, dict)
            or target_payload.get("schema_version") != "v0.9-phase3-target-1"
            or target_payload.get("backbone_checkpoint_sha256")
            != file_sha256(backbone_checkpoint)
        ):
            raise ValueError("Phase-3 frozen target-cache provenance mismatch")
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
        *resolved.v0_9_training.rollout_horizons,
        *resolved.v0_9_training.active_observable_horizons,
    )
    test_dataset = RawFieldAdaptiveRolloutDataset(
        cache,
        dataset.records,
        "test",
        history,
        maximum_horizon,
        stride=resolved.v0_9_phase3.raw_field_rollout_stride,
        frozen_target_latents=frozen_targets,
    )
    loader = DataLoader(
        test_dataset,
        batch_size=resolved.v0_9_phase3.raw_field_batch_size,
        shuffle=False,
        num_workers=0,
    )
    residual_scale_cpu = torch.as_tensor(payload["residual_training_scale"]).float()
    condition_mean_cpu, condition_std_cpu = phase2_condition_scales(cache)
    metrics = _evaluate(
        backbone,
        reference_encoder,
        reference_decoder,
        adaptive_model,
        loader,
        normalizer,
        residual_scale_cpu.to(selected),
        condition_mean_cpu.to(selected),
        condition_std_cpu.to(selected),
        resolved,
        selected,
        resolved.v0_9_training.epochs - 1,
        "frozen",
        True,
    )
    (destination / "evaluation" / "locked_test_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[V0.9][phase3:frozen:{resolved.v0_9_adaptive.condition_mode}] PASS ",
        f"total={metrics['total']:.6g}",
        flush=True,
    )
    return Phase3FrozenEvaluationResult("frozen", destination, 0, metrics)
