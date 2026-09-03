"""Raw-field online re-encoding and objectives for the Phase-3 joint route."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from jka_model.adaptive import AdaptiveCache, AdaptiveKoopmanModel
from jka_model.adaptive.identifiability import condition_targets
from jka_model.adaptive.objectives import (
    AdaptiveObjectiveResult,
    Phase2TrainingState,
    adaptive_stabilization_objective,
    curriculum_state,
)
from jka_model.config import ProjectConfig, V09EvaluationConfig, V09Phase2Config, V09Phase3Config
from jka_model.data import ChannelStandardizer
from jka_model.data.datasets import TrajectoryDataset
from jka_model.manifold.physical import central_difference_2d, physical_manifold_metrics
from jka_model.models import FieldJEPAKoopmanModel


class RawFieldAdaptiveRolloutDataset(Dataset[dict[str, Any]]):
    """Phase-2-aligned rollout windows that retain raw fields, never cached latents."""

    def __init__(
        self,
        cache: AdaptiveCache,
        records: TrajectoryDataset,
        split: str,
        history: int,
        horizon: int,
        *,
        stride: int,
        frozen_target_latents: Mapping[str, Tensor] | None = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("invalid Phase-3 raw-field split")
        if min(history, horizon, stride) < 1:
            raise ValueError("Phase-3 history/horizon/stride must be positive")
        record_by_id = {record.trajectory_id: record for record in records}
        self.items: list[tuple[Any, Any, int]] = []
        self.frozen_target_latents = frozen_target_latents
        for trajectory in cache.select(split):
            record = record_by_id.get(trajectory.trajectory_id)
            if record is None:
                raise ValueError("Phase-3 raw dataset misses a cached trajectory")
            if record.states_raw.shape[0] != trajectory.latents.shape[0] or not torch.equal(
                record.dts.float(), trajectory.dts.float()
            ):
                raise ValueError("Phase-3 raw/cache trajectory alignment mismatch")
            if record.valid_mask is None:
                raise ValueError("Phase-3 cylinder route requires a valid fluid mask")
            if frozen_target_latents is not None:
                target = frozen_target_latents.get(trajectory.trajectory_id)
                if target is None or target.shape != trajectory.latents.shape:
                    raise ValueError("Phase-3 frozen target-latent alignment mismatch")
            for target_index in range(
                history - 1,
                trajectory.dts.shape[0] - horizon + 1,
                stride,
            ):
                self.items.append((trajectory, record, target_index))
        if not self.items:
            raise ValueError("Phase-3 raw-field dataset has no valid rollout windows")
        self.history = history
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory, record, target_index = self.items[index]
        start = target_index - self.history + 1
        stop = target_index + self.horizon
        targets = condition_targets(trajectory.conditions, trajectory.dts)
        result = {
            "history_raw": record.states_raw[start : target_index + 1].clone(),
            "target_raw": record.states_raw[target_index + 1 : stop + 1].clone(),
            "valid_mask": record.valid_mask.clone(),
            "history_dts": trajectory.dts[start:target_index].clone(),
            "future_dts": trajectory.dts[target_index:stop].clone(),
            "future_conditions": trajectory.conditions[target_index:stop].clone(),
            "future_condition_targets": targets[target_index:stop].clone(),
            "context_parameters": trajectory.context_parameters.clone(),
            "schedule_type": trajectory.schedule_type,
            "trajectory_id": trajectory.trajectory_id,
            "target_index": target_index,
        }
        if self.frozen_target_latents is not None:
            result["target_latents"] = self.frozen_target_latents[trajectory.trajectory_id][
                target_index + 1 : stop + 1
            ].clone()
        return result


@dataclass(slots=True)
class JointMarkovObjectiveResult:
    total: Tensor
    terms: dict[str, Tensor]
    adaptive: AdaptiveObjectiveResult


def centered_linear_cka(candidate: Tensor, reference: Tensor) -> Tensor:
    """Linear CKA after centering; invariant to orthogonal basis and isotropic scale."""
    if candidate.shape != reference.shape or candidate.ndim != 2 or candidate.shape[0] < 2:
        raise ValueError("linear CKA requires aligned [N,d] representations with N >= 2")
    candidate = candidate.float() - candidate.float().mean(dim=0, keepdim=True)
    reference = reference.float() - reference.float().mean(dim=0, keepdim=True)
    cross = reference.T @ candidate
    numerator = cross.square().sum()
    denominator = (reference.T @ reference).square().sum().sqrt() * (
        (candidate.T @ candidate).square().sum().sqrt()
    )
    return numerator / denominator.clamp_min(1.0e-12)


def orthogonal_procrustes_nrmse(candidate: Tensor, reference: Tensor) -> Tensor:
    """Centered/normalized residual after the optimal orthogonal latent alignment."""
    if candidate.shape != reference.shape or candidate.ndim != 2 or candidate.shape[0] < 2:
        raise ValueError("Procrustes requires aligned [N,d] representations with N >= 2")
    candidate = candidate.float() - candidate.float().mean(dim=0, keepdim=True)
    reference = reference.float() - reference.float().mean(dim=0, keepdim=True)
    candidate = candidate / candidate.norm().clamp_min(1.0e-12)
    reference = reference / reference.norm().clamp_min(1.0e-12)
    left, _, right_h = torch.linalg.svd(candidate.T @ reference, full_matrices=False)
    aligned = candidate @ (left @ right_h)
    return (aligned - reference).norm()


def dynamical_gauge_metrics(
    candidate: Tensor,
    reference: Tensor,
    nominal_generator: Tensor,
) -> dict[str, Tensor]:
    """Separate coordinate gauge drift from incompatibility with the frozen A0.

    The optimal orthogonal map aligns the candidate coordinates to the reference.
    Its commutator with ``A0`` then tests whether this apparent gauge change preserves
    the nominal Koopman dynamics rather than merely preserving pairwise geometry.
    """
    if candidate.shape != reference.shape or candidate.ndim != 2 or candidate.shape[0] < 2:
        raise ValueError("dynamical gauge audit requires aligned [N,d] representations")
    if nominal_generator.shape != (candidate.shape[1], candidate.shape[1]):
        raise ValueError("dynamical gauge audit requires a matching square generator")
    candidate_centered = candidate.float() - candidate.float().mean(dim=0, keepdim=True)
    reference_centered = reference.float() - reference.float().mean(dim=0, keepdim=True)
    candidate_scale = candidate_centered.norm().clamp_min(1.0e-12)
    reference_scale = reference_centered.norm().clamp_min(1.0e-12)
    candidate_normalized = candidate_centered / candidate_scale
    reference_normalized = reference_centered / reference_scale
    left, _, right_h = torch.linalg.svd(
        candidate_normalized.T @ reference_normalized,
        full_matrices=False,
    )
    transform = left @ right_h
    aligned_nrmse = (candidate_normalized @ transform - reference_normalized).norm()
    generator = nominal_generator.detach().to(device=candidate.device, dtype=torch.float32)
    commutator = generator @ transform - transform @ generator
    normalized_commutator = commutator.norm() / generator.norm().clamp_min(1.0e-12)
    return {
        "dynamical_gauge_nrmse": aligned_nrmse,
        "generator_commutator": normalized_commutator,
    }


def representation_effective_rank(representation: Tensor) -> Tensor:
    """Entropy effective rank of the centered latent covariance."""
    if representation.ndim != 2 or representation.shape[0] < 2:
        raise ValueError("effective rank requires an [N,d] representation with N >= 2")
    centered = representation.float() - representation.float().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    probabilities = energy / energy.sum().clamp_min(1.0e-12)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
    return entropy.exp()


@dataclass(slots=True)
class MatureCheckpointTracker:
    """Early stopping whose patience begins only after the full curriculum is active."""

    earliest_epoch: int
    patience: int
    best_key: tuple[float, ...] | None = None
    best_epoch: int | None = None
    stale_epochs: int = 0
    last_epoch: int = 0

    def __post_init__(self) -> None:
        if min(self.earliest_epoch, self.patience) < 1:
            raise ValueError("Phase-3 mature checkpoint epochs/patience must be positive")

    def consider(self, completed_epoch: int, key: tuple[float, ...]) -> tuple[bool, bool]:
        if completed_epoch <= self.last_epoch or not key or not all(math.isfinite(v) for v in key):
            raise ValueError("invalid Phase-3 checkpoint update")
        self.last_epoch = completed_epoch
        if completed_epoch < self.earliest_epoch:
            return False, False
        selected = self.best_key is None or key < self.best_key
        if selected:
            self.best_key = key
            self.best_epoch = completed_epoch
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return selected, self.stale_epochs >= self.patience


def phase3_checkpoint_key(
    metrics: Mapping[str, float],
    phase3: V09Phase3Config,
    evaluation: V09EvaluationConfig,
    phase2: V09Phase2Config,
    *,
    condition_mode: str,
    route: str = "joint",
) -> tuple[float, ...]:
    """Rank mature checkpoints by feasibility before matched predictive skill."""
    if condition_mode not in {"known", "latent_inferred"}:
        raise ValueError("invalid Phase-3 condition mode")
    if route not in {"joint", "from_scratch"}:
        raise ValueError("invalid trainable Phase-3 route")
    physical = max(float(metrics["physical_manifold_violation"]) / 1.0e-8 - 1.0, 0.0)
    drift = (
        max(
            float(metrics["representation_drift"])
            / phase3.max_normalized_representation_drift
            - 1.0,
            0.0,
        )
        if route == "joint"
        else 0.0
    )
    roundtrip = max(float(metrics["roundtrip"]) / phase3.max_roundtrip_nrmse - 1.0, 0.0)
    gauge_nrmse = (
        max(
            float(metrics.get("dynamical_gauge_nrmse", float("inf")))
            / phase3.max_dynamical_gauge_nrmse
            - 1.0,
            0.0,
        )
        if phase3.physics_aligned_latent_enabled and route == "joint"
        else 0.0
    )
    commutator = (
        max(
            float(metrics.get("generator_commutator", float("inf")))
            / phase3.max_generator_commutator
            - 1.0,
            0.0,
        )
        if phase3.physics_aligned_latent_enabled and route == "joint"
        else 0.0
    )
    observer_rmse = (
        max(
            float(metrics.get("observer_normalized_rmse", float("inf")))
            / phase2.max_condition_observer_normalized_rmse
            - 1.0,
            0.0,
        )
        if condition_mode == "latent_inferred"
        else 0.0
    )
    observer_r2 = (
        max(
            (
                phase2.min_condition_observer_r2
                - float(metrics.get("observer_minimum_r2", float("-inf")))
            )
            / max(abs(phase2.min_condition_observer_r2), 1.0e-12),
            0.0,
        )
        if condition_mode == "latent_inferred"
        else 0.0
    )
    observer_admission = (
        0.0
        if condition_mode != "latent_inferred"
        or not phase3.observer_admission_enabled
        or bool(metrics.get("observer_admitted", 0.0))
        else 1.0
    )
    feasibility = (
        physical,
        drift,
        roundtrip,
        gauge_nrmse,
        commutator,
        observer_admission,
        observer_rmse,
        observer_r2,
    )
    feasibility_count = sum(value > 0 for value in feasibility)
    gains = [float(metrics[f"rollout_gain_h{horizon}"]) for horizon in evaluation.rollout_horizons]
    predictive_shortfall = sum(
        max(evaluation.material_relative_gain - gain, 0.0)
        / evaluation.material_relative_gain
        for gain in gains
    )
    decoded_score = sum(
        phase3.lambda_decoded_field
        * float(metrics[f"decoded_field_relative_l2_h{horizon}"])
        + phase3.lambda_decoded_velocity
        * float(metrics[f"decoded_velocity_relative_l2_h{horizon}"])
        + phase3.lambda_decoded_vorticity
        * float(metrics[f"decoded_vorticity_relative_l2_h{horizon}"])
        for horizon in evaluation.rollout_horizons
    ) / (
        len(evaluation.rollout_horizons)
        * (
            phase3.lambda_decoded_field
            + phase3.lambda_decoded_velocity
            + phase3.lambda_decoded_vorticity
        )
    )
    return (
        float(feasibility_count),
        sum(feasibility),
        decoded_score,
        predictive_shortfall,
        -min(gains) / evaluation.material_relative_gain,
        float(metrics["total"]),
    )


def classify_phase3_joint_run(
    metrics: Mapping[str, float],
    phase3: V09Phase3Config,
    evaluation: V09EvaluationConfig,
    phase2: V09Phase2Config,
    *,
    condition_mode: str,
    route: str = "joint",
) -> dict[str, bool]:
    """Apply every declared joint-route gate; no partial fraction is called success."""
    if route not in {"joint", "from_scratch"}:
        raise ValueError("invalid trainable Phase-3 route")
    finite = all(math.isfinite(float(value)) for value in metrics.values())
    physics = finite and float(metrics["physical_manifold_violation"]) <= 1.0e-8
    drift = finite and (
        route == "from_scratch"
        or float(metrics["representation_drift"])
        <= phase3.max_normalized_representation_drift
    )
    roundtrip = finite and float(metrics["roundtrip"]) <= phase3.max_roundtrip_nrmse
    dynamical_gauge = finite and (
        route == "from_scratch"
        or not phase3.physics_aligned_latent_enabled
        or (
            float(metrics.get("dynamical_gauge_nrmse", float("inf")))
            <= phase3.max_dynamical_gauge_nrmse
            and float(metrics.get("generator_commutator", float("inf")))
            <= phase3.max_generator_commutator
        )
    )
    predictive = finite and all(
        float(metrics[f"rollout_gain_h{horizon}"]) >= evaluation.material_relative_gain
        for horizon in evaluation.rollout_horizons
    )
    observer = True
    if condition_mode == "latent_inferred":
        observer = finite and (
            (
                not phase3.observer_admission_enabled
                or bool(metrics.get("observer_admitted", 0.0))
            )
            and
            float(metrics.get("observer_normalized_rmse", float("inf")))
            <= phase2.max_condition_observer_normalized_rmse
            and float(metrics.get("observer_minimum_r2", float("-inf")))
            >= phase2.min_condition_observer_r2
        )
    geometry_names = (
        "representation_linear_cka",
        "representation_procrustes_nrmse",
        "representation_effective_rank",
    )
    geometry_present = all(name in metrics for name in geometry_names)
    geometry_finite = finite and (
        (route == "joint" and not geometry_present)
        or (
            geometry_present
            and all(math.isfinite(float(metrics[name])) for name in geometry_names)
        )
    )
    representation = physics and drift and roundtrip and dynamical_gauge and geometry_finite
    strict_route = representation and predictive and observer
    return {
        "finite": finite,
        "physics": physics,
        "representation_drift": drift,
        "roundtrip": roundtrip,
        "dynamical_gauge": dynamical_gauge,
        "coordinate_invariant_geometry": geometry_finite,
        "representation_feasible": representation,
        "predictive": predictive,
        "observer": observer,
        "strict_route": strict_route,
        "strict_joint": strict_route,
    }


def move_raw_field_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    for name in (
        "history_raw",
        "target_raw",
        "history_dts",
        "future_dts",
        "future_conditions",
        "future_condition_targets",
        "context_parameters",
        "target_latents",
    ):
        if name in batch:
            result[name] = batch[name].to(device=device, dtype=torch.float32)
    result["valid_mask"] = batch["valid_mask"].to(device=device, dtype=torch.bool)
    return result


def _relative_mse(candidate: Tensor, reference: Tensor, floor: float = 1.0e-6) -> Tensor:
    return (candidate - reference).square().mean() / reference.square().mean().clamp_min(floor)


def _mean_relative_l2(
    candidate: Tensor, reference: Tensor, floor: float = 1.0e-12
) -> Tensor:
    if candidate.shape != reference.shape or candidate.ndim < 2:
        raise ValueError("relative L2 requires aligned tensors with an explicit batch axis")
    dimensions = tuple(range(1, candidate.ndim))
    numerator = (candidate - reference).square().sum(dim=dimensions).sqrt()
    denominator = reference.square().sum(dim=dimensions).sqrt().clamp_min(floor)
    return (numerator / denominator).mean()


def decoded_physical_supervision(
    predicted_raw: Tensor,
    target_raw: Tensor,
    valid_mask: Tensor,
    *,
    dx: float,
    dy: float,
) -> dict[str, Tensor]:
    """Dimensionless differentiable supervision in decoded physical coordinates."""
    if (
        predicted_raw.shape != target_raw.shape
        or predicted_raw.ndim != 4
        or predicted_raw.shape[1] < 2
    ):
        raise ValueError("decoded supervision requires aligned [B,C,Nx,Ny] fields")
    if valid_mask.dtype != torch.bool or valid_mask.shape not in {
        predicted_raw.shape[-2:],
        (predicted_raw.shape[0], *predicted_raw.shape[-2:]),
    }:
        raise ValueError("decoded supervision requires an aligned boolean fluid mask")
    predicted_vorticity = central_difference_2d(
        predicted_raw[:, 1], dx, -2
    ) - central_difference_2d(predicted_raw[:, 0], dy, -1)
    target_vorticity = central_difference_2d(
        target_raw[:, 1], dx, -2
    ) - central_difference_2d(target_raw[:, 0], dy, -1)
    predicted_vorticity = predicted_vorticity.masked_fill(~valid_mask, 0.0)
    target_vorticity = target_vorticity.masked_fill(~valid_mask, 0.0)
    return {
        "field": _relative_mse(predicted_raw, target_raw),
        "velocity": _relative_mse(predicted_raw[:, :2], target_raw[:, :2]),
        "vorticity": _relative_mse(predicted_vorticity, target_vorticity),
        "predicted_vorticity": predicted_vorticity,
        "target_vorticity": target_vorticity,
    }


def frozen_decoder_pullback_supervision(
    reference_decoder: nn.Module,
    predicted_latent: Tensor,
    target_latent: Tensor,
    target_raw: Tensor,
    valid_mask: Tensor,
    normalizer: ChannelStandardizer,
    *,
    dx: float,
    dy: float,
) -> dict[str, Tensor]:
    """Measure latent error in the frozen decoder's local physical metric.

    This computes ``J_D(z*) (z_hat-z*)`` by a Jacobian-vector product.  The
    decoder is frozen, so the metric cannot collapse to make a latent error appear
    small.  Channel standardization is removed by its linear tangent map only.
    """
    if predicted_latent.shape != target_latent.shape or predicted_latent.ndim != 2:
        raise ValueError("pullback supervision requires aligned [B,d] latents")
    if target_raw.ndim != 4 or target_raw.shape[0] != predicted_latent.shape[0]:
        raise ValueError("pullback supervision requires aligned physical targets")
    delta = predicted_latent.float() - target_latent.float()
    with torch.autocast(device_type=delta.device.type, enabled=False):
        _, tangent_model = torch.func.jvp(
            reference_decoder,
            (target_latent.float(),),
            (delta,),
        )
        tangent_raw = normalizer.inverse_transform_tangent(tangent_model.float())
    tangent_vorticity = central_difference_2d(
        tangent_raw[:, 1], dx, -2
    ) - central_difference_2d(tangent_raw[:, 0], dy, -1)
    target_vorticity = central_difference_2d(
        target_raw[:, 1], dx, -2
    ) - central_difference_2d(target_raw[:, 0], dy, -1)
    tangent_vorticity = tangent_vorticity.masked_fill(~valid_mask, 0.0)
    target_vorticity = target_vorticity.masked_fill(~valid_mask, 0.0)

    def relative_energy(tangent: Tensor, reference: Tensor) -> Tensor:
        return tangent.square().mean() / reference.square().mean().clamp_min(1.0e-6)

    return {
        "field": relative_energy(tangent_raw, target_raw),
        "velocity": relative_energy(tangent_raw[:, :2], target_raw[:, :2]),
        "vorticity": relative_energy(tangent_vorticity, target_vorticity),
    }


def joint_markov_objective(
    backbone: FieldJEPAKoopmanModel,
    reference_encoder: nn.Module,
    reference_decoder: nn.Module,
    adaptive_model: AdaptiveKoopmanModel,
    batch: dict[str, Any],
    normalizer: ChannelStandardizer,
    residual_scale: Tensor,
    condition_mean: Tensor,
    condition_std: Tensor,
    config: ProjectConfig,
    *,
    epoch: int,
    validation: bool,
    route: str = "joint",
    observer_admitted: bool = True,
) -> JointMarkovObjectiveResult:
    """Joint Koopman/JEPA/round-trip objective with inequality physics gates.

    Physical terms penalize only threshold violations.  Since the entry audit
    established that the inherited reconstruction is already feasible, this
    prevents the representation from sacrificing forecast skill merely to make
    an already-passing divergence value smaller.
    """
    phase3 = config.v0_9_phase3
    phase2 = config.v0_9_phase2
    training = config.v0_9_training
    evaluation = config.v0_9_evaluation
    cylinder = config.cylinder_wake_2d
    adaptive_config = config.v0_9_adaptive
    required = (phase3, phase2, training, evaluation, cylinder, adaptive_config)
    if any(value is None for value in required):
        raise ValueError("joint Phase-3 objective requires complete V0.9 cylinder configuration")
    assert phase3 and phase2 and training and evaluation and cylinder and adaptive_config
    if not phase3.enabled or not phase2.enabled:
        raise ValueError("joint Phase-3 objective requires enabled Phase-2/Phase-3")
    if route not in {"frozen", "joint", "from_scratch"}:
        raise ValueError("invalid Phase-3 objective route")

    history_model = normalizer.transform(batch["history_raw"])
    future_model = normalizer.transform(batch["target_raw"])
    history_z = backbone.encode(history_model)
    with torch.no_grad():
        target_latents = (
            batch["target_latents"]
            if "target_latents" in batch
            else backbone.encode_target(future_model)
        )
        target_current = backbone.encode_target(history_model[:, -1])
        reference_current = reference_encoder(history_model[:, -1]).float()
    curriculum = curriculum_state(training, epoch, validation=validation)
    inferred_without_admission = (
        adaptive_config.condition_mode == "latent_inferred"
        and phase3.observer_admission_enabled
        and not observer_admitted
    )
    phase2_state = Phase2TrainingState(
        name="phase3_joint_refinement",
        active_components="dynamic" if inferred_without_admission else "full",
        train_component="dynamic" if inferred_without_admission else "full",
        use_oracle_condition=False,
        detach_static=False,
        observer_only=False,
        observer_weight=0.0 if phase3.observer_admission_enabled else 1.0,
        delta_budget=phase2.symmetric_delta_budget,
        condition_admitted=not inferred_without_admission,
    )
    adaptive_batch = {
        "history_z": history_z,
        "history_dts": batch["history_dts"],
        "future_dts": batch["future_dts"],
        "future_conditions": batch["future_conditions"],
        "future_condition_targets": batch["future_condition_targets"],
        "target_latents": target_latents,
        "context_parameters": batch["context_parameters"],
    }
    smooth_mask = torch.tensor(
        ["abrupt" not in str(value) for value in batch["schedule_type"]],
        device=history_z.device,
        dtype=torch.bool,
    )
    adaptive = adaptive_stabilization_objective(
        adaptive_model,
        adaptive_batch,
        residual_scale,
        condition_mean,
        condition_std,
        training,
        adaptive_config.condition_mode,
        curriculum,
        smooth_mask,
        phase2,
        phase2_state,
    )

    current = history_z[:, -1]
    reconstructed_model = backbone.decode(current)
    reconstruction = F.smooth_l1_loss(reconstructed_model, history_model[:, -1], beta=1.0)
    reencoded = backbone.encode(reconstructed_model)
    roundtrip = _relative_mse(reencoded, current)
    jepa = _relative_mse(current, target_current)
    representation_drift = _relative_mse(current, reference_current).sqrt()
    absolute_drift_excess = torch.relu(
        representation_drift - phase3.max_normalized_representation_drift
    )
    # A dimensionless inequality penalty makes a 4x tolerance violation O(10)
    # rather than O(1e-1). This keeps the declared 0.10 drift limit meaningful
    # without fitting the penalty to a returned locked-test result.
    drift_excess = torch.relu(
        representation_drift / phase3.max_normalized_representation_drift - 1.0
    ).square()

    physics = current.new_zeros(())
    decoded_field_supervision = current.new_zeros(())
    decoded_velocity_supervision = current.new_zeros(())
    decoded_vorticity_supervision = current.new_zeros(())
    pullback_field = current.new_zeros(())
    pullback_velocity = current.new_zeros(())
    pullback_vorticity = current.new_zeros(())
    decoded_terms: dict[str, Tensor] = {}
    divergence_limit = math.sqrt(evaluation.max_divergence_mse)
    selected_horizons = curriculum.observable_horizons
    selected_weights = curriculum.observable_weights
    for horizon, horizon_weight in zip(
        selected_horizons, selected_weights, strict=True
    ):
        predicted_model = backbone.decode(adaptive.rollout["adapted"][:, horizon - 1])
        predicted_raw = normalizer.inverse_transform(predicted_model.float())
        target_raw = batch["target_raw"][:, horizon - 1]
        metrics = physical_manifold_metrics(
            predicted_raw,
            valid_mask=batch["valid_mask"],
            dx=cylinder.dx,
            dy=cylinder.dy,
            boundary_target=target_raw,
        )
        horizon_physics = (
            torch.relu(metrics["divergence_rms"] - divergence_limit)
            / divergence_limit
        ).square()
        horizon_physics = horizon_physics + (
            torch.relu(metrics["boundary_no_slip_mse"] - evaluation.max_boundary_mse)
            / evaluation.max_boundary_mse
        ).square()
        horizon_physics = horizon_physics + (
            torch.relu(metrics["outer_boundary_mse"] - evaluation.max_boundary_mse)
            / evaluation.max_boundary_mse
        ).square()
        physics = physics + horizon_weight * horizon_physics

        supervised = decoded_physical_supervision(
            predicted_raw,
            target_raw,
            batch["valid_mask"],
            dx=cylinder.dx,
            dy=cylinder.dy,
        )
        predicted_vorticity = supervised["predicted_vorticity"]
        target_vorticity = supervised["target_vorticity"]
        decoded_field_supervision = (
            decoded_field_supervision + horizon_weight * supervised["field"]
        )
        decoded_velocity_supervision = (
            decoded_velocity_supervision
            + horizon_weight * supervised["velocity"]
        )
        decoded_vorticity_supervision = (
            decoded_vorticity_supervision
            + horizon_weight * supervised["vorticity"]
        )
        if phase3.physics_aligned_latent_enabled and route == "joint":
            pullback = frozen_decoder_pullback_supervision(
                reference_decoder,
                adaptive.rollout["adapted"][:, horizon - 1],
                target_latents[:, horizon - 1],
                target_raw,
                batch["valid_mask"],
                normalizer,
                dx=cylinder.dx,
                dy=cylinder.dy,
            )
            pullback_field = pullback_field + horizon_weight * pullback["field"]
            pullback_velocity = pullback_velocity + horizon_weight * pullback["velocity"]
            pullback_vorticity = pullback_vorticity + horizon_weight * pullback["vorticity"]
        if validation:
            decoded_terms.update(
                {
                    f"decoded_field_relative_l2_h{horizon}": _mean_relative_l2(
                        predicted_raw, target_raw
                    ),
                    f"decoded_velocity_relative_l2_h{horizon}": _mean_relative_l2(
                        predicted_raw[:, :2], target_raw[:, :2]
                    ),
                    f"decoded_vorticity_relative_l2_h{horizon}": _mean_relative_l2(
                        predicted_vorticity, target_vorticity
                    ),
                    f"decoded_divergence_rms_h{horizon}": metrics["divergence_rms"],
                    f"decoded_boundary_mse_h{horizon}": metrics[
                        "boundary_no_slip_mse"
                    ],
                    f"decoded_outer_boundary_mse_h{horizon}": metrics[
                        "outer_boundary_mse"
                    ],
                }
            )
    if selected_horizons:
        normalizer_weight = max(curriculum.observable_normalizer, 1.0e-12)
        physics = physics / normalizer_weight
        decoded_field_supervision = decoded_field_supervision / normalizer_weight
        decoded_velocity_supervision = decoded_velocity_supervision / normalizer_weight
        decoded_vorticity_supervision = decoded_vorticity_supervision / normalizer_weight
        pullback_field = pullback_field / normalizer_weight
        pullback_velocity = pullback_velocity / normalizer_weight
        pullback_vorticity = pullback_vorticity / normalizer_weight
    decoded_supervision = (
        phase3.lambda_decoded_field * decoded_field_supervision
        + phase3.lambda_decoded_velocity * decoded_velocity_supervision
        + phase3.lambda_decoded_vorticity * decoded_vorticity_supervision
    )
    pullback_supervision = (
        phase3.lambda_decoded_field * pullback_field
        + phase3.lambda_decoded_velocity * pullback_velocity
        + phase3.lambda_decoded_vorticity * pullback_vorticity
    )

    total = (
        adaptive.total
        + phase3.lambda_reconstruction * reconstruction
        + phase3.lambda_roundtrip * roundtrip
        + phase3.lambda_jepa_consistency * jepa
        + curriculum.physics_scale
        * (
            phase3.lambda_physical_manifold * physics
            + decoded_supervision
            + phase3.lambda_physical_pullback * pullback_supervision
        )
        + (
            phase3.lambda_representation_drift * drift_excess
            if route == "joint"
            else current.new_zeros(())
        )
    )
    terms = {
        **adaptive.terms,
        "reconstruction": reconstruction,
        "roundtrip": roundtrip.sqrt(),
        "jepa_consistency": jepa,
        "representation_drift": representation_drift,
        "representation_drift_excess": drift_excess,
        "representation_drift_absolute_excess": absolute_drift_excess,
        "physical_manifold_violation": physics,
        "decoded_field_supervision": decoded_field_supervision,
        "decoded_velocity_supervision": decoded_velocity_supervision,
        "decoded_vorticity_supervision": decoded_vorticity_supervision,
        "decoded_supervision": decoded_supervision,
        "decoded_supervision_scale": current.new_tensor(curriculum.physics_scale),
        "pullback_field_supervision": pullback_field,
        "pullback_velocity_supervision": pullback_velocity,
        "pullback_vorticity_supervision": pullback_vorticity,
        "pullback_supervision": pullback_supervision,
        "observer_admitted": current.new_tensor(float(observer_admitted)),
        **decoded_terms,
    }
    if not torch.isfinite(total):
        raise FloatingPointError("Phase-3 joint objective became non-finite")
    return JointMarkovObjectiveResult(total, terms, adaptive)
