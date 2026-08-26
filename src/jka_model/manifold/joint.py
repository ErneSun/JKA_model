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
from jka_model.config import ProjectConfig
from jka_model.data import ChannelStandardizer
from jka_model.data.datasets import TrajectoryDataset
from jka_model.manifold.physical import physical_manifold_metrics
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


def joint_markov_objective(
    backbone: FieldJEPAKoopmanModel,
    reference_encoder: nn.Module,
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
    phase2_state = Phase2TrainingState(
        name="phase3_joint_refinement",
        active_components="full",
        train_component="full",
        use_oracle_condition=False,
        detach_static=False,
        observer_only=False,
        observer_weight=1.0,
        delta_budget=phase2.symmetric_delta_budget,
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
    drift_excess = torch.relu(
        representation_drift - phase3.max_normalized_representation_drift
    ).square()

    physics = current.new_zeros(())
    divergence_limit = math.sqrt(evaluation.max_divergence_mse)
    selected_horizons = curriculum.observable_horizons or curriculum.active_horizons or (1,)
    for horizon in selected_horizons:
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
        physics = physics + (
            torch.relu(metrics["divergence_rms"] - divergence_limit)
            / divergence_limit
        ).square()
        physics = physics + (
            torch.relu(metrics["boundary_no_slip_mse"] - evaluation.max_boundary_mse)
            / evaluation.max_boundary_mse
        ).square()
        physics = physics + (
            torch.relu(metrics["outer_boundary_mse"] - evaluation.max_boundary_mse)
            / evaluation.max_boundary_mse
        ).square()
    physics = physics / len(selected_horizons)

    total = (
        adaptive.total
        + phase3.lambda_reconstruction * reconstruction
        + phase3.lambda_roundtrip * roundtrip
        + phase3.lambda_jepa_consistency * jepa
        + phase3.lambda_physical_manifold * physics
        + phase3.lambda_representation_drift * drift_excess
    )
    terms = {
        **adaptive.terms,
        "reconstruction": reconstruction,
        "roundtrip": roundtrip.sqrt(),
        "jepa_consistency": jepa,
        "representation_drift": representation_drift,
        "representation_drift_excess": drift_excess,
        "physical_manifold_violation": physics,
    }
    if not torch.isfinite(total):
        raise FloatingPointError("Phase-3 joint objective became non-finite")
    return JointMarkovObjectiveResult(total, terms, adaptive)
