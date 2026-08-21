"""Frozen-decoder physical anchors for adaptive-operator training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from jka_model.config import ProjectConfig, V09TrainingConfig
from jka_model.data import (
    ChannelStandardizer,
    load_cylinder_wake_dataset,
    velocity_vorticity_divergence,
)
from jka_model.data.datasets import TrajectoryRecord
from jka_model.models import FieldJEPAKoopmanModel
from jka_model.residual.cache import file_sha256
from jka_model.utils import load_checkpoint


@dataclass(slots=True)
class PhysicalLossResult:
    total: Tensor
    terms: dict[str, Tensor]


class FrozenCylinderPhysics:
    """Own frozen decode/provenance state while retaining gradients to latent inputs."""

    def __init__(
        self,
        model: FieldJEPAKoopmanModel,
        normalizer: ChannelStandardizer,
        records: dict[str, TrajectoryRecord],
        config: ProjectConfig,
        device: torch.device,
    ) -> None:
        if config.cylinder_wake_2d is None:
            raise ValueError("V0.9 frozen physics requires the cylinder configuration")
        self.model = model
        self.normalizer = normalizer
        self.records = records
        self.config = config.cylinder_wake_2d
        self.device = device

    @classmethod
    def from_artifacts(
        cls,
        config: ProjectConfig,
        *,
        backbone_checkpoint: str | Path,
        physical_dataset: str | Path,
        expected_backbone_sha256: str,
        device: torch.device,
    ) -> FrozenCylinderPhysics:
        # Local import keeps the reusable physics objective independent of the
        # train package's public V0.9 imports.
        from train.train_v0_6 import initialize_v0_6_model

        if file_sha256(backbone_checkpoint) != expected_backbone_sha256:
            raise ValueError("V0.9 physical trainer/backbone fingerprint mismatch")
        saved = load_checkpoint(backbone_checkpoint, map_location="cpu")
        if saved.online_model_state is None or saved.target_model_state is None:
            raise ValueError("V0.9 physical trainer requires a JEPA backbone checkpoint")
        if saved.normalizer_state is None or saved.config is None:
            raise ValueError("V0.9 physical trainer lacks backbone normalization provenance")
        for name in ("koopman", "field_autoencoder", "field_loss", "jepa_loss", "ema"):
            inherited = getattr(saved.config, name)
            current = getattr(config, name)
            if inherited is None or current is None or inherited.to_dict() != current.to_dict():
                raise ValueError(f"V0.9 physical trainer inheritance mismatch in {name}")
        model = initialize_v0_6_model(config, device=device)
        model.load_online_state_dict(saved.online_model_state)
        model.target_encoder.load_state_dict(saved.target_model_state, strict=True)
        model.requires_grad_(False)
        model.eval()
        normalizer = ChannelStandardizer(eps=config.data.normalization.eps)
        normalizer.load_state_dict(saved.normalizer_state)
        if config.cylinder_wake_2d is None:
            raise ValueError("V0.9 physical trainer lacks cylinder config")
        dataset = load_cylinder_wake_dataset(physical_dataset, config.cylinder_wake_2d)
        records = {record.trajectory_id: record for record in dataset.records}
        return cls(model, normalizer, records, config, device)

    def target_batch(
        self,
        trajectory_ids: list[str] | tuple[str, ...],
        target_indices: Tensor,
        horizon: int,
        limit: int,
    ) -> tuple[Tensor, Tensor]:
        states: list[Tensor] = []
        masks: list[Tensor] = []
        for trajectory_id, target_index in zip(
            trajectory_ids[:limit], target_indices[:limit].tolist(), strict=True
        ):
            record = self.records.get(str(trajectory_id))
            if record is None:
                raise ValueError(f"missing physical record for {trajectory_id}")
            state_index = int(target_index) + horizon
            if state_index >= record.states_raw.shape[0]:
                raise ValueError("physical target lies beyond its trajectory")
            if record.valid_mask is None:
                raise ValueError("physical target lacks a cylinder valid mask")
            states.append(record.states_raw[state_index])
            masks.append(record.valid_mask)
        return (
            torch.stack(states).to(self.device, dtype=torch.float32),
            torch.stack(masks).to(self.device, dtype=torch.bool),
        )

    def loss(
        self,
        predicted_latent: Tensor,
        target_raw: Tensor,
        valid_mask: Tensor,
        weights: V09TrainingConfig,
    ) -> PhysicalLossResult:
        predicted_model = self.model.decode(predicted_latent)
        # Spatial derivatives and relative-energy denominators are evaluated in
        # FP32 even when the frozen decoder runs under BF16 autocast.
        with torch.autocast(device_type=predicted_model.device.type, enabled=False):
            predicted_raw = self.normalizer.inverse_transform(predicted_model.float())
            target_raw = target_raw.float()
        fluid = valid_mask.unsqueeze(1).to(predicted_raw.dtype)
        solid = (~valid_mask).unsqueeze(1).to(predicted_raw.dtype)
        velocity_error = (
            (predicted_raw[:, :2] - target_raw[:, :2]).square() * fluid
        ).sum() / (target_raw[:, :2].square() * fluid).sum().clamp_min(1e-12)
        predicted_vorticity, predicted_divergence = velocity_vorticity_divergence(
            predicted_raw, self.config
        )
        target_vorticity, _ = velocity_vorticity_divergence(target_raw, self.config)
        fluid_scalar = valid_mask.to(predicted_raw.dtype)
        vorticity = (
            (predicted_vorticity - target_vorticity).square() * fluid_scalar
        ).sum() / (target_vorticity.square() * fluid_scalar).sum().clamp_min(1e-12)
        reference_gradient = (
            target_vorticity.square() * fluid_scalar
        ).sum().div(fluid_scalar.sum().clamp_min(1.0))
        divergence = (
            predicted_divergence.square() * fluid_scalar
        ).sum().div(fluid_scalar.sum().clamp_min(1.0)) / reference_gradient.clamp_min(1e-12)
        reference_velocity = (
            target_raw[:, :2].square() * fluid
        ).sum().div(fluid.sum().clamp_min(1.0))
        boundary = (
            predicted_raw[:, :2].square() * solid
        ).sum().div(solid.sum().clamp_min(1.0)) / reference_velocity.clamp_min(1e-12)
        total = (
            weights.physics_velocity_weight * velocity_error
            + weights.physics_vorticity_weight * vorticity
            + weights.physics_divergence_weight * divergence
            + weights.physics_boundary_weight * boundary
        )
        return PhysicalLossResult(
            total,
            {
                "physics_velocity": velocity_error,
                "physics_vorticity": vorticity,
                "physics_divergence": divergence,
                "physics_boundary": boundary,
            },
        )
