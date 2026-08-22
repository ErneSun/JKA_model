"""Frozen-decoder bridge from adaptive latents to problem-owned observables."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from jka_model.config import ProjectConfig
from jka_model.data import ChannelStandardizer
from jka_model.data.datasets import TrajectoryRecord
from jka_model.observables import (
    ObservableLossResult,
    ObservableObjective,
    RobustObservableScaleState,
)
from jka_model.problems import create_observable_problem_adapter
from jka_model.problems.cylinder_observables import CylinderWakeObservableObjective
from jka_model.residual.cache import file_sha256
from jka_model.utils import load_checkpoint

PhysicalLossResult = ObservableLossResult


class FrozenDecoderObservables:
    """Decode with frozen weights while gradients continue to adaptive latent inputs."""

    def __init__(
        self,
        model: nn.Module,
        normalizer: ChannelStandardizer,
        records: dict[str, TrajectoryRecord],
        objective: ObservableObjective,
        device: torch.device,
    ) -> None:
        if not hasattr(model, "decode"):
            raise TypeError("observable bridge requires a model.decode method")
        self.model = model
        self.normalizer = normalizer
        self.records = records
        self.objective = objective
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
    ) -> FrozenDecoderObservables:
        from train.train_v0_6 import initialize_v0_6_model

        if file_sha256(backbone_checkpoint) != expected_backbone_sha256:
            raise ValueError("V0.9 observable trainer/backbone fingerprint mismatch")
        saved = load_checkpoint(backbone_checkpoint, map_location="cpu")
        if saved.online_model_state is None or saved.target_model_state is None:
            raise ValueError("V0.9 observable trainer requires a JEPA backbone checkpoint")
        if saved.normalizer_state is None or saved.config is None:
            raise ValueError("V0.9 observable trainer lacks normalization provenance")
        for name in ("koopman", "field_autoencoder", "field_loss", "jepa_loss", "ema"):
            inherited = getattr(saved.config, name)
            current = getattr(config, name)
            if inherited is None or current is None or inherited.to_dict() != current.to_dict():
                raise ValueError(f"V0.9 observable inheritance mismatch in {name}")
        model = initialize_v0_6_model(config, device=device)
        model.load_online_state_dict(saved.online_model_state)
        model.target_encoder.load_state_dict(saved.target_model_state, strict=True)
        model.requires_grad_(False)
        model.eval()
        normalizer = ChannelStandardizer(eps=config.data.normalization.eps)
        normalizer.load_state_dict(saved.normalizer_state)
        adapter = create_observable_problem_adapter(config)
        configured_source = getattr(getattr(adapter, "config", None), "dataset_path", None)
        if (
            configured_source
            and Path(configured_source).resolve() != Path(physical_dataset).resolve()
        ):
            raise ValueError("V0.9 observable dataset/config source mismatch")
        records = {
            record.trajectory_id: record
            for record in adapter.build_dataset(seed=config.training.seed)
        }
        objective = adapter.build_observable_objective(
            training=config.v0_9_training,
            evaluation=config.v0_9_evaluation,
        )
        return cls(model, normalizer, records, objective, device)

    def target_batch(
        self,
        trajectory_ids: list[str] | tuple[str, ...],
        target_indices: Tensor,
        horizon: int,
        limit: int,
    ) -> tuple[Tensor, dict[str, Any]]:
        states: list[Tensor] = []
        masks: list[Tensor] = []
        record_metadata: list[Mapping[str, Any]] = []
        for trajectory_id, target_index in zip(
            trajectory_ids[:limit], target_indices[:limit].tolist(), strict=True
        ):
            record = self.records.get(str(trajectory_id))
            if record is None:
                raise ValueError(f"missing observable record for {trajectory_id}")
            state_index = int(target_index) + horizon
            if state_index >= record.states_raw.shape[0]:
                raise ValueError("observable target lies beyond its trajectory")
            states.append(record.states_raw[state_index])
            if record.valid_mask is not None:
                masks.append(record.valid_mask)
            record_metadata.append(record.metadata)
        metadata: dict[str, Any] = {"records": record_metadata}
        if masks:
            if len(masks) != len(states):
                raise ValueError("observable masks must be present for every selected record")
            metadata["valid_mask"] = torch.stack(masks).to(
                self.device, dtype=torch.bool
            )
        return torch.stack(states).to(self.device, dtype=torch.float32), metadata

    def target_sequence(
        self,
        trajectory_ids: list[str] | tuple[str, ...],
        target_indices: Tensor,
        steps: tuple[int, ...],
        limit: int,
    ) -> tuple[Tensor, dict[str, Any]]:
        if not steps or tuple(sorted(set(steps))) != steps or steps[0] < 1:
            raise ValueError("observable sequence steps must be positive and increasing")
        sequences: list[Tensor] = []
        record_metadata: list[Mapping[str, Any]] = []
        for trajectory_id, target_index in zip(
            trajectory_ids[:limit], target_indices[:limit].tolist(), strict=True
        ):
            record = self.records.get(str(trajectory_id))
            if record is None:
                raise ValueError(f"missing observable record for {trajectory_id}")
            indices = torch.tensor(
                [int(target_index) + step for step in steps], dtype=torch.long
            )
            if int(indices[-1]) >= record.states_raw.shape[0]:
                raise ValueError("observable sequence lies beyond its trajectory")
            sequences.append(record.states_raw.index_select(0, indices))
            record_metadata.append(record.metadata)
        return (
            torch.stack(sequences).to(self.device, dtype=torch.float32),
            {"records": record_metadata, "sequence_steps": list(steps)},
        )

    def fit_training_scales(
        self,
        trajectory_ids: tuple[str, ...],
        *,
        split_fingerprint: str,
    ) -> RobustObservableScaleState:
        fitter = getattr(self.objective, "fit_training_scales", None)
        if fitter is None:
            raise TypeError("observable objective does not implement train-only scale fitting")
        return fitter(
            self.records,
            trajectory_ids,
            split_fingerprint=split_fingerprint,
        )

    def set_scale_state(self, state: RobustObservableScaleState) -> None:
        setter = getattr(self.objective, "set_scale_state", None)
        if setter is None:
            raise TypeError("observable objective does not accept robust scale state")
        setter(state)

    def decode(self, predicted_latent: Tensor) -> Tensor:
        leading = predicted_latent.shape[:-1]
        flattened = predicted_latent.reshape(-1, predicted_latent.shape[-1])
        predicted_model = self.model.decode(flattened)  # type: ignore[attr-defined]
        with torch.autocast(device_type=predicted_model.device.type, enabled=False):
            raw = self.normalizer.inverse_transform(predicted_model.float())
        return raw.reshape(*leading, *raw.shape[1:])

    def loss(
        self,
        predicted_latent: Tensor,
        target_raw: Tensor,
        metadata: Mapping[str, Any],
    ) -> ObservableLossResult:
        return self.objective.training_loss(
            self.decode(predicted_latent),
            target_raw.float(),
            metadata,
        )

    def force_window_loss(
        self,
        predicted_latent: Tensor,
        target_raw: Tensor,
        metadata: Mapping[str, Any],
    ) -> ObservableLossResult:
        objective = getattr(self.objective, "force_window_loss", None)
        if objective is None:
            raise TypeError("observable objective does not implement a force-window loss")
        return objective(self.decode(predicted_latent), target_raw.float(), metadata)


class FrozenCylinderPhysics(FrozenDecoderObservables):
    """Backward-compatible constructor; new code uses FrozenDecoderObservables."""

    def __init__(
        self,
        model: nn.Module,
        normalizer: ChannelStandardizer,
        records: dict[str, TrajectoryRecord],
        config: ProjectConfig,
        device: torch.device,
    ) -> None:
        if config.cylinder_wake_2d is None:
            raise ValueError("V0.9 cylinder observables require the cylinder configuration")
        objective = CylinderWakeObservableObjective(
            config.cylinder_wake_2d,
            config.v0_9_training,
            config.v0_9_evaluation,
        )
        super().__init__(model, normalizer, records, objective, device)
