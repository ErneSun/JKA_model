"""Raw-field Phase-3 reconstruction and tangent-space audit."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from jka_model.config import ProjectConfig
from jka_model.data import ChannelStandardizer, SplitManifest, load_cylinder_wake_dataset
from jka_model.manifold.physical import physical_manifold_metrics
from jka_model.utils import load_checkpoint


@dataclass(frozen=True, slots=True)
class Phase3RepresentationAudit:
    backbone_seed: int
    sample_count: int
    reconstruction_relative_l2: float
    roundtrip_nrmse: float
    data_divergence_rms: float
    reconstruction_divergence_rms: float
    divergence_degradation: float
    data_boundary_no_slip_mse: float
    reconstruction_boundary_no_slip_mse: float
    reconstruction_outer_boundary_mse: float
    boundary_degradation: float
    nominal_tangent_divergence: float
    nominal_tangent_boundary: float
    nominal_tangent_outer_boundary: float
    reconstruction_physics_status: str
    roundtrip_status: str
    tangent_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _relative_degradation(candidate: Tensor, reference: Tensor, floor: float = 1.0e-12) -> Tensor:
    return (candidate - reference) / reference.abs().clamp_min(floor)


def _sample_indices(length: int, count: int) -> Tensor:
    if length < 1 or count < 1:
        raise ValueError("Phase-3 sampling dimensions must be positive")
    return torch.linspace(0, length - 1, min(length, count)).round().long().unique()


@torch.no_grad()
def audit_representation_checkpoint(
    config: ProjectConfig,
    *,
    backbone_checkpoint: str | Path,
    physical_dataset: str | Path,
    split_manifest: Mapping[str, Any] | None = None,
    device: str | torch.device,
    output_path: str | Path | None = None,
) -> Phase3RepresentationAudit:
    """Audit reconstruction, round-trip and nominal tangent consistency on locked test data."""
    from train.train_v0_6 import initialize_v0_6_model

    phase3 = config.v0_9_phase3
    cylinder = config.cylinder_wake_2d
    if phase3 is None or not phase3.enabled or cylinder is None:
        raise ValueError("Phase-3 representation audit requires enabled Phase-3/cylinder config")
    selected = torch.device(device)
    saved = load_checkpoint(backbone_checkpoint, map_location="cpu")
    if (
        saved.online_model_state is None
        or saved.target_model_state is None
        or saved.normalizer_state is None
        or not isinstance(saved.split_manifest, dict)
    ):
        raise ValueError("Phase-3 audit backbone lacks representation provenance")
    model = initialize_v0_6_model(config, device=selected)
    model.load_online_state_dict(saved.online_model_state)
    model.target_encoder.load_state_dict(saved.target_model_state, strict=True)
    model.requires_grad_(False)
    model.eval()
    normalizer = ChannelStandardizer(eps=config.data.normalization.eps)
    normalizer.load_state_dict(saved.normalizer_state)
    dataset = load_cylinder_wake_dataset(physical_dataset, cylinder)
    manifest = SplitManifest.from_dict(
        saved.split_manifest if split_manifest is None else split_manifest
    )
    test_ids = set(manifest.test)
    records = [record for record in dataset.records if record.trajectory_id in test_ids]
    if not records:
        raise ValueError("Phase-3 audit has no locked-test trajectories")

    reconstruction_errors: list[Tensor] = []
    roundtrip_errors: list[Tensor] = []
    data_divergence: list[Tensor] = []
    reconstruction_divergence: list[Tensor] = []
    data_boundary: list[Tensor] = []
    reconstruction_boundary: list[Tensor] = []
    reconstruction_outer_boundary: list[Tensor] = []
    tangent_divergence: list[Tensor] = []
    tangent_boundary: list[Tensor] = []
    tangent_outer_boundary: list[Tensor] = []
    sample_count = 0
    epsilon = phase3.tangent_epsilon
    for record in records:
        if record.valid_mask is None:
            raise ValueError("Phase-3 cylinder audit requires a valid fluid mask")
        indices = _sample_indices(record.states_raw.shape[0], phase3.audit_samples_per_trajectory)
        raw = record.states_raw.index_select(0, indices).to(selected, dtype=torch.float32)
        mask = record.valid_mask.to(selected, dtype=torch.bool)
        normalized = normalizer.transform(raw)
        latent = model.encode(normalized)
        reconstructed_model = model.decode(latent)
        reconstructed = normalizer.inverse_transform(reconstructed_model.float())
        reencoded = model.encode(normalizer.transform(reconstructed))
        reconstruction_errors.append(
            (reconstructed - raw).square().sum().sqrt() / raw.square().sum().sqrt().clamp_min(1e-12)
        )
        latent_scale = latent.square().mean().sqrt().clamp_min(1e-12)
        roundtrip_errors.append((reencoded - latent).square().mean().sqrt() / latent_scale)
        target_metrics = physical_manifold_metrics(
            raw, valid_mask=mask, dx=cylinder.dx, dy=cylinder.dy
        )
        reconstruction_metrics = physical_manifold_metrics(
            reconstructed,
            valid_mask=mask,
            dx=cylinder.dx,
            dy=cylinder.dy,
            boundary_target=raw,
        )
        data_divergence.append(target_metrics["divergence_rms"])
        reconstruction_divergence.append(reconstruction_metrics["divergence_rms"])
        data_boundary.append(target_metrics["boundary_no_slip_mse"])
        reconstruction_boundary.append(reconstruction_metrics["boundary_no_slip_mse"])
        reconstruction_outer_boundary.append(reconstruction_metrics["outer_boundary_mse"])

        # Directional physical derivative along the inherited nominal generator.
        vector = torch.einsum("ij,bj->bi", model.koopman_core.A.float(), latent.float())
        perturbed = latent + epsilon * vector
        perturbed_raw = normalizer.inverse_transform(model.decode(perturbed).float())
        perturbed_metrics = physical_manifold_metrics(
            perturbed_raw,
            valid_mask=mask,
            dx=cylinder.dx,
            dy=cylinder.dy,
            boundary_target=reconstructed,
        )
        tangent_divergence.append(
            (
                perturbed_metrics["divergence_rms"]
                - reconstruction_metrics["divergence_rms"]
            ).abs()
            / epsilon
        )
        tangent_boundary.append(
            (
                perturbed_metrics["boundary_no_slip_mse"]
                - reconstruction_metrics["boundary_no_slip_mse"]
            ).abs()
            / epsilon
        )
        tangent_outer_boundary.append(
            perturbed_metrics["outer_boundary_mse"].sqrt() / epsilon
        )
        sample_count += raw.shape[0]

    def mean(values: list[Tensor]) -> Tensor:
        return torch.stack([value.detach().float().cpu() for value in values]).mean()

    reconstruction_relative_l2 = mean(reconstruction_errors)
    roundtrip_nrmse = mean(roundtrip_errors)
    data_divergence_rms = mean(data_divergence)
    reconstruction_divergence_rms = mean(reconstruction_divergence)
    data_boundary_mse = mean(data_boundary)
    reconstruction_boundary_mse = mean(reconstruction_boundary)
    reconstruction_outer_boundary_mse = mean(reconstruction_outer_boundary)
    assert config.v0_9_evaluation is not None
    divergence_degradation = _relative_degradation(
        reconstruction_divergence_rms,
        data_divergence_rms,
        floor=config.v0_9_evaluation.max_divergence_mse,
    )
    boundary_degradation = _relative_degradation(
        reconstruction_boundary_mse,
        data_boundary_mse,
        floor=config.v0_9_evaluation.max_boundary_mse,
    )
    nominal_tangent_divergence = mean(tangent_divergence)
    nominal_tangent_boundary = mean(tangent_boundary)
    nominal_tangent_outer_boundary = mean(tangent_outer_boundary)
    reconstruction_pass = bool(
        divergence_degradation <= phase3.max_reconstruction_physics_degradation
        and boundary_degradation <= phase3.max_reconstruction_physics_degradation
        and reconstruction_divergence_rms <= config.v0_9_evaluation.max_divergence_mse
        and reconstruction_boundary_mse <= config.v0_9_evaluation.max_boundary_mse
        and reconstruction_outer_boundary_mse <= config.v0_9_evaluation.max_boundary_mse
    )
    roundtrip_pass = bool(roundtrip_nrmse <= phase3.max_roundtrip_nrmse)
    tangent_pass = bool(nominal_tangent_divergence <= phase3.max_tangent_divergence)
    result = Phase3RepresentationAudit(
        backbone_seed=config.training.seed,
        sample_count=sample_count,
        reconstruction_relative_l2=float(reconstruction_relative_l2),
        roundtrip_nrmse=float(roundtrip_nrmse),
        data_divergence_rms=float(data_divergence_rms),
        reconstruction_divergence_rms=float(reconstruction_divergence_rms),
        divergence_degradation=float(divergence_degradation),
        data_boundary_no_slip_mse=float(data_boundary_mse),
        reconstruction_boundary_no_slip_mse=float(reconstruction_boundary_mse),
        reconstruction_outer_boundary_mse=float(reconstruction_outer_boundary_mse),
        boundary_degradation=float(boundary_degradation),
        nominal_tangent_divergence=float(nominal_tangent_divergence),
        nominal_tangent_boundary=float(nominal_tangent_boundary),
        nominal_tangent_outer_boundary=float(nominal_tangent_outer_boundary),
        reconstruction_physics_status="PASS" if reconstruction_pass else "FAIL",
        roundtrip_status="PASS" if roundtrip_pass else "FAIL",
        tangent_status="PASS" if tangent_pass else "FAIL",
    )
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result
