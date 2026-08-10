"""V0.5 field representation losses with differentiable raw-unit physics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from jka_model.config import FieldLossConfig
from jka_model.contracts import ProblemBatch, ProblemSpec
from jka_model.data import ChannelStandardizer
from jka_model.losses.koopman import variance_loss
from jka_model.models import FieldKoopmanAutoencoder
from jka_model.physics import PhysicsConstraint


@dataclass(frozen=True, slots=True)
class FieldLossBreakdown:
    total: Tensor
    koopman_one_step: Tensor
    koopman_multi_step: Tensor
    reconstruction: Tensor
    forecast_model: Tensor
    variance: Tensor
    mass: Tensor
    operator: Tensor
    physics_scale: float

    def as_scalars(self) -> dict[str, float]:
        return {
            "total_loss": float(self.total.detach()),
            "koopman_one_step_loss": float(self.koopman_one_step.detach()),
            "koopman_multi_step_loss": float(self.koopman_multi_step.detach()),
            "reconstruction_loss": float(self.reconstruction.detach()),
            "forecast_model_mse": float(self.forecast_model.detach()),
            "variance_loss": float(self.variance.detach()),
            "mass_loss": float(self.mass.detach()),
            "operator_loss": float(self.operator.detach()),
            "physics_scale": self.physics_scale,
        }


def compute_field_koopman_loss(
    model: FieldKoopmanAutoencoder,
    batch: ProblemBatch,
    normalizer: ChannelStandardizer,
    spec: ProblemSpec,
    config: FieldLossConfig,
    constraints: Mapping[str, PhysicsConstraint],
    *,
    physics_scale: float,
) -> FieldLossBreakdown:
    """Compute closed-loop latent/field losses and raw-space physics penalties."""
    if not 0 <= physics_scale <= 1:
        raise ValueError("physics_scale must lie in [0,1]")
    current = batch.context_states_model[:, -1]
    future = batch.future_states_model
    if current.ndim != 4 or future.ndim != 5 or future.shape[1] < 2:
        raise ValueError("V0.5 requires [B,C,Nx,Ny] current and [B,H,C,Nx,Ny] future")
    sequence = torch.cat((current.unsqueeze(1), future), dim=1)
    latent_targets = model.encode(sequence)
    latent_rollout = model.core.rollout(latent_targets[:, 0], batch.future_dts)
    predicted_model = model.decode(latent_rollout[:, 1:])
    reconstructed_model = model.decode(latent_targets)
    one_step = (latent_rollout[:, 1] - latent_targets[:, 1]).square().mean()
    multi_step = (latent_rollout[:, 1:] - latent_targets[:, 1:]).square().mean()
    reconstruction = (reconstructed_model - sequence).square().mean()
    forecast_model = (predicted_model - future).square().mean()
    variance = variance_loss(latent_targets, min_std=config.min_std)

    # AMP is useful for the CNNs, but raw-unit differences, quadrature, and reductions
    # remain FP32 to avoid half-precision cancellation and overflow.
    with torch.autocast(device_type=predicted_model.device.type, enabled=False):
        predicted_raw = normalizer.inverse_transform(predicted_model.float())
        metadata = {
            "mu_static": batch.mu_static,
            "cell_weights": batch.cell_weights,
            "coordinates": batch.coordinates,
            "valid_mask": batch.valid_mask,
        }
        try:
            mass_constraint = constraints["mass"]
            operator_constraint = constraints["operator"]
        except KeyError as error:
            raise ValueError("V0.5 requires named 'mass' and 'operator' constraints") from error
        mass_terms: list[Tensor] = []
        operator_terms: list[Tensor] = []
        previous = batch.context_states_raw[:, -1].float()
        for index in range(predicted_raw.shape[1]):
            prediction = predicted_raw[:, index]
            mass_result = mass_constraint.loss(
                prediction, prev_state_raw=previous, metadata=metadata
            )
            operator_result = operator_constraint.loss(
                prediction,
                prev_state_raw=previous,
                dt=batch.future_dts[:, index].float(),
                spec=spec,
                metadata=metadata,
            )
            if len(mass_result) != 1 or len(operator_result) != 1:
                raise ValueError("V0.5 named constraints must each return one scalar penalty")
            mass_terms.append(next(iter(mass_result.values())))
            operator_terms.append(next(iter(operator_result.values())))
            previous = prediction
        mass = torch.stack(mass_terms).mean()
        operator = torch.stack(operator_terms).mean()
    physics = config.lambda_mass * mass + config.lambda_operator * operator
    total = (
        config.lambda_k * one_step
        + config.lambda_multi * multi_step
        + config.lambda_rec * reconstruction
        + config.lambda_var * variance
        + physics_scale * config.lambda_physics * physics
    )
    return FieldLossBreakdown(
        total,
        one_step,
        multi_step,
        reconstruction,
        forecast_model,
        variance,
        mass,
        operator,
        physics_scale,
    )
