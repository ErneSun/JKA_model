"""Minimal Koopman autoencoder losses for V0.4 online representation learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from jka_model.contracts import ProblemBatch
from jka_model.models import ContinuousKoopmanCore, KoopmanAutoencoder

if TYPE_CHECKING:
    from jka_model.config import RepresentationLossConfig


def koopman_one_step_loss(
    core: ContinuousKoopmanCore,
    z_current: Tensor,
    z_next_target: Tensor,
    dt: Tensor,
) -> Tensor:
    """Online-encoder consistency for one exact matrix-exponential step."""
    if z_current.shape != z_next_target.shape or z_current.ndim != 2:
        raise ValueError("one-step latents must share shape [B,d]")
    if dt.shape != (z_current.shape[0],):
        raise ValueError("one-step dt must have shape [B]")
    return (core.step(z_current, dt) - z_next_target).square().mean()


def koopman_multi_step_loss(
    core: ContinuousKoopmanCore,
    z_current: Tensor,
    z_future_targets: Tensor,
    future_dts: Tensor,
) -> Tensor:
    """Closed-loop latent rollout loss; truth is used only as the final target tensor."""
    if z_current.ndim != 2 or z_future_targets.ndim != 3:
        raise ValueError("multi-step latents must have shapes [B,d] and [B,H,d]")
    if z_future_targets.shape[0] != z_current.shape[0]:
        raise ValueError("multi-step latent batch sizes must match")
    if z_future_targets.shape[2] != z_current.shape[1]:
        raise ValueError("multi-step latent dimensions must match")
    if future_dts.shape != z_future_targets.shape[:2]:
        raise ValueError("future_dts must have shape [B,H]")
    prediction = core.rollout(z_current, future_dts)[:, 1:]
    return (prediction - z_future_targets).square().mean()


def reconstruction_loss(decoded_model: Tensor, target_model: Tensor) -> Tensor:
    if decoded_model.shape != target_model.shape:
        raise ValueError("decoded and target model states must have identical shapes")
    return (decoded_model - target_model).square().mean()


def variance_loss(z_k: Tensor, *, min_std: float) -> Tensor:
    """Penalize collapsed dimensions using stable population standard deviation."""
    if z_k.ndim < 2 or z_k.shape[-1] < 1:
        raise ValueError("z_k must have at least a sample and latent dimension")
    if min_std <= 0:
        raise ValueError("min_std must be positive")
    flattened = z_k.reshape(-1, z_k.shape[-1])
    std = flattened.std(dim=0, unbiased=False)
    threshold = torch.as_tensor(min_std, dtype=z_k.dtype, device=z_k.device)
    return torch.relu(threshold - std).square().mean()


def stability_regularizer(
    generator: Tensor,
    *,
    margin: float = 0.0,
) -> Tensor:
    """Optional differentiable logarithmic-norm penalty, not an eig-spectrum loss."""
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError("generator must have shape [d,d]")
    symmetric = 0.5 * (generator + generator.transpose(-1, -2))
    maximum = torch.linalg.eigvalsh(symmetric).max()
    return torch.relu(maximum - margin).square()


@dataclass(frozen=True, slots=True)
class RepresentationLossBreakdown:
    total: Tensor
    koopman_one_step: Tensor
    koopman_multi_step: Tensor
    reconstruction: Tensor
    variance: Tensor
    spectral: Tensor

    def as_scalars(self) -> dict[str, float]:
        return {
            "total_loss": float(self.total.detach().item()),
            "koopman_one_step_loss": float(self.koopman_one_step.detach().item()),
            "koopman_multi_step_loss": float(self.koopman_multi_step.detach().item()),
            "reconstruction_loss": float(self.reconstruction.detach().item()),
            "variance_loss": float(self.variance.detach().item()),
            "spectral_loss": float(self.spectral.detach().item()),
        }


def compute_representation_loss(
    model: KoopmanAutoencoder,
    batch: ProblemBatch,
    config: RepresentationLossConfig,
) -> RepresentationLossBreakdown:
    """Compute the complete V0.4 loss from canonical model-space batch fields."""
    current = batch.context_states_model[:, -1]
    future = batch.future_states_model
    if current.ndim != 2 or future.ndim != 3:
        raise ValueError("V0.4 vector batches require [B,C] current and [B,H,C] future")
    if future.shape[1] < 2:
        raise ValueError("V0.4 multi-step loss requires horizon > 1")
    observation_sequence = torch.cat((current.unsqueeze(1), future), dim=1)
    latent_sequence = model.encode(observation_sequence)
    z_current = latent_sequence[:, 0]
    z_future = latent_sequence[:, 1:]
    prediction = model.core.rollout(z_current, batch.future_dts)[:, 1:]
    one_step = (prediction[:, 0] - z_future[:, 0]).square().mean()
    multi_step = (prediction - z_future).square().mean()
    reconstruction = reconstruction_loss(
        model.decode(latent_sequence), observation_sequence
    )
    variance = variance_loss(latent_sequence, min_std=config.min_std)
    spectral = stability_regularizer(
        model.core.generator_matrix(), margin=config.stability_margin
    )
    total = (
        config.lambda_k * one_step
        + config.lambda_multi * multi_step
        + config.lambda_rec * reconstruction
        + config.lambda_var * variance
        + config.lambda_spec * spectral
    )
    return RepresentationLossBreakdown(
        total=total,
        koopman_one_step=one_step,
        koopman_multi_step=multi_step,
        reconstruction=reconstruction,
        variance=variance,
        spectral=spectral,
    )
