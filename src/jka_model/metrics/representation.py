"""Coordinate-invariant V0.4 latent alignment and non-collapse diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class AffineLatentAlignment:
    """Train-fitted affine map from learned coordinates to hidden evaluation state."""

    coefficients: Tensor

    def __post_init__(self) -> None:
        if self.coefficients.ndim != 2 or self.coefficients.requires_grad:
            raise ValueError("alignment coefficients must be a detached matrix")

    def apply(self, z_k: Tensor) -> Tensor:
        if z_k.ndim != 2 or z_k.shape[1] + 1 != self.coefficients.shape[0]:
            raise ValueError("alignment input shape is incompatible with coefficients")
        ones = torch.ones((z_k.shape[0], 1), dtype=z_k.dtype, device=z_k.device)
        design = torch.cat((z_k, ones), dim=1)
        return design @ self.coefficients.to(dtype=z_k.dtype, device=z_k.device)


@dataclass(frozen=True, slots=True)
class LatentAlignmentMetrics:
    r2: float
    mse: float


@dataclass(frozen=True, slots=True)
class LatentDiagnostics:
    mean: Tensor
    std: Tensor
    minimum_std: float
    maximum_std: float
    covariance_condition: float


def fit_affine_latent_alignment(z_train: Tensor, hidden_train: Tensor) -> AffineLatentAlignment:
    """Fit ``hidden ~= [z,1] @ coefficients`` using train pairs only."""
    if z_train.ndim != 2 or hidden_train.ndim != 2:
        raise ValueError("alignment inputs must have shape [N,d]")
    if z_train.shape[0] != hidden_train.shape[0] or z_train.shape[0] < z_train.shape[1] + 1:
        raise ValueError("alignment requires enough paired samples")
    if z_train.dtype != hidden_train.dtype or z_train.device != hidden_train.device:
        raise ValueError("alignment inputs must share dtype/device")
    ones = torch.ones((z_train.shape[0], 1), dtype=z_train.dtype, device=z_train.device)
    design = torch.cat((z_train.detach(), ones), dim=1)
    coefficients = torch.linalg.lstsq(design, hidden_train.detach()).solution
    return AffineLatentAlignment(coefficients.detach())


def evaluate_affine_latent_alignment(
    alignment: AffineLatentAlignment,
    z_test: Tensor,
    hidden_test: Tensor,
) -> LatentAlignmentMetrics:
    if hidden_test.ndim != 2 or hidden_test.shape[0] != z_test.shape[0]:
        raise ValueError("test hidden/latent pairs must share the sample dimension")
    prediction = alignment.apply(z_test.detach())
    target = hidden_test.detach()
    residual = (prediction - target).square().sum()
    centered = (target - target.mean(dim=0, keepdim=True)).square().sum()
    if centered <= 0:
        raise ValueError("alignment R2 requires non-constant hidden targets")
    return LatentAlignmentMetrics(
        r2=float((1.0 - residual / centered).item()),
        mse=float((prediction - target).square().mean().item()),
    )


def latent_diagnostics(z_k: Tensor) -> LatentDiagnostics:
    """Return detached population statistics across all non-latent axes."""
    if z_k.ndim < 2 or z_k.shape[-1] < 1:
        raise ValueError("z_k must contain sample and latent axes")
    values = z_k.detach().reshape(-1, z_k.shape[-1])
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    centered = values - mean
    covariance = centered.transpose(0, 1) @ centered / max(values.shape[0], 1)
    condition = torch.linalg.cond(covariance)
    return LatentDiagnostics(
        mean=mean,
        std=std,
        minimum_std=float(std.min().item()),
        maximum_std=float(std.max().item()),
        covariance_condition=float(condition.item()),
    )
