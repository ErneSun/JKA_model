"""Collapse, tracking, and continuous-time diagnostics for V0.6."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from jka_model.models import FieldJEPAKoopmanModel, normalized_parameter_distance


def latent_statistics(latent: Tensor, eps: float = 1e-12) -> dict[str, float]:
    if latent.ndim < 2:
        raise ValueError("latent samples must have shape [...,d]")
    flat = latent.detach().reshape(-1, latent.shape[-1]).float()
    std = flat.std(dim=0, unbiased=False)
    centered = flat - flat.mean(dim=0)
    covariance = centered.T @ centered / max(flat.shape[0], 1)
    singular = torch.linalg.svdvals(covariance)
    condition = singular.max() / singular.min().clamp_min(eps)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min_dimension_std": float(std.min()),
        "max_dimension_std": float(std.max()),
        "mean_norm": float(flat.norm(dim=-1).mean()),
        "covariance_condition": float(condition),
    }


def latent_tracking_distance(online: Tensor, target: Tensor, eps: float = 1e-12) -> float:
    if online.shape != target.shape:
        raise ValueError("online and target latents must have identical shapes")
    numerator = (online.detach() - target.detach()).norm()
    denominator = online.detach().norm().clamp_min(eps)
    return float(numerator / denominator)


def near_identity_diagnostic(generator: Tensor, dts: Tensor) -> dict[str, Any]:
    """Report ||exp(A dt)-I||_F/||I||_F at min/median/max observed dt."""
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError("generator must be square")
    values = dts.detach().flatten().float()
    if values.numel() == 0 or (values <= 0).any():
        raise ValueError("diagnostic dts must be non-empty and positive")
    chosen = {
        "small": values.min(),
        "median": values.median(),
        "large": values.max(),
    }
    identity = torch.eye(generator.shape[0], device=generator.device, dtype=generator.dtype)
    scale = identity.norm()
    return {
        name: {
            "dt": float(dt),
            "relative_frobenius": float(
                (torch.matrix_exp(generator * dt) - identity).norm() / scale
            ),
        }
        for name, dt in chosen.items()
    }


@torch.no_grad()
def model_tracking_diagnostics(model: FieldJEPAKoopmanModel, fields: Tensor) -> dict[str, Any]:
    online = model.encode(fields)
    target = model.encode_target(fields)
    return {
        "online": latent_statistics(online),
        "target": latent_statistics(target),
        "latent_distance": latent_tracking_distance(online, target),
        "parameter_distance": normalized_parameter_distance(model),
    }
