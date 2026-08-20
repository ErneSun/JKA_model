"""V0.8 prediction and context-collapse diagnostics."""

from __future__ import annotations

import torch
from torch import Tensor


def _r2(prediction: Tensor, target: Tensor) -> float:
    centered = target - target.mean(dim=0, keepdim=True)
    denominator = centered.square().sum().clamp_min(1e-12)
    return float(1.0 - (prediction - target).square().sum() / denominator)


def context_prediction_metrics(
    residual_prediction: Tensor,
    residual_target: Tensor,
    adequacy_prediction: Tensor,
    adequacy_target: Tensor,
    residual_scale: Tensor,
    adequacy_scale: Tensor,
) -> dict[str, float]:
    error = residual_prediction - residual_target
    standardized = error / residual_scale
    target_rms = residual_target.square().mean().sqrt().clamp_min(1e-12)
    adequacy_error = adequacy_prediction - adequacy_target
    left = adequacy_prediction.flatten() - adequacy_prediction.mean()
    right = adequacy_target.flatten() - adequacy_target.mean()
    correlation = float(
        (left * right).sum()
        / (left.square().sum().sqrt() * right.square().sum().sqrt()).clamp_min(1e-12)
    )
    return {
        "residual_mse": float(error.square().mean()),
        "residual_standardized_mse": float(standardized.square().mean()),
        "residual_nrmse": float(error.square().mean().sqrt() / target_rms),
        "residual_r2": _r2(residual_prediction, residual_target),
        "adequacy_mse": float(adequacy_error.square().mean()),
        "adequacy_standardized_mse": float((adequacy_error / adequacy_scale).square().mean()),
        "adequacy_mae": float(adequacy_error.abs().mean()),
        "adequacy_r2": _r2(adequacy_prediction, adequacy_target),
        "adequacy_correlation": correlation,
    }


def context_diagnostics(context: Tensor) -> dict[str, object]:
    if context.ndim != 2 or context.shape[0] < 2:
        raise ValueError("context diagnostics require [N,d_c] with N>=2")
    centered = context - context.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered.float())
    energy = singular.square()
    effective_rank = float(energy.sum().square() / energy.square().sum().clamp_min(1e-12))
    return {
        "per_dimension_variance": context.var(dim=0, unbiased=False).tolist(),
        "effective_rank": effective_rank,
        "mean_norm": float(context.norm(dim=-1).mean()),
        "std_norm": float(context.norm(dim=-1).std(unbiased=False)),
        "time_variation_rms": float((context[1:] - context[:-1]).square().mean().sqrt()),
        "collapsed": bool(effective_rank <= 1.05 or float(centered.square().mean()) <= 1e-10),
    }
