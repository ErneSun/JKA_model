"""Residual statistics, closure metrics, and evidence-bounded memory labels."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from jka_model.residual.cache import ResidualCache


def residual_structure_metrics(cache: ResidualCache, split: str) -> dict[str, Any]:
    """Return closure-independent residual scale and Koopman adequacy metrics."""
    trajectories = cache.select(split)
    if not trajectories:
        raise ValueError(f"no residual trajectories in split {split}")
    residual = torch.cat([item.residuals.double() for item in trajectories])
    true_increment = torch.cat(
        [(item.latents[1:] - item.latents[:-1]).double() for item in trajectories]
    )
    residual_energy = residual.square().sum(dim=-1).mean()
    increment_energy = true_increment.square().sum(dim=-1).mean()
    return {
        "split": split,
        "sample_count": residual.shape[0],
        "residual_rms": float(residual.square().mean().sqrt()),
        "true_increment_rms": float(true_increment.square().mean().sqrt()),
        "residual_significance": float(residual_energy / increment_energy.clamp_min(1e-12)),
        "per_dimension_residual_rms": residual.square().mean(dim=0).sqrt().tolist(),
    }


def _correlation(left: Tensor, right: Tensor, eps: float = 1e-12) -> float:
    left = left.double().flatten()
    right = right.double().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if denominator <= eps:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def residual_statistics(cache: ResidualCache, split: str, max_acf_lag: int) -> dict[str, Any]:
    trajectories = cache.select(split)
    if not trajectories:
        raise ValueError(f"no residual trajectories in split {split}")
    residual = torch.cat([item.residuals.double() for item in trajectories])
    latent = torch.cat([item.latents[:-1].double() for item in trajectories])
    next_latent = torch.cat([item.latents[1:].double() for item in trajectories])
    dts = torch.cat([item.dts.double() for item in trajectories])
    parameters = torch.cat(
        [item.parameters.double().expand(item.residuals.shape[0], -1) for item in trajectories]
    )
    residual_norm = residual.norm(dim=-1)
    latent_norm = latent.norm(dim=-1)
    base_next = next_latent - residual
    base_increment = base_next - latent
    true_increment = next_latent - latent
    residual_energy = residual.square().sum(dim=-1).mean()
    true_increment_energy = true_increment.square().sum(dim=-1).mean()
    base_increment_norm = base_increment.norm(dim=-1)
    closure_burden = residual_norm / (residual_norm + base_increment_norm + 1e-12)
    centered = residual - residual.mean(dim=0)
    acf: dict[str, list[float]] = {}
    for dimension in range(residual.shape[1]):
        values: list[float] = []
        for lag in range(1, max_acf_lag + 1):
            pairs: list[tuple[Tensor, Tensor]] = []
            for item in trajectories:
                series = item.residuals[:, dimension].double()
                if series.numel() > lag:
                    pairs.append((series[:-lag], series[lag:]))
            if not pairs:
                values.append(float("nan"))
            else:
                values.append(
                    _correlation(
                        torch.cat([pair[0] for pair in pairs]),
                        torch.cat([pair[1] for pair in pairs]),
                    )
                )
        acf[str(dimension)] = values
    quantiles = torch.quantile(
        residual_norm,
        torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=residual_norm.dtype),
    ).tolist()
    per_dim_std = residual.std(dim=0, unbiased=False)
    residual_latent_correlation = [
        [
            _correlation(residual[:, residual_index], latent[:, latent_index])
            for latent_index in range(latent.shape[1])
        ]
        for residual_index in range(residual.shape[1])
    ]
    covariance = centered.T @ centered / max(residual.shape[0], 1)
    std_outer = per_dim_std[:, None] * per_dim_std[None, :]
    correlation = torch.where(
        std_outer > 1e-15, covariance / std_outer, torch.zeros_like(covariance)
    )
    return {
        "split": split,
        "sample_count": residual.shape[0],
        "mean": residual.mean(dim=0).tolist(),
        "std": per_dim_std.tolist(),
        "per_dimension_variance": residual.var(dim=0, unbiased=False).tolist(),
        "rms": float(residual.square().mean().sqrt()),
        "residual_significance": float(residual_energy / true_increment_energy.clamp_min(1e-12)),
        "true_increment_rms": float(true_increment.square().mean().sqrt()),
        "normalized_rms_by_true_increment": float(
            residual.square().mean().sqrt() / true_increment.square().mean().sqrt().clamp_min(1e-12)
        ),
        "normalized_rms_by_base_increment": float(
            residual.square().mean().sqrt() / base_increment.square().mean().sqrt().clamp_min(1e-12)
        ),
        "per_dimension_rms": residual.square().mean(dim=0).sqrt().tolist(),
        "max_abs": float(residual.abs().max()),
        "norm_quantiles": dict(zip(("q50", "q90", "q95", "q99"), quantiles, strict=True)),
        "acf": acf,
        "dimension_correlation": correlation.tolist(),
        "residual_latent_cross_correlation": residual_latent_correlation,
        "residual_norm_latent_norm_correlation": _correlation(residual_norm, latent_norm),
        "residual_norm_dt_correlation": _correlation(residual_norm, dts),
        "residual_dimension_dt_correlations": [
            _correlation(residual[:, index], dts) for index in range(residual.shape[1])
        ],
        "residual_norm_parameter_correlations": [
            _correlation(residual_norm, parameters[:, index])
            for index in range(parameters.shape[1])
        ],
        "closure_burden_rms_ratio": float(
            residual.square().mean().sqrt() / latent.square().mean().sqrt().clamp_min(1e-12)
        ),
        "residual_to_base_increment_ratio": {
            "mean": float((residual_norm / base_increment_norm.clamp_min(1e-12)).mean()),
            "q50": float(
                torch.quantile(residual_norm / base_increment_norm.clamp_min(1e-12), 0.50)
            ),
            "q95": float(
                torch.quantile(residual_norm / base_increment_norm.clamp_min(1e-12), 0.95)
            ),
        },
        "closure_burden": {
            "definition": "||r||/(||r||+||delta_z_base||+eps)",
            "mean": float(closure_burden.mean()),
            "q50": float(torch.quantile(closure_burden, 0.50)),
            "q95": float(torch.quantile(closure_burden, 0.95)),
        },
    }


def closure_metrics(
    prediction: Tensor, target: Tensor, residual_scale: Tensor | None = None
) -> dict[str, Any]:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("closure predictions and targets must have shape [N,d]")
    prediction = prediction.double()
    target = target.double()
    error = prediction - target
    target_mean = target.mean(dim=0)
    ss_res = error.square().sum(dim=0)
    ss_total = (target - target_mean).square().sum(dim=0)
    per_dimension_r2 = torch.where(
        ss_total > 1e-15, 1.0 - ss_res / ss_total, torch.zeros_like(ss_total)
    )
    global_total = (target - target.mean()).square().sum()
    r2 = 0.0 if global_total <= 1e-15 else float(1.0 - error.square().sum() / global_total)
    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1, eps=1e-12)
    target_rms = target.square().mean().sqrt()
    per_dimension_mse = error.square().mean(dim=0)
    per_dimension_target_rms = target.square().mean(dim=0).sqrt()
    result = {
        "mse": float(error.square().mean()),
        "target_rms": float(target_rms),
        "normalized_rmse": float(error.square().mean().sqrt() / target_rms.clamp_min(1e-12)),
        "r2": r2,
        "per_dimension_mse": per_dimension_mse.tolist(),
        "per_dimension_target_rms": per_dimension_target_rms.tolist(),
        "per_dimension_normalized_rmse": (
            per_dimension_mse.sqrt() / per_dimension_target_rms.clamp_min(1e-12)
        ).tolist(),
        "per_dimension_r2": per_dimension_r2.tolist(),
        "mean_cosine_similarity": float(cosine.mean()),
    }
    if residual_scale is not None:
        scale = residual_scale.detach().double().reshape(1, -1)
        if scale.shape[1] != target.shape[1] or torch.any(scale <= 0):
            raise ValueError("residual_scale must be positive with shape [latent_dim]")
        standardized = error / scale
        result["standardized_mse"] = float(standardized.square().mean())
        result["per_dimension_standardized_mse"] = standardized.square().mean(dim=0).tolist()
    return result


def classify_memory_evidence(
    *,
    residual_rms: float,
    min_residual_rms: float,
    history_r2: float,
    instantaneous_r2: float,
    shuffled_history_r2: float | None,
    min_history_gain: float,
    history_closed_loop_improves: bool,
) -> dict[str, Any]:
    history_gain = history_r2 - instantaneous_r2
    shuffled_gain = None if shuffled_history_r2 is None else history_r2 - shuffled_history_r2
    if residual_rms < min_residual_rms:
        label = "NONE"
        reason = "residual signal is below the configured identification floor"
    elif (
        history_gain >= min_history_gain
        and history_closed_loop_improves
        and (shuffled_gain is None or shuffled_gain >= min_history_gain / 2)
    ):
        label = "STRONG"
        reason = "ordered history improves held-out residual fit and closed-loop rollout"
    elif history_gain > 0 or history_closed_loop_improves:
        label = "WEAK"
        reason = "history has partial but not jointly decisive evidence"
    else:
        label = "NONE"
        reason = "ordered history does not improve held-out prediction and rollout"
    return {
        "label": label,
        "reason": reason,
        "history_r2_gain": history_gain,
        "ordered_vs_shuffled_r2_gain": shuffled_gain,
        "history_closed_loop_improves": history_closed_loop_improves,
        "exact_mori_zwanzig_kernel_claimed": False,
    }
