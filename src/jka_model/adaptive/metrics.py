"""Auditable residual decomposition and adaptive-operator metrics."""

from __future__ import annotations

from torch import Tensor


def residual_decomposition(
    truth_next: Tensor,
    nominal_next: Tensor,
    adapted_next: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if truth_next.shape != nominal_next.shape or truth_next.shape != adapted_next.shape:
        raise ValueError("residual-decomposition tensors must share shape")
    nominal = truth_next - nominal_next
    explained = adapted_next - nominal_next
    remaining = truth_next - adapted_next
    return nominal, explained, remaining


def operator_explained_fraction(
    nominal_residual: Tensor,
    remaining_residual: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    if nominal_residual.shape != remaining_residual.shape:
        raise ValueError("Gamma_op residuals must share shape")
    numerator = remaining_residual.square().sum(dim=-1).mean()
    denominator = nominal_residual.square().sum(dim=-1).mean() + eps
    return 1.0 - numerator / denominator


def latent_prediction_metrics(
    truth: Tensor,
    prediction: Tensor,
    nominal_prediction: Tensor,
) -> dict[str, float]:
    nominal, _, remaining = residual_decomposition(truth, nominal_prediction, prediction)
    rmse = (prediction - truth).square().mean().sqrt()
    nominal_rmse = (nominal_prediction - truth).square().mean().sqrt()
    return {
        "latent_rmse": float(rmse),
        "nominal_latent_rmse": float(nominal_rmse),
        "relative_gain": float(1.0 - rmse / nominal_rmse.clamp_min(1e-12)),
        "gamma_operator": float(operator_explained_fraction(nominal, remaining)),
    }
