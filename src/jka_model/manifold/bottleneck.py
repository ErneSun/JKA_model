"""Read-only, same-sample diagnostics. These are not scientific admission gates."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor

from jka_model.manifold.physical import central_difference_2d


def error_decomposition(prediction: Tensor, reference: Tensor, truth: Tensor) -> dict[str, Any]:
    """e = (prediction-reference) + (reference-truth), including signed interference.

    All three arguments must already have identical units, gauge, and spatial mask.
    Float64 accumulation makes closure meaningful even when the terms cancel.
    A teacher cross-decoding error is NOT an autoencoder reconstruction lower bound.
    """
    if prediction.shape != truth.shape or reference.shape != truth.shape or truth.numel() == 0:
        raise ValueError("decomposition requires nonempty aligned tensors")
    p, r, y = (v.detach().double() for v in (prediction, reference, truth))
    if not all(bool(torch.isfinite(v).all()) for v in (p, r, y)):
        raise ValueError("decomposition inputs must be finite")
    propagation, reconstruction, error = p - r, r - y, p - y
    total = float(error.square().mean())
    prop = float(propagation.square().mean())
    recon = float(reconstruction.square().mean())
    cross = float(2 * (propagation * reconstruction).mean())
    target = float(y.square().mean())
    scale = max(total, prop + recon + abs(cross), 1e-30)
    closure = abs(total - prop - recon - cross) / scale
    if closure > 1e-10:
        raise ArithmeticError("same-sample error decomposition failed closure")
    return {
        "total_mse": total,
        "propagation_mse": prop,
        "reference_decode_mse": recon,
        "cross_term": cross,
        "target_mse": target,
        "closure_relative": closure,
        "relative_l2": math.sqrt(total / target) if target > 1e-30 else None,
        "elements": y.numel(),
    }


def field_views(field: Tensor, mask: Tensor, *, dx: float, dy: float) -> dict[str, Tensor]:
    """Nondimensional raw cylinder channels; no hidden reweighting or gauge correction.

    Pressure demeaning is a separate diagnostic. Vorticity excludes stencils touching
    solids or the outer boundary; field_all preserves the inherited unmasked metric.
    """
    if field.ndim != 3 or field.shape[0] != 3 or mask.shape != field.shape[1:]:
        raise ValueError("field views require [3,Nx,Ny] and [Nx,Ny] fluid mask")
    if mask.dtype != torch.bool or not bool(mask.any()) or min(dx, dy) <= 0:
        raise ValueError("invalid fluid mask or spacing")
    f = field.double()
    interior = torch.zeros_like(mask)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    )
    if not bool(interior.any()):
        raise ValueError("no valid interior vorticity stencil")
    pressure = f[2, mask]
    vorticity = central_difference_2d(f[1], dx, -2) - central_difference_2d(f[0], dy, -1)
    return {
        "field_all": f.flatten(),
        "field_fluid": f[:, mask].flatten(),
        "velocity_fluid": f[:2, mask].flatten(),
        "u_fluid": f[0, mask],
        "v_fluid": f[1, mask],
        "pressure_fluid": pressure,
        "pressure_demeaned_fluid": pressure - pressure.mean(),
        "vorticity_interior": vorticity[interior],
    }


def latent_spectrum(values: Tensor) -> dict[str, Any]:
    if values.ndim != 2 or len(values) < 2 or not bool(torch.isfinite(values).all()):
        raise ValueError("spectrum requires finite [N,d], N >= 2")
    s = torch.linalg.svdvals(values.double() - values.double().mean(0))
    energy = s.square()
    total = float(energy.sum())
    if total == 0:
        return {
            "singular_values": s.tolist(),
            "effective_rank": 0.0,
            "rank_99": 0,
            "numerical_rank": 0,
            "resolved_condition_number": None,
        }
    probabilities = energy / total
    resolved = s > max(values.shape) * torch.finfo(torch.float64).eps * s[0]
    return {
        "singular_values": s.tolist(),
        "effective_rank": float(
            (-(probabilities * probabilities.clamp_min(1e-300).log()).sum()).exp()
        ),
        "rank_99": int(torch.searchsorted(probabilities.cumsum(0), 0.99)) + 1,
        "numerical_rank": int(resolved.sum()),
        "resolved_condition_number": float(s[0] / s[resolved][-1]),
    }


def fit_gauge(candidate_train: Tensor, reference_train: Tensor) -> dict[str, Tensor]:
    """Fit a row-vector orthogonal gauge on TRAIN only, with train-only centering/scale."""
    if candidate_train.shape != reference_train.shape:
        raise ValueError("gauge requires aligned training states")
    latent_spectrum(candidate_train)
    latent_spectrum(reference_train)
    x, y = candidate_train.double(), reference_train.double()
    mx, my = x.mean(0), y.mean(0)
    sx, sy = (x - mx).square().mean().sqrt(), (y - my).square().mean().sqrt()
    if min(float(sx), float(sy)) <= 1e-15:
        raise ValueError("gauge is undefined for a constant representation")
    xn, yn = (x - mx) / sx, (y - my) / sy
    u, _, vh = torch.linalg.svd(xn.T @ yn)
    _, s, vx = torch.linalg.svd(xn, full_matrices=False)
    rank = int(torch.searchsorted((s.square() / s.square().sum()).cumsum(0), 0.99)) + 1
    return {
        "transform": u @ vh,
        "candidate_mean": mx,
        "reference_mean": my,
        "candidate_scale": sx,
        "reference_scale": sy,
        "train_subspace": vx[:rank].T,
    }


def evaluate_gauge(
    fit: dict[str, Tensor], candidate: Tensor, reference: Tensor, generator: Tensor
) -> dict[str, Any]:
    """Report all-axis AND data-excited commutators; neither replaces old gates.

    Row states evolve with A.T, so the row alignment Q commutator is A.T Q-Q A.T.
    Uncentered candidate states are used for dynamical action: centering would erase
    mean-state contributions. Mean/scale shifts are reported separately.
    """
    if candidate.shape != reference.shape:
        raise ValueError("held-out gauge states must align")
    x, y, a = candidate.double(), reference.double(), generator.double()
    q = fit["transform"]
    xn = (x - fit["candidate_mean"]) / fit["candidate_scale"]
    yn = (y - fit["reference_mean"]) / fit["reference_scale"]
    c = a.T @ q - q @ a.T

    def ratio(numerator: Tensor, denominator: Tensor) -> float | None:
        d = float(denominator.norm())
        return float(numerator.norm()) / d if d > 1e-15 else None

    projected = x @ fit["train_subspace"] @ fit["train_subspace"].T
    alpha = fit["reference_scale"] / fit["candidate_scale"]
    offset = fit["reference_mean"] - alpha * fit["candidate_mean"] @ q
    mapped = alpha * x @ q + offset
    return {
        "alignment_fit_split": "train",
        "spectrum": latent_spectrum(x),
        "aligned_nrmse": ratio(xn @ q - yn, yn),
        "full_commutator": ratio(c, a),
        "data_action_commutator": ratio(x @ c, x @ a.T),
        "affine_dynamical_action_defect": ratio(alpha * x @ c - offset @ a.T, mapped @ a.T),
        "affine_offset_norm": float(offset.norm()),
        "train_99pct_subspace_action_commutator": ratio(projected @ c, projected @ a.T),
        "train_subspace_rank": fit["train_subspace"].shape[1],
        "train_scale_ratio": float(fit["reference_scale"] / fit["candidate_scale"]),
        "heldout_mean_shift": float((xn.mean(0) @ q - yn.mean(0)).norm()),
        "note": "Low action error on sampled states does not prove global dynamical equivalence.",
    }


def transition_contract(conditions: Tensor, start: int, horizon: int) -> dict[str, Any]:
    """u[t] drives x[t] -> x[t+1]; x[t] predates this transition's input."""
    if conditions.ndim != 2 or start < 1 or horizon < 1 or start + horizon > len(conditions):
        raise ValueError("invalid causal input window")
    c = conditions.double()
    if not bool(torch.isfinite(c).all()):
        raise ValueError("nonfinite input schedule")
    first = c[start] - c[start - 1]
    future = c[start + 1 : start + horizon] - c[start : start + horizon - 1]
    # Scale-aware numeric change detector, not an estimated predictability threshold.
    tol = 1e-6 * c.abs().amax(0).clamp_min(1.0)
    return {
        "first_transition_delta": first.tolist(),
        "changed_at_origin": bool((first.abs() > tol).any()),
        "changes_after_origin": int((future.abs() > tol).any(dim=1).sum()),
        "input_indices": [start, start + horizon - 1],
        "target_state_indices": [start + 1, start + horizon],
    }


def summarize_errors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average windows within trajectory, then trajectories equally (no pooled seeds)."""
    groups: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    keys = ("split", "predictor", "reference", "horizon", "channel", "input_group")
    for row in rows:
        groups[tuple(row[k] for k in keys)][row["trajectory_id"]].append(row)
    metrics = (
        "total_mse",
        "propagation_mse",
        "reference_decode_mse",
        "cross_term",
        "target_mse",
        "relative_l2",
    )
    result = []
    for key, trajectories in sorted(groups.items()):
        item = dict(zip(keys, key, strict=True))
        for metric in metrics:
            values = [[r[metric] for r in values] for values in trajectories.values()]
            item[metric] = (
                sum(sum(v) / len(v) for v in values) / len(values)
                if all(x is not None for v in values for x in v)
                else None
            )
        item["trajectories"] = len(trajectories)
        item["windows"] = sum(len(v) for v in trajectories.values())
        item["max_closure_relative"] = max(
            r["closure_relative"] for values in trajectories.values() for r in values
        )
        result.append(item)
    return result


def diagnostic_priority(row: dict[str, Any]) -> str:
    """Predeclared heuristic, not an acceptance gate or causal identification claim."""
    r, p, c = (row[k] for k in ("reference_decode_mse", "propagation_mse", "cross_term"))
    if r + p <= 1e-20:
        return "UNRESOLVED_NEAR_ZERO"
    if abs(c) > 0.5 * (r + p):
        return "COUPLED_CROSS_TERM"
    if r > 2 * p:
        return "REPRESENTATION_DECODE_PRIORITY"
    if p > 2 * r:
        return "PROPAGATION_PRIORITY"
    return "MIXED"
