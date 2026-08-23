"""Problem-independent Phase-2 condition and history identifiability tools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from jka_model.adaptive.cache import AdaptiveCache


def condition_targets(conditions: Tensor, dts: Tensor) -> Tensor:
    """Augment ``[Re, U]`` with the causal backward condition rate ``dRe/dt``."""
    if conditions.ndim < 2 or conditions.shape[-1] != 2:
        raise ValueError("conditions must end in [Re,U]")
    if dts.shape != conditions.shape[:-1]:
        raise ValueError("condition/dt alignment mismatch")
    rate = torch.zeros_like(conditions[..., 0])
    if conditions.shape[-2] > 1:
        rate[..., 1:] = (
            conditions[..., 1:, 0] - conditions[..., :-1, 0]
        ) / dts[..., :-1].clamp_min(1.0e-12)
    return torch.cat((conditions, rate.unsqueeze(-1)), dim=-1)


def phase2_condition_scales(cache: AdaptiveCache) -> tuple[Tensor, Tensor]:
    """Fit observer target normalization from training trajectories only."""
    train = cache.select("train")
    if not train:
        raise ValueError("V0.9 Phase-2 condition scales require training trajectories")
    values = torch.cat(
        [condition_targets(item.conditions, item.dts) for item in train], dim=0
    )
    mean = values.mean(dim=0)
    floor = values.abs().mean(dim=0).clamp_min(1.0) * 1.0e-6
    std = values.std(dim=0).clamp_min(floor)
    return mean, std


def conditional_centering_loss(
    innovations: Tensor,
    conditions: Tensor,
    *,
    bandwidth: float,
) -> Tensor:
    """Kernel estimate of ``E[xi|q]``; no innovation-variance floor is imposed."""
    if innovations.ndim != 2 or conditions.ndim != 2:
        raise ValueError("conditional centering expects [N,r] innovations and [N,q]")
    if innovations.shape[0] != conditions.shape[0] or bandwidth <= 0:
        raise ValueError("invalid conditional-centering alignment or bandwidth")
    if innovations.shape[0] < 2:
        return innovations.square().sum() * 0.0
    distances = torch.cdist(conditions.float(), conditions.float()).square()
    weights = torch.exp(-0.5 * distances / bandwidth**2)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    conditional_mean = weights.to(innovations.dtype) @ innovations
    return conditional_mean.square().mean()


def condition_observer_metrics(predicted: Tensor, target: Tensor) -> dict[str, float]:
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError("observer metrics require aligned [N,3] normalized conditions")
    error = predicted - target
    rmse = error.square().mean(dim=0).sqrt()
    centered = target - target.mean(dim=0, keepdim=True)
    r2 = 1.0 - error.square().sum(dim=0) / centered.square().sum(dim=0).clamp_min(
        1.0e-12
    )
    names = ("reynolds_number", "u_infinity", "condition_rate")
    result: dict[str, float] = {
        "normalized_rmse": float(rmse.mean()),
        "minimum_r2": float(r2.min()),
    }
    for index, name in enumerate(names):
        result[f"{name}_normalized_rmse"] = float(rmse[index])
        result[f"{name}_r2"] = float(r2[index])
    return result


@dataclass(frozen=True, slots=True)
class MatchedHistoryPair:
    first: int
    second: int
    condition_distance: float
    latent_distance: float
    history_distance: float
    future_separation: float


def matched_history_pairs(
    conditions: Tensor,
    current_latents: Tensor,
    older_histories: Tensor,
    futures: Tensor,
    *,
    condition_tolerance: float,
    latent_tolerance: float,
    minimum_history_separation: float,
    minimum_future_separation: float,
    group_ids: Sequence[str] | None = None,
) -> tuple[MatchedHistoryPair, ...]:
    """Greedily select disjoint pairs with matched present and different histories."""
    count = conditions.shape[0]
    if (
        conditions.ndim != 2
        or current_latents.ndim != 2
        or older_histories.ndim != 3
        or futures.ndim != 3
        or not (
            current_latents.shape[0]
            == older_histories.shape[0]
            == futures.shape[0]
            == count
        )
    ):
        raise ValueError("invalid Phase-2 matched-pair tensor contract")
    if group_ids is not None and len(group_ids) != count:
        raise ValueError("Phase-2 matched-pair group ids must align with samples")
    candidates: list[MatchedHistoryPair] = []
    for first in range(count):
        for second in range(first + 1, count):
            if group_ids is not None and group_ids[first] == group_ids[second]:
                continue
            condition_distance = float(
                (conditions[first] - conditions[second]).square().mean().sqrt()
            )
            latent_distance = float(
                (current_latents[first] - current_latents[second])
                .square()
                .mean()
                .sqrt()
            )
            if (
                condition_distance > condition_tolerance
                or latent_distance > latent_tolerance
            ):
                continue
            history_distance = float(
                (older_histories[first] - older_histories[second])
                .square()
                .mean()
                .sqrt()
            )
            future_separation = float(
                (futures[first] - futures[second]).square().mean().sqrt()
            )
            if (
                history_distance < minimum_history_separation
                or future_separation < minimum_future_separation
            ):
                continue
            candidates.append(
                MatchedHistoryPair(
                    first,
                    second,
                    condition_distance,
                    latent_distance,
                    history_distance,
                    future_separation,
                )
            )
    candidates.sort(
        key=lambda item: (
            item.condition_distance + item.latent_distance,
            -item.history_distance,
            -item.future_separation,
            item.first,
            item.second,
        )
    )
    selected: list[MatchedHistoryPair] = []
    occupied: set[int] = set()
    for item in candidates:
        if item.first in occupied or item.second in occupied:
            continue
        selected.append(item)
        occupied.update((item.first, item.second))
    return tuple(selected)
