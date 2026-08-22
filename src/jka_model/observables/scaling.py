"""Train-only robust scaling for differentiable scientific observables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


def deterministic_subsample(values: Tensor, maximum: int) -> Tensor:
    """Bound scale-estimation memory without introducing a stochastic dependency."""
    flat = values.detach().float().cpu().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() <= maximum:
        return flat
    indices = torch.linspace(0, flat.numel() - 1, maximum).round().long()
    return flat.index_select(0, indices)


@dataclass(frozen=True, slots=True)
class RobustObservableScaleState:
    """Serializable scales fitted exclusively from the training trajectories."""

    method: str
    scales: dict[str, float]
    centers: dict[str, float]
    sample_counts: dict[str, int]
    split_fingerprint: str
    epsilon: float

    def __post_init__(self) -> None:
        if self.method not in {"mad", "rms"}:
            raise ValueError("observable scale method must be 'mad' or 'rms'")
        if not self.scales or set(self.scales) != set(self.centers):
            raise ValueError("observable scale state is incomplete")
        if set(self.scales) != set(self.sample_counts):
            raise ValueError("observable scale sample counts are incomplete")
        if self.epsilon <= 0 or not self.split_fingerprint:
            raise ValueError("observable scale provenance is invalid")
        if any(
            not torch.isfinite(torch.tensor(value)) or value <= 0
            for value in self.scales.values()
        ):
            raise ValueError("observable scales must be finite and positive")
        if any(count < 1 for count in self.sample_counts.values()):
            raise ValueError("observable scale sample counts must be positive")

    def scale(self, name: str, reference: Tensor) -> Tensor:
        if name not in self.scales:
            raise KeyError(f"observable scale is missing {name!r}")
        return reference.new_tensor(self.scales[name])

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "scales": dict(self.scales),
            "centers": dict(self.centers),
            "sample_counts": dict(self.sample_counts),
            "split_fingerprint": self.split_fingerprint,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RobustObservableScaleState:
        return cls(
            method=str(payload["method"]),
            scales={str(key): float(value) for key, value in dict(payload["scales"]).items()},
            centers={str(key): float(value) for key, value in dict(payload["centers"]).items()},
            sample_counts={
                str(key): int(value) for key, value in dict(payload["sample_counts"]).items()
            },
            split_fingerprint=str(payload["split_fingerprint"]),
            epsilon=float(payload["epsilon"]),
        )


def fit_robust_observable_scales(
    samples: Mapping[str, Tensor],
    *,
    method: str,
    epsilon: float,
    split_fingerprint: str,
    maximum_samples: int = 262_144,
    relative_floors: Mapping[str, tuple[str, float]] | None = None,
) -> RobustObservableScaleState:
    """Fit MAD or centered RMS scales with optional dimensionally matched floors."""
    if method not in {"mad", "rms"} or epsilon <= 0 or maximum_samples < 1:
        raise ValueError("invalid robust observable scale configuration")
    reduced = {
        name: deterministic_subsample(value, maximum_samples)
        for name, value in samples.items()
    }
    if not reduced or any(value.numel() == 0 for value in reduced.values()):
        raise ValueError("observable scale fitting received empty or non-finite samples")
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name, values in reduced.items():
        center = values.median() if method == "mad" else values.mean()
        deviation = values - center
        scale = (
            deviation.abs().median() * 1.4826
            if method == "mad"
            else deviation.square().mean().sqrt()
        )
        centers[name] = float(center)
        scales[name] = max(float(scale), epsilon)
    for name, (reference, fraction) in (relative_floors or {}).items():
        if name not in scales or reference not in scales or fraction < 0:
            raise ValueError("invalid observable relative scale floor")
        scales[name] = max(scales[name], scales[reference] * fraction, epsilon)
    return RobustObservableScaleState(
        method=method,
        scales=scales,
        centers=centers,
        sample_counts={name: int(value.numel()) for name, value in reduced.items()},
        split_fingerprint=split_fingerprint,
        epsilon=epsilon,
    )


def standardized_huber(
    error: Tensor,
    scale: Tensor | float,
    *,
    delta: float,
    mask: Tensor | None = None,
) -> Tensor:
    """Mean Huber loss of a dimensionless error, optionally on a physical mask."""
    if delta <= 0:
        raise ValueError("Huber delta must be positive")
    denominator = torch.as_tensor(scale, device=error.device, dtype=error.dtype)
    if denominator.numel() != 1 or not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise ValueError("observable Huber scale must be finite and positive")
    normalized = error / denominator
    if mask is not None:
        expanded = mask.to(device=error.device, dtype=torch.bool)
        while expanded.ndim < normalized.ndim:
            expanded = expanded.unsqueeze(1)
        expanded = expanded.expand_as(normalized)
        normalized = normalized[expanded]
    if normalized.numel() == 0:
        raise ValueError("observable Huber mask selected no values")
    return F.huber_loss(normalized, torch.zeros_like(normalized), delta=delta, reduction="mean")
