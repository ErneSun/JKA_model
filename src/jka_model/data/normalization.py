"""Train-split-only channel-wise state normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from jka_model.contracts import ProblemSpec
from jka_model.data.datasets import TrajectoryRecord
from jka_model.data.splits import SplitManifest


class ChannelStandardizer:
    """Fitted channel mean and scale without changing physical raw states."""

    def __init__(self, *, eps: float = 1e-6) -> None:
        if eps <= 0:
            raise ValueError("normalizer eps must be positive")
        self.eps = eps
        self.mean: Tensor | None = None
        self.scale: Tensor | None = None
        self.spatial_dim: int | None = None
        self.layout: str | None = None
        self.fitted_trajectory_ids: tuple[str, ...] = ()

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.scale is not None

    def fit(
        self,
        records: Sequence[TrajectoryRecord],
        manifest: SplitManifest,
        spec: ProblemSpec,
    ) -> ChannelStandardizer:
        """Fit exclusively on IDs declared by ``manifest.train``."""
        lookup = {record.trajectory_id: record for record in records}
        missing = set(manifest.train) - set(lookup)
        if missing:
            raise ValueError(f"training split references missing IDs: {', '.join(sorted(missing))}")
        if not manifest.train:
            raise ValueError("normalizer requires a non-empty training split")
        channel_axis = 1 if spec.grid.layout == "channels_first" else -1
        flattened: list[Tensor] = []
        for identifier in manifest.train:
            state = lookup[identifier].states_raw.detach().to(dtype=torch.float64, device="cpu")
            moved = state.movedim(channel_axis, 0)
            flattened.append(moved.reshape(moved.shape[0], -1))
        samples = torch.cat(flattened, dim=1)
        self.mean = samples.mean(dim=1)
        variance = ((samples - self.mean[:, None]) ** 2).mean(dim=1)
        self.scale = torch.sqrt(variance) + self.eps
        self.spatial_dim = spec.spatial_dim
        self.layout = spec.grid.layout
        self.fitted_trajectory_ids = tuple(manifest.train)
        return self

    def _statistics_for(self, value: Tensor) -> tuple[Tensor, Tensor]:
        if not self.is_fitted or self.spatial_dim is None or self.layout is None:
            raise RuntimeError("normalizer must be fitted before transform")
        assert self.mean is not None and self.scale is not None
        channel_axis = value.ndim - self.spatial_dim - 1
        if self.layout == "channels_last":
            channel_axis = value.ndim - 1
        if value.shape[channel_axis] != self.mean.numel():
            raise ValueError("input channel dimension does not match fitted normalizer")
        shape = [1] * value.ndim
        shape[channel_axis] = self.mean.numel()
        mean = self.mean.to(device=value.device, dtype=value.dtype).reshape(shape)
        scale = self.scale.to(device=value.device, dtype=value.dtype).reshape(shape)
        return mean, scale

    def transform(self, states_raw: Tensor) -> Tensor:
        mean, scale = self._statistics_for(states_raw)
        return (states_raw - mean) / scale

    def inverse_transform(self, states_model: Tensor) -> Tensor:
        mean, scale = self._statistics_for(states_model)
        return states_model * scale + mean

    def state_dict(self) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("cannot serialize an unfitted normalizer")
        assert self.mean is not None and self.scale is not None
        return {
            "kind": "channel_standardizer",
            "eps": self.eps,
            "mean": self.mean.clone(),
            "scale": self.scale.clone(),
            "spatial_dim": self.spatial_dim,
            "layout": self.layout,
            "fitted_trajectory_ids": list(self.fitted_trajectory_ids),
        }

    def matches_state_dict(self, state: Mapping[str, Any]) -> bool:
        """Compare fitted statistics exactly after canonicalizing tensors onto the CPU.

        Checkpoints may be loaded with a CUDA ``map_location`` even though normalizer
        statistics are intentionally device-independent. Direct dictionary equality is
        invalid for multi-element tensors and can also attempt cross-device comparison.
        """
        if not self.is_fitted:
            raise RuntimeError("normalizer must be fitted before state comparison")
        candidate = ChannelStandardizer(eps=self.eps)
        candidate.load_state_dict(state)
        assert self.mean is not None and self.scale is not None
        assert candidate.mean is not None and candidate.scale is not None
        return (
            self.eps == candidate.eps
            and self.spatial_dim == candidate.spatial_dim
            and self.layout == candidate.layout
            and self.fitted_trajectory_ids == candidate.fitted_trajectory_ids
            and torch.equal(self.mean.detach().cpu(), candidate.mean)
            and torch.equal(self.scale.detach().cpu(), candidate.scale)
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("kind") != "channel_standardizer":
            raise ValueError("unsupported normalizer state kind")
        mean, scale = state.get("mean"), state.get("scale")
        if not isinstance(mean, Tensor) or not isinstance(scale, Tensor):
            raise TypeError("normalizer mean and scale must be tensors")
        if mean.ndim != 1 or mean.shape != scale.shape or not torch.all(scale > 0):
            raise ValueError("invalid normalizer mean/scale shapes")
        layout = str(state.get("layout"))
        if layout not in {"channels_first", "channels_last"}:
            raise ValueError("invalid normalizer layout")
        self.eps = float(state["eps"])
        self.mean = mean.detach().cpu().to(torch.float64).clone()
        self.scale = scale.detach().cpu().to(torch.float64).clone()
        self.spatial_dim = int(state["spatial_dim"])
        self.layout = layout
        self.fitted_trajectory_ids = tuple(str(v) for v in state["fitted_trajectory_ids"])
