"""Optional diagnostics that observe raw physical states without changing them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from jka_model.contracts import ProblemBatch, ProblemSpec


class PhysicalProbe(Protocol):
    name: str

    def measure(self, state_raw: Tensor, spec: ProblemSpec) -> Tensor: ...


def _spatial_axes(state_raw: Tensor, spec: ProblemSpec) -> tuple[int, ...]:
    if state_raw.ndim < spec.spatial_dim + 2:
        raise ValueError("probe state must have batch, channel, and spatial axes")
    if spec.grid.layout == "channels_first":
        channel_axis = state_raw.ndim - spec.spatial_dim - 1
        axes = tuple(range(state_raw.ndim - spec.spatial_dim, state_raw.ndim))
    else:
        channel_axis = state_raw.ndim - 1
        axes = tuple(
            range(state_raw.ndim - spec.spatial_dim - 1, state_raw.ndim - 1)
        )
    if state_raw.shape[channel_axis] != len(spec.channels):
        raise ValueError("probe state channel count does not match ProblemSpec")
    return axes


@dataclass(frozen=True, slots=True)
class ChannelMeanProbe:
    name: str = "channel_mean"

    def measure(self, state_raw: Tensor, spec: ProblemSpec) -> Tensor:
        return state_raw.mean(dim=_spatial_axes(state_raw, spec))


@dataclass(frozen=True, slots=True)
class ChannelRMSProbe:
    name: str = "channel_rms"

    def measure(self, state_raw: Tensor, spec: ProblemSpec) -> Tensor:
        return state_raw.square().mean(dim=_spatial_axes(state_raw, spec)).sqrt()


def evaluate_probes(
    probes: Sequence[PhysicalProbe], state_raw: Tensor, spec: ProblemSpec
) -> Mapping[str, Tensor]:
    measurements: dict[str, Tensor] = {}
    for probe in probes:
        if probe.name in measurements:
            raise ValueError(f"duplicate probe name: {probe.name!r}")
        measurements[probe.name] = probe.measure(state_raw, spec)
    return measurements


def evaluate_batch_probes(
    probes: Sequence[PhysicalProbe],
    batch: ProblemBatch,
    spec: ProblemSpec,
    *,
    future_index: int = 0,
) -> Mapping[str, Tensor]:
    """Route only canonical raw-unit batch state into physical probes."""
    if not 0 <= future_index < batch.future_states_raw.shape[1]:
        raise IndexError("future_index is outside the batch horizon")
    return evaluate_probes(probes, batch.future_states_raw[:, future_index], spec)
