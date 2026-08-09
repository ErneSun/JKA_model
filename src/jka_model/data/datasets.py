"""Trajectory-level data records and validation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, overload

import torch
from torch import Tensor

from jka_model.contracts import DtMode, ProblemSpec, validate_trajectory_alignment


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    """One complete trajectory using the sole legal transition alignment.

    ``states_raw[i]`` and ``states_raw[i + 1]`` are joined by ``actions[i]``
    (when actions exist) and ``dts[i]``. No windowing occurs in this object.
    """

    trajectory_id: str
    states_raw: Tensor
    dts: Tensor
    actions: Tensor | None = None
    mu_static: Tensor | None = None
    coordinates: Tensor | None = None
    cell_weights: Tensor | None = None
    valid_mask: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trajectory_id.strip():
            raise ValueError("trajectory_id must not be empty")
        validate_trajectory_alignment(self.states_raw, self.actions, self.dts)
        if not torch.is_floating_point(self.states_raw):
            raise TypeError("states_raw must have a floating-point dtype")
        if not torch.isfinite(self.states_raw).all():
            raise ValueError("states_raw must contain only finite values")
        optional = {
            "mu_static": self.mu_static,
            "coordinates": self.coordinates,
            "cell_weights": self.cell_weights,
            "valid_mask": self.valid_mask,
        }
        for name, value in optional.items():
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if self.mu_static is not None and self.mu_static.ndim != 1:
            raise ValueError("mu_static must have shape [d_mu]")
        if self.cell_weights is not None:
            if self.cell_weights.numel() == 0 or not torch.all(self.cell_weights > 0):
                raise ValueError("cell_weights must be non-empty and positive")

    @property
    def num_steps(self) -> int:
        """Number of state transitions in the trajectory."""
        return self.dts.shape[0]


class TrajectoryDataset(Sequence[TrajectoryRecord]):
    """Immutable sequence with unique trajectory identifiers."""

    def __init__(self, records: Sequence[TrajectoryRecord]) -> None:
        self._records = tuple(records)
        if not self._records:
            raise ValueError("TrajectoryDataset requires at least one record")
        identifiers = [record.trajectory_id for record in self._records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("trajectory_id values must be unique")

    def __len__(self) -> int:
        return len(self._records)

    @overload
    def __getitem__(self, index: int) -> TrajectoryRecord: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TrajectoryRecord, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> TrajectoryRecord | tuple[TrajectoryRecord, ...]:
        return self._records[index]

    def __iter__(self) -> Iterator[TrajectoryRecord]:
        return iter(self._records)


def validate_trajectories_against_spec(
    records: Sequence[TrajectoryRecord], spec: ProblemSpec
) -> None:
    """Validate dynamic records against one static :class:`ProblemSpec`."""
    if not records:
        raise ValueError("at least one trajectory is required")
    channel_axis = 1 if spec.grid.layout == "channels_first" else -1
    expected_channels = len(spec.channels)
    identifiers: set[str] = set()
    for record in records:
        if record.trajectory_id in identifiers:
            raise ValueError(f"duplicate trajectory_id: {record.trajectory_id!r}")
        identifiers.add(record.trajectory_id)
        state = record.states_raw
        if state.ndim != spec.spatial_dim + 2:
            raise ValueError(
                f"trajectory {record.trajectory_id!r} state rank does not match spatial_dim"
            )
        if state.shape[channel_axis] != expected_channels:
            raise ValueError(
                f"trajectory {record.trajectory_id!r} channel count does not match spec"
            )
        spatial_shape = state.shape[2:] if channel_axis == 1 else state.shape[1:-1]
        if spec.grid.shape is not None and tuple(spatial_shape) != spec.grid.shape:
            raise ValueError(
                f"trajectory {record.trajectory_id!r} spatial shape does not match spec"
            )
        action_dim = 0 if record.actions is None else record.actions.shape[1]
        if action_dim != spec.action_dim:
            raise ValueError(
                f"trajectory {record.trajectory_id!r} action dimension does not match spec"
            )
        parameter_dim = 0 if record.mu_static is None else record.mu_static.numel()
        if parameter_dim != spec.parameter_dim:
            raise ValueError(
                f"trajectory {record.trajectory_id!r} parameter dimension does not match spec"
            )
        if spec.grid.coordinates_required and record.coordinates is None:
            raise ValueError("spec requires coordinates")
        if spec.grid.cell_weights_required and record.cell_weights is None:
            raise ValueError("spec requires cell_weights")
        if spec.geometry.mask_required and record.valid_mask is None:
            raise ValueError("spec requires valid_mask")
        if spec.dt_mode is DtMode.CONSTANT:
            assert spec.constant_dt is not None
            expected_dt = torch.full_like(record.dts, spec.constant_dt)
            if not torch.allclose(record.dts, expected_dt, rtol=0.0, atol=1e-12):
                raise ValueError("constant-dt spec received non-constant dts")
