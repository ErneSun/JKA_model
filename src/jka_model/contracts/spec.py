"""Serializable, immutable description of a physical problem."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


def _freeze_metadata(value: Any) -> Any:
    """Recursively freeze JSON-like metadata without changing its meaning."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"metadata values must be JSON-compatible, got {type(value).__name__}")


def _thaw_metadata(value: Any) -> Any:
    """Convert frozen metadata into plain JSON/YAML-compatible containers."""
    if isinstance(value, Mapping):
        return {str(key): _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], owner: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {owner} field(s): {names}")


class DtMode(str, Enum):
    """Whether transition time intervals are constant or sample-dependent."""

    CONSTANT = "constant"
    VARIABLE = "variable"


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """Physical meaning of one state channel."""

    name: str
    unit: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("channel name must not be empty")
        if not self.unit.strip():
            raise ValueError(f"unit for channel {self.name!r} must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "unit": self.unit}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChannelSpec:
        _reject_unknown(data, {"name", "unit"}, "ChannelSpec")
        return cls(name=str(data["name"]), unit=str(data["unit"]))


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Static grid/mesh contract; actual coordinates remain batch data."""

    layout: str = "channels_first"
    shape: tuple[int, ...] | None = None
    spacing: tuple[float, ...] | None = None
    coordinates_required: bool = False
    cell_weights_required: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layout not in {"channels_first", "channels_last"}:
            raise ValueError("grid layout must be 'channels_first' or 'channels_last'")
        if self.shape is not None and any(size <= 0 for size in self.shape):
            raise ValueError("grid shape entries must be positive")
        if self.spacing is not None and any(value <= 0 for value in self.spacing):
            raise ValueError("grid spacing entries must be positive")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "shape": None if self.shape is None else list(self.shape),
            "spacing": None if self.spacing is None else list(self.spacing),
            "coordinates_required": self.coordinates_required,
            "cell_weights_required": self.cell_weights_required,
            "metadata": _thaw_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GridSpec:
        allowed = {
            "layout",
            "shape",
            "spacing",
            "coordinates_required",
            "cell_weights_required",
            "metadata",
        }
        _reject_unknown(data, allowed, "GridSpec")
        shape = data.get("shape")
        spacing = data.get("spacing")
        return cls(
            layout=str(data.get("layout", "channels_first")),
            shape=None if shape is None else tuple(int(v) for v in shape),
            spacing=None if spacing is None else tuple(float(v) for v in spacing),
            coordinates_required=bool(data.get("coordinates_required", False)),
            cell_weights_required=bool(data.get("cell_weights_required", False)),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class BoundarySpec:
    """Boundary-condition metadata without embedding solver behavior."""

    kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("boundary kind must not be empty")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "metadata": _thaw_metadata(self.metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoundarySpec:
        _reject_unknown(data, {"kind", "metadata"}, "BoundarySpec")
        return cls(kind=str(data["kind"]), metadata=data.get("metadata", {}))


@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    """Declaration of preprocessing semantics, not fitted normalizer state."""

    method: str = "external"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ValueError("normalization method must not be empty")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "metadata": _thaw_metadata(self.metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NormalizationSpec:
        _reject_unknown(data, {"method", "metadata"}, "NormalizationSpec")
        return cls(method=str(data.get("method", "external")), metadata=data.get("metadata", {}))


@dataclass(frozen=True, slots=True)
class GeometrySpec:
    """Geometry/mask requirements; masks themselves belong to ProblemBatch."""

    mask_required: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {"mask_required": self.mask_required, "metadata": _thaw_metadata(self.metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GeometrySpec:
        _reject_unknown(data, {"mask_required", "metadata"}, "GeometrySpec")
        return cls(
            mask_required=bool(data.get("mask_required", False)),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Immutable static definition of one physical problem.

    This object never stores batch tensors or fitted normalization statistics. Those
    belong to :class:`ProblemBatch` and checkpoint ``normalizer_state`` respectively.
    """

    name: str
    channels: tuple[ChannelSpec, ...]
    spatial_dim: int
    grid: GridSpec
    boundary: BoundarySpec
    action_dim: int = 0
    parameter_dim: int = 0
    dt_mode: DtMode = DtMode.CONSTANT
    constant_dt: float | None = None
    normalization: NormalizationSpec = field(default_factory=NormalizationSpec)
    geometry: GeometrySpec = field(default_factory=GeometrySpec)
    observable_requirements: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("problem name must not be empty")
        if not self.channels:
            raise ValueError("ProblemSpec must declare at least one state channel")
        names = [channel.name for channel in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("state channel names must be unique")
        if self.spatial_dim < 0:
            raise ValueError("spatial_dim must be non-negative")
        if self.action_dim < 0 or self.parameter_dim < 0:
            raise ValueError("action_dim and parameter_dim must be non-negative")
        if self.grid.shape is not None and len(self.grid.shape) != self.spatial_dim:
            raise ValueError("grid shape length must equal spatial_dim")
        if self.grid.spacing is not None and len(self.grid.spacing) != self.spatial_dim:
            raise ValueError("grid spacing length must equal spatial_dim")
        if self.dt_mode is DtMode.CONSTANT:
            if self.constant_dt is None or self.constant_dt <= 0:
                raise ValueError("constant dt mode requires a positive constant_dt")
        elif self.constant_dt is not None:
            raise ValueError("variable dt mode must not set constant_dt")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    @property
    def units(self) -> tuple[str, ...]:
        return tuple(channel.unit for channel in self.channels)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, stable, JSON/YAML-compatible representation."""
        return {
            "name": self.name,
            "channels": [channel.to_dict() for channel in self.channels],
            "spatial_dim": self.spatial_dim,
            "grid": self.grid.to_dict(),
            "boundary": self.boundary.to_dict(),
            "action_dim": self.action_dim,
            "parameter_dim": self.parameter_dim,
            "dt_mode": self.dt_mode.value,
            "constant_dt": self.constant_dt,
            "normalization": self.normalization.to_dict(),
            "geometry": self.geometry.to_dict(),
            "observable_requirements": list(self.observable_requirements),
            "metadata": _thaw_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProblemSpec:
        allowed = {
            "name",
            "channels",
            "spatial_dim",
            "grid",
            "boundary",
            "action_dim",
            "parameter_dim",
            "dt_mode",
            "constant_dt",
            "normalization",
            "geometry",
            "observable_requirements",
            "metadata",
        }
        _reject_unknown(data, allowed, "ProblemSpec")
        return cls(
            name=str(data["name"]),
            channels=tuple(ChannelSpec.from_dict(item) for item in data["channels"]),
            spatial_dim=int(data["spatial_dim"]),
            grid=GridSpec.from_dict(data["grid"]),
            boundary=BoundarySpec.from_dict(data["boundary"]),
            action_dim=int(data.get("action_dim", 0)),
            parameter_dim=int(data.get("parameter_dim", 0)),
            dt_mode=DtMode(str(data.get("dt_mode", DtMode.CONSTANT.value))),
            constant_dt=(None if data.get("constant_dt") is None else float(data["constant_dt"])),
            normalization=NormalizationSpec.from_dict(data.get("normalization", {})),
            geometry=GeometrySpec.from_dict(data.get("geometry", {})),
            observable_requirements=tuple(str(v) for v in data.get("observable_requirements", ())),
            metadata=data.get("metadata", {}),
        )
