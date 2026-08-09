"""Small strict dataclass configuration system with stable hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jka_model.constants import ARCHITECTURE_REVISION, PROJECT_VERSION
from jka_model.contracts import DtMode
from jka_model.training import TrainStage


def _ensure_mapping(value: Any, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{owner} must be a mapping")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], owner: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown {owner} field(s): {', '.join(sorted(unknown))}")


@dataclass(frozen=True, slots=True)
class ArchitectureConfig:
    """Architecture identity only; V0.1 has no neural hyperparameters."""

    revision: str = ARCHITECTURE_REVISION
    package: str = "jka_model"

    def __post_init__(self) -> None:
        if self.revision != ARCHITECTURE_REVISION:
            raise ValueError(
                f"architecture revision {self.revision!r} is incompatible with "
                f"runtime revision {ARCHITECTURE_REVISION!r}"
            )
        if not self.package.strip():
            raise ValueError("architecture package must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {"revision": self.revision, "package": self.package}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArchitectureConfig:
        _reject_unknown(data, {"revision", "package"}, "architecture config")
        return cls(
            revision=str(data.get("revision", ARCHITECTURE_REVISION)),
            package=str(data.get("package", "jka_model")),
        )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Run controls needed before any trainer exists."""

    seed: int = 0
    stage: TrainStage = TrainStage.KOOPMAN
    deterministic: bool = True
    run_root: str = "runs"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("training seed must be non-negative")
        if not self.run_root.strip():
            raise ValueError("run_root must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "stage": self.stage.value,
            "deterministic": self.deterministic,
            "run_root": self.run_root,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainingConfig:
        _reject_unknown(data, {"seed", "stage", "deterministic", "run_root"}, "training config")
        return cls(
            seed=int(data.get("seed", 0)),
            stage=TrainStage(str(data.get("stage", TrainStage.KOOPMAN.value))),
            deterministic=bool(data.get("deterministic", True)),
            run_root=str(data.get("run_root", "runs")),
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Static data expectations; loading/window generation begins in V0.2."""

    problem_name: str
    action_dim: int = 0
    parameter_dim: int = 0
    dt_mode: DtMode = DtMode.CONSTANT
    constant_dt: float | None = None
    normalization: str = "external"

    def __post_init__(self) -> None:
        if not self.problem_name.strip():
            raise ValueError("problem_name must not be empty")
        if self.action_dim < 0 or self.parameter_dim < 0:
            raise ValueError("action_dim and parameter_dim must be non-negative")
        if self.dt_mode is DtMode.CONSTANT:
            if self.constant_dt is None or self.constant_dt <= 0:
                raise ValueError("constant dt mode requires a positive constant_dt")
        elif self.constant_dt is not None:
            raise ValueError("variable dt mode must not set constant_dt")
        if not self.normalization.strip():
            raise ValueError("normalization declaration must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_name": self.problem_name,
            "action_dim": self.action_dim,
            "parameter_dim": self.parameter_dim,
            "dt_mode": self.dt_mode.value,
            "constant_dt": self.constant_dt,
            "normalization": self.normalization,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DataConfig:
        allowed = {
            "problem_name",
            "action_dim",
            "parameter_dim",
            "dt_mode",
            "constant_dt",
            "normalization",
        }
        _reject_unknown(data, allowed, "data config")
        return cls(
            problem_name=str(data["problem_name"]),
            action_dim=int(data.get("action_dim", 0)),
            parameter_dim=int(data.get("parameter_dim", 0)),
            dt_mode=DtMode(str(data.get("dt_mode", DtMode.CONSTANT.value))),
            constant_dt=(None if data.get("constant_dt") is None else float(data["constant_dt"])),
            normalization=str(data.get("normalization", "external")),
        )


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Resolved V0.1 configuration, split by architecture/training/data concerns."""

    architecture: ArchitectureConfig
    training: TrainingConfig
    data: DataConfig
    project_version: str = PROJECT_VERSION
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.project_version != PROJECT_VERSION:
            raise ValueError(
                f"config project version {self.project_version!r} does not match "
                f"runtime version {PROJECT_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
            "project_version": self.project_version,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProjectConfig:
        _reject_unknown(
            data,
            {"architecture", "training", "data", "project_version", "tags"},
            "project config",
        )
        return cls(
            architecture=ArchitectureConfig.from_dict(
                _ensure_mapping(data.get("architecture", {}), "architecture config")
            ),
            training=TrainingConfig.from_dict(
                _ensure_mapping(data.get("training", {}), "training config")
            ),
            data=DataConfig.from_dict(_ensure_mapping(data["data"], "data config")),
            project_version=str(data.get("project_version", PROJECT_VERSION)),
            tags=tuple(str(tag) for tag in data.get("tags", ())),
        )

    @property
    def stable_hash(self) -> str:
        return stable_config_hash(self)


def stable_config_hash(config: ProjectConfig | Mapping[str, Any]) -> str:
    """SHA-256 of a canonical, fully resolved config representation."""
    payload = config.to_dict() if isinstance(config, ProjectConfig) else dict(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def save_config(config: ProjectConfig, path: str | Path) -> None:
    """Save a resolved config as deterministic, human-readable YAML."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )


def load_config(path: str | Path) -> ProjectConfig:
    """Load and strictly validate a YAML configuration."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ProjectConfig.from_dict(_ensure_mapping(data, "project config"))
