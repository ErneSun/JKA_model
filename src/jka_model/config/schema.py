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
    """Architecture identity shared by all versioned experiments."""

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
    """Run controls shared by all versioned experiments."""

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
class SplitConfig:
    """Deterministic trajectory-level split settings."""

    train: float = 0.67
    validation: float = 0.17
    test: float = 0.16
    seed: int = 42

    def __post_init__(self) -> None:
        ratios = (self.train, self.validation, self.test)
        if any(ratio < 0 or ratio > 1 for ratio in ratios):
            raise ValueError("split ratios must lie in [0, 1]")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("train/validation/test split ratios must sum to 1")
        if self.train <= 0:
            raise ValueError("training split ratio must be positive")
        if self.seed < 0:
            raise ValueError("split seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SplitConfig:
        _reject_unknown(data, {"train", "validation", "test", "seed"}, "split config")
        return cls(
            train=float(data.get("train", 0.67)),
            validation=float(data.get("validation", 0.17)),
            test=float(data.get("test", 0.16)),
            seed=int(data.get("seed", 42)),
        )


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Channel-wise state normalization settings."""

    kind: str = "standard"
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.kind not in {"standard", "external"}:
            raise ValueError("normalization kind must be 'standard' or 'external'")
        if self.eps <= 0:
            raise ValueError("normalization eps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "eps": self.eps}

    @classmethod
    def from_value(cls, value: Any) -> NormalizationConfig:
        if isinstance(value, str):
            aliases = {
                "standard": "standard",
                "standardize": "standard",
                "standardize_from_train_split": "standard",
                "external": "external",
            }
            if value not in aliases:
                raise ValueError(f"unsupported normalization declaration: {value!r}")
            return cls(kind=aliases[value])
        data = _ensure_mapping(value, "normalization config")
        _reject_unknown(data, {"kind", "eps"}, "normalization config")
        return cls(kind=str(data.get("kind", "standard")), eps=float(data.get("eps", 1e-6)))


@dataclass(frozen=True, slots=True)
class ToyAdvectionDiffusionConfig:
    """Small analytic 1D periodic advection-diffusion dataset settings."""

    num_trajectories: int = 12
    num_steps: int = 16
    nx: int = 65
    length: float = 6.283185307179586
    base_dt: float = 0.03
    variable_dt: bool = True
    modes: int = 3
    c_min: float = 0.5
    c_max: float = 1.2
    nu_min: float = 0.01
    nu_max: float = 0.05
    enable_probe: bool = True

    def __post_init__(self) -> None:
        if self.num_trajectories < 1 or self.num_steps < 1:
            raise ValueError("toy num_trajectories and num_steps must be positive")
        if self.nx < 5:
            raise ValueError("toy nx must be at least 5")
        if self.length <= 0 or self.base_dt <= 0:
            raise ValueError("toy length and base_dt must be positive")
        if self.modes < 1 or 2 * self.modes >= self.nx - 1:
            raise ValueError("toy modes must be below the spatial Nyquist limit")
        if self.c_min > self.c_max:
            raise ValueError("toy c_min must not exceed c_max")
        if self.nu_min < 0 or self.nu_min > self.nu_max:
            raise ValueError("toy diffusivity range must satisfy 0 <= nu_min <= nu_max")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_trajectories": self.num_trajectories,
            "num_steps": self.num_steps,
            "nx": self.nx,
            "length": self.length,
            "base_dt": self.base_dt,
            "variable_dt": self.variable_dt,
            "modes": self.modes,
            "c_min": self.c_min,
            "c_max": self.c_max,
            "nu_min": self.nu_min,
            "nu_max": self.nu_max,
            "enable_probe": self.enable_probe,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToyAdvectionDiffusionConfig:
        allowed = {
            "num_trajectories",
            "num_steps",
            "nx",
            "length",
            "base_dt",
            "variable_dt",
            "modes",
            "c_min",
            "c_max",
            "nu_min",
            "nu_max",
            "enable_probe",
        }
        _reject_unknown(data, allowed, "toy advection-diffusion config")
        defaults = cls()
        return cls(
            num_trajectories=int(data.get("num_trajectories", defaults.num_trajectories)),
            num_steps=int(data.get("num_steps", defaults.num_steps)),
            nx=int(data.get("nx", defaults.nx)),
            length=float(data.get("length", defaults.length)),
            base_dt=float(data.get("base_dt", defaults.base_dt)),
            variable_dt=bool(data.get("variable_dt", defaults.variable_dt)),
            modes=int(data.get("modes", defaults.modes)),
            c_min=float(data.get("c_min", defaults.c_min)),
            c_max=float(data.get("c_max", defaults.c_max)),
            nu_min=float(data.get("nu_min", defaults.nu_min)),
            nu_max=float(data.get("nu_max", defaults.nu_max)),
            enable_probe=bool(data.get("enable_probe", defaults.enable_probe)),
        )


@dataclass(frozen=True, slots=True)
class KoopmanConfig:
    """Direct-state continuous-time generator settings."""

    state_dim: int = 2
    trainable: bool = True
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.state_dim < 1:
            raise ValueError("Koopman state_dim must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("Koopman dtype must be 'float32' or 'float64'")

    def to_dict(self) -> dict[str, Any]:
        return {"state_dim": self.state_dim, "trainable": self.trainable, "dtype": self.dtype}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KoopmanConfig:
        _reject_unknown(data, {"state_dim", "trainable", "dtype"}, "Koopman config")
        return cls(
            state_dim=int(data.get("state_dim", 2)),
            trainable=bool(data.get("trainable", True)),
            dtype=str(data.get("dtype", "float64")),
        )


@dataclass(frozen=True, slots=True)
class DampedOscillatorConfig:
    """Reference damped harmonic oscillator data settings."""

    omega0: float = 2.0
    gamma: float = 0.15
    base_dt: float = 0.04
    variable_dt: bool = True
    dt_jitter: float = 0.25
    num_steps: int = 160
    num_trajectories: int = 12

    def __post_init__(self) -> None:
        if self.omega0 <= 0 or not 0 < self.gamma < self.omega0:
            raise ValueError("oscillator requires omega0 > gamma > 0")
        if self.base_dt <= 0 or self.num_steps < 1 or self.num_trajectories < 1:
            raise ValueError("oscillator dt, steps, and trajectories must be positive")
        if not 0 <= self.dt_jitter < 1:
            raise ValueError("oscillator dt_jitter must lie in [0, 1)")
        if not self.variable_dt and self.dt_jitter != 0:
            raise ValueError("constant-dt oscillator requires dt_jitter=0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "omega0": self.omega0,
            "gamma": self.gamma,
            "base_dt": self.base_dt,
            "variable_dt": self.variable_dt,
            "dt_jitter": self.dt_jitter,
            "num_steps": self.num_steps,
            "num_trajectories": self.num_trajectories,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DampedOscillatorConfig:
        allowed = {
            "omega0",
            "gamma",
            "base_dt",
            "variable_dt",
            "dt_jitter",
            "num_steps",
            "num_trajectories",
        }
        _reject_unknown(data, allowed, "damped oscillator config")
        defaults = cls()
        return cls(
            omega0=float(data.get("omega0", defaults.omega0)),
            gamma=float(data.get("gamma", defaults.gamma)),
            base_dt=float(data.get("base_dt", defaults.base_dt)),
            variable_dt=bool(data.get("variable_dt", defaults.variable_dt)),
            dt_jitter=float(data.get("dt_jitter", defaults.dt_jitter)),
            num_steps=int(data.get("num_steps", defaults.num_steps)),
            num_trajectories=int(data.get("num_trajectories", defaults.num_trajectories)),
        )


@dataclass(frozen=True, slots=True)
class DuffingConfig:
    """Unforced Duffing reference-data settings."""

    delta: float = 0.2
    alpha: float = 1.0
    beta: float = 0.5
    dt: float = 0.02
    num_steps: int = 160
    num_trajectories: int = 8
    rk4_substeps: int = 2

    def __post_init__(self) -> None:
        if self.delta < 0 or self.alpha <= 0 or self.beta == 0:
            raise ValueError("Duffing requires delta >= 0, alpha > 0, and beta != 0")
        if self.dt <= 0 or self.num_steps < 1 or self.num_trajectories < 1:
            raise ValueError("Duffing dt, steps, and trajectories must be positive")
        if self.rk4_substeps < 1:
            raise ValueError("Duffing rk4_substeps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta": self.delta,
            "alpha": self.alpha,
            "beta": self.beta,
            "dt": self.dt,
            "num_steps": self.num_steps,
            "num_trajectories": self.num_trajectories,
            "rk4_substeps": self.rk4_substeps,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DuffingConfig:
        allowed = {
            "delta",
            "alpha",
            "beta",
            "dt",
            "num_steps",
            "num_trajectories",
            "rk4_substeps",
        }
        _reject_unknown(data, allowed, "Duffing config")
        defaults = cls()
        return cls(
            delta=float(data.get("delta", defaults.delta)),
            alpha=float(data.get("alpha", defaults.alpha)),
            beta=float(data.get("beta", defaults.beta)),
            dt=float(data.get("dt", defaults.dt)),
            num_steps=int(data.get("num_steps", defaults.num_steps)),
            num_trajectories=int(data.get("num_trajectories", defaults.num_trajectories)),
            rk4_substeps=int(data.get("rk4_substeps", defaults.rk4_substeps)),
        )


@dataclass(frozen=True, slots=True)
class DirectIdentificationConfig:
    """Minimal optimizer settings for learning only the generator matrix."""

    epochs: int = 700
    learning_rate: float = 0.03
    init_scale: float = 0.05
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0 or self.init_scale < 0:
            raise ValueError(
                "identification epochs/lr must be positive and init_scale non-negative"
            )
        if self.weight_decay < 0:
            raise ValueError("identification weight_decay must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "init_scale": self.init_scale,
            "weight_decay": self.weight_decay,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DirectIdentificationConfig:
        allowed = {"epochs", "learning_rate", "init_scale", "weight_decay"}
        _reject_unknown(data, allowed, "direct identification config")
        defaults = cls()
        return cls(
            epochs=int(data.get("epochs", defaults.epochs)),
            learning_rate=float(data.get("learning_rate", defaults.learning_rate)),
            init_scale=float(data.get("init_scale", defaults.init_scale)),
            weight_decay=float(data.get("weight_decay", defaults.weight_decay)),
        )


@dataclass(frozen=True, slots=True)
class KoopmanEvaluationConfig:
    """V0.3 deterministic rollout evaluation settings."""

    rollout_horizon: int = 100

    def __post_init__(self) -> None:
        if self.rollout_horizon < 1:
            raise ValueError("rollout_horizon must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {"rollout_horizon": self.rollout_horizon}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KoopmanEvaluationConfig:
        _reject_unknown(data, {"rollout_horizon"}, "Koopman evaluation config")
        return cls(rollout_horizon=int(data.get("rollout_horizon", 100)))


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Static data expectations shared by V0.2+ pipelines."""

    problem_name: str
    action_dim: int = 0
    parameter_dim: int = 0
    dt_mode: DtMode = DtMode.CONSTANT
    constant_dt: float | None = None
    history: int = 4
    horizon: int = 2
    split: SplitConfig = field(default_factory=SplitConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    toy_advection_diffusion: ToyAdvectionDiffusionConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.normalization, str):
            object.__setattr__(
                self,
                "normalization",
                NormalizationConfig.from_value(self.normalization),
            )
        if not self.problem_name.strip():
            raise ValueError("problem_name must not be empty")
        if self.action_dim < 0 or self.parameter_dim < 0:
            raise ValueError("action_dim and parameter_dim must be non-negative")
        if self.dt_mode is DtMode.CONSTANT:
            if self.constant_dt is None or self.constant_dt <= 0:
                raise ValueError("constant dt mode requires a positive constant_dt")
        elif self.constant_dt is not None:
            raise ValueError("variable dt mode must not set constant_dt")
        if self.history < 2 or self.horizon < 1:
            raise ValueError("history must be at least 2 and horizon must be positive")
        toy = self.toy_advection_diffusion
        if toy is not None:
            if toy.num_steps < self.history + self.horizon - 1:
                raise ValueError("toy num_steps is too short for the configured history/horizon")
            if self.action_dim != 0 or self.parameter_dim != 2:
                raise ValueError(
                    "toy advection-diffusion requires action_dim=0 and parameter_dim=2"
                )
            if self.dt_mode is DtMode.VARIABLE and not toy.variable_dt:
                raise ValueError("variable dt data config requires toy variable_dt=true")
            if self.dt_mode is DtMode.CONSTANT:
                if toy.variable_dt:
                    raise ValueError("constant dt data config requires toy variable_dt=false")
                assert self.constant_dt is not None
                if abs(float(self.constant_dt) - toy.base_dt) > 1e-12:
                    raise ValueError("constant_dt must match toy base_dt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_name": self.problem_name,
            "action_dim": self.action_dim,
            "parameter_dim": self.parameter_dim,
            "dt_mode": self.dt_mode.value,
            "constant_dt": self.constant_dt,
            "history": self.history,
            "horizon": self.horizon,
            "split": self.split.to_dict(),
            "normalization": self.normalization.to_dict(),
            "toy_advection_diffusion": (
                None
                if self.toy_advection_diffusion is None
                else self.toy_advection_diffusion.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DataConfig:
        allowed = {
            "problem_name",
            "action_dim",
            "parameter_dim",
            "dt_mode",
            "constant_dt",
            "history",
            "horizon",
            "split",
            "normalization",
            "toy_advection_diffusion",
        }
        _reject_unknown(data, allowed, "data config")
        toy_data = data.get("toy_advection_diffusion")
        return cls(
            problem_name=str(data["problem_name"]),
            action_dim=int(data.get("action_dim", 0)),
            parameter_dim=int(data.get("parameter_dim", 0)),
            dt_mode=DtMode(str(data.get("dt_mode", DtMode.CONSTANT.value))),
            constant_dt=(None if data.get("constant_dt") is None else float(data["constant_dt"])),
            history=int(data.get("history", 4)),
            horizon=int(data.get("horizon", 2)),
            split=SplitConfig.from_dict(
                _ensure_mapping(data.get("split", {}), "split config")
            ),
            normalization=NormalizationConfig.from_value(data.get("normalization", "external")),
            toy_advection_diffusion=(
                None
                if toy_data is None
                else ToyAdvectionDiffusionConfig.from_dict(
                    _ensure_mapping(toy_data, "toy advection-diffusion config")
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Resolved configuration with optional V0.3 direct-state sections."""

    architecture: ArchitectureConfig
    training: TrainingConfig
    data: DataConfig
    koopman: KoopmanConfig | None = None
    oscillator: DampedOscillatorConfig | None = None
    duffing: DuffingConfig | None = None
    identification: DirectIdentificationConfig | None = None
    evaluation: KoopmanEvaluationConfig | None = None
    project_version: str = PROJECT_VERSION
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.project_version != PROJECT_VERSION:
            raise ValueError(
                f"config project version {self.project_version!r} does not match "
                f"runtime version {PROJECT_VERSION!r}"
            )
        v0_3_sections = (
            self.koopman,
            self.oscillator,
            self.duffing,
            self.identification,
            self.evaluation,
        )
        if any(section is not None for section in v0_3_sections) and not all(
            section is not None for section in v0_3_sections
        ):
            raise ValueError("V0.3 config must provide all direct-state sections together")
        if self.koopman is not None and self.koopman.state_dim != 2:
            raise ValueError("V0.3 oscillator experiments require Koopman state_dim=2")
        if self.koopman is not None:
            assert self.oscillator is not None
            expected_mode = (
                DtMode.VARIABLE if self.oscillator.variable_dt else DtMode.CONSTANT
            )
            if self.data.problem_name != "damped_harmonic_oscillator":
                raise ValueError("V0.3 data problem_name must be damped_harmonic_oscillator")
            if self.data.action_dim != 0 or self.data.parameter_dim != 2:
                raise ValueError("V0.3 oscillator data requires action_dim=0 and parameter_dim=2")
            if self.data.dt_mode is not expected_mode:
                raise ValueError("V0.3 data dt_mode must match oscillator variable_dt")
            if expected_mode is DtMode.CONSTANT:
                assert self.data.constant_dt is not None
                if abs(self.data.constant_dt - self.oscillator.base_dt) > 1e-12:
                    raise ValueError("V0.3 constant_dt must match oscillator base_dt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
            "koopman": None if self.koopman is None else self.koopman.to_dict(),
            "oscillator": None if self.oscillator is None else self.oscillator.to_dict(),
            "duffing": None if self.duffing is None else self.duffing.to_dict(),
            "identification": (
                None if self.identification is None else self.identification.to_dict()
            ),
            "evaluation": None if self.evaluation is None else self.evaluation.to_dict(),
            "project_version": self.project_version,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProjectConfig:
        _reject_unknown(
            data,
            {
                "architecture",
                "training",
                "data",
                "koopman",
                "oscillator",
                "duffing",
                "identification",
                "evaluation",
                "project_version",
                "tags",
            },
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
            koopman=(
                None
                if data.get("koopman") is None
                else KoopmanConfig.from_dict(_ensure_mapping(data["koopman"], "Koopman config"))
            ),
            oscillator=(
                None
                if data.get("oscillator") is None
                else DampedOscillatorConfig.from_dict(
                    _ensure_mapping(data["oscillator"], "damped oscillator config")
                )
            ),
            duffing=(
                None
                if data.get("duffing") is None
                else DuffingConfig.from_dict(_ensure_mapping(data["duffing"], "Duffing config"))
            ),
            identification=(
                None
                if data.get("identification") is None
                else DirectIdentificationConfig.from_dict(
                    _ensure_mapping(data["identification"], "direct identification config")
                )
            ),
            evaluation=(
                None
                if data.get("evaluation") is None
                else KoopmanEvaluationConfig.from_dict(
                    _ensure_mapping(data["evaluation"], "Koopman evaluation config")
                )
            ),
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
