"""Small strict dataclass configuration system with stable hashing."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jka_model.constants import (
    ARCHITECTURE_REVISION,
    PROJECT_VERSION,
    SUPPORTED_CONFIG_PROJECT_VERSIONS,
    V0_6_PROJECT_VERSION,
    V0_7_PROJECT_VERSION,
    V0_8_PROJECT_VERSION,
)
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
class KnownLatentConfig:
    """Known-latent nonlinear-observation synthetic system settings."""

    alpha: float = 0.08
    omega: float = 1.6
    base_dt: float = 0.05
    variable_dt: bool = True
    dt_jitter: float = 0.2
    num_steps: int = 72
    num_trajectories: int = 18

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.omega <= 0 or self.base_dt <= 0:
            raise ValueError("known-latent alpha, omega, and base_dt must be positive")
        if not 0 <= self.dt_jitter < 1:
            raise ValueError("known-latent dt_jitter must lie in [0,1)")
        if not self.variable_dt and self.dt_jitter != 0:
            raise ValueError("constant-dt known-latent data requires dt_jitter=0")
        if self.num_steps < 2 or self.num_trajectories < 3:
            raise ValueError("known-latent data requires at least 2 steps and 3 trajectories")

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "omega": self.omega,
            "base_dt": self.base_dt,
            "variable_dt": self.variable_dt,
            "dt_jitter": self.dt_jitter,
            "num_steps": self.num_steps,
            "num_trajectories": self.num_trajectories,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KnownLatentConfig:
        allowed = {
            "alpha",
            "omega",
            "base_dt",
            "variable_dt",
            "dt_jitter",
            "num_steps",
            "num_trajectories",
        }
        _reject_unknown(data, allowed, "known-latent config")
        defaults = cls()
        return cls(
            alpha=float(data.get("alpha", defaults.alpha)),
            omega=float(data.get("omega", defaults.omega)),
            base_dt=float(data.get("base_dt", defaults.base_dt)),
            variable_dt=bool(data.get("variable_dt", defaults.variable_dt)),
            dt_jitter=float(data.get("dt_jitter", defaults.dt_jitter)),
            num_steps=int(data.get("num_steps", defaults.num_steps)),
            num_trajectories=int(data.get("num_trajectories", defaults.num_trajectories)),
        )


@dataclass(frozen=True, slots=True)
class KoopmanAutoencoderConfig:
    """Small vector MLP architecture for V0.4 learned coordinates."""

    observation_dim: int = 5
    latent_dim: int = 2
    hidden_dim: int = 32
    encoder_hidden_layers: int = 1
    decoder_hidden_layers: int = 2
    activation: str = "tanh"

    def __post_init__(self) -> None:
        if self.observation_dim < 1 or self.latent_dim < 1 or self.hidden_dim < 1:
            raise ValueError("autoencoder dimensions must be positive")
        if self.encoder_hidden_layers not in {0, 1, 2}:
            raise ValueError("V0.4 encoder_hidden_layers must be 0, 1, or 2")
        if self.decoder_hidden_layers not in {1, 2}:
            raise ValueError("V0.4 decoder_hidden_layers must be 1 or 2")
        if self.activation not in {"tanh", "silu"}:
            raise ValueError("autoencoder activation must be tanh or silu")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_dim": self.observation_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "encoder_hidden_layers": self.encoder_hidden_layers,
            "decoder_hidden_layers": self.decoder_hidden_layers,
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KoopmanAutoencoderConfig:
        allowed = {
            "observation_dim",
            "latent_dim",
            "hidden_dim",
            "encoder_hidden_layers",
            "decoder_hidden_layers",
            "activation",
        }
        _reject_unknown(data, allowed, "Koopman autoencoder config")
        defaults = cls()
        return cls(
            observation_dim=int(data.get("observation_dim", defaults.observation_dim)),
            latent_dim=int(data.get("latent_dim", defaults.latent_dim)),
            hidden_dim=int(data.get("hidden_dim", defaults.hidden_dim)),
            encoder_hidden_layers=int(
                data.get("encoder_hidden_layers", defaults.encoder_hidden_layers)
            ),
            decoder_hidden_layers=int(
                data.get("decoder_hidden_layers", defaults.decoder_hidden_layers)
            ),
            activation=str(data.get("activation", defaults.activation)),
        )


@dataclass(frozen=True, slots=True)
class RepresentationLossConfig:
    """Independent V0.4 representation-loss weights."""

    lambda_k: float = 1.0
    lambda_multi: float = 1.0
    lambda_rec: float = 1.0
    lambda_var: float = 0.2
    lambda_spec: float = 0.0
    min_std: float = 0.15
    stability_margin: float = 0.0

    def __post_init__(self) -> None:
        weights = (
            self.lambda_k,
            self.lambda_multi,
            self.lambda_rec,
            self.lambda_var,
            self.lambda_spec,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("representation loss weights must be non-negative and nonzero")
        if self.min_std <= 0:
            raise ValueError("representation min_std must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lambda_k": self.lambda_k,
            "lambda_multi": self.lambda_multi,
            "lambda_rec": self.lambda_rec,
            "lambda_var": self.lambda_var,
            "lambda_spec": self.lambda_spec,
            "min_std": self.min_std,
            "stability_margin": self.stability_margin,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RepresentationLossConfig:
        allowed = {
            "lambda_k",
            "lambda_multi",
            "lambda_rec",
            "lambda_var",
            "lambda_spec",
            "min_std",
            "stability_margin",
        }
        _reject_unknown(data, allowed, "representation loss config")
        defaults = cls()
        return cls(
            lambda_k=float(data.get("lambda_k", defaults.lambda_k)),
            lambda_multi=float(data.get("lambda_multi", defaults.lambda_multi)),
            lambda_rec=float(data.get("lambda_rec", defaults.lambda_rec)),
            lambda_var=float(data.get("lambda_var", defaults.lambda_var)),
            lambda_spec=float(data.get("lambda_spec", defaults.lambda_spec)),
            min_std=float(data.get("min_std", defaults.min_std)),
            stability_margin=float(data.get("stability_margin", defaults.stability_margin)),
        )


@dataclass(frozen=True, slots=True)
class RepresentationTrainingConfig:
    """Minimal optimizer and diagnostic-run controls for V0.4."""

    epochs: int = 700
    batch_size: int = 256
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    init_scale: float = 0.05
    ablation_epochs: int = 160
    duffing_epochs: int = 300
    diagnostic_interval: int = 100

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 2 or self.learning_rate <= 0:
            raise ValueError("representation epochs/batch_size/lr must be positive")
        if (
            self.weight_decay < 0
            or self.init_scale < 0
            or self.ablation_epochs < 1
            or self.duffing_epochs < 1
            or self.diagnostic_interval < 1
        ):
            raise ValueError("representation regularization/diagnostic epochs are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "init_scale": self.init_scale,
            "ablation_epochs": self.ablation_epochs,
            "duffing_epochs": self.duffing_epochs,
            "diagnostic_interval": self.diagnostic_interval,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RepresentationTrainingConfig:
        allowed = {
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "init_scale",
            "ablation_epochs",
            "duffing_epochs",
            "diagnostic_interval",
        }
        _reject_unknown(data, allowed, "representation training config")
        defaults = cls()
        return cls(
            epochs=int(data.get("epochs", defaults.epochs)),
            batch_size=int(data.get("batch_size", defaults.batch_size)),
            learning_rate=float(data.get("learning_rate", defaults.learning_rate)),
            weight_decay=float(data.get("weight_decay", defaults.weight_decay)),
            init_scale=float(data.get("init_scale", defaults.init_scale)),
            ablation_epochs=int(data.get("ablation_epochs", defaults.ablation_epochs)),
            duffing_epochs=int(data.get("duffing_epochs", defaults.duffing_epochs)),
            diagnostic_interval=int(data.get("diagnostic_interval", defaults.diagnostic_interval)),
        )


@dataclass(frozen=True, slots=True)
class RepresentationEvaluationConfig:
    """Held-out V0.4 evaluation controls."""

    rollout_horizon: int = 60
    max_test_reconstruction_mse: float = 1e-3
    min_alignment_r2: float = 0.98
    max_frequency_relative_error: float = 0.02
    min_latent_std: float = 0.01

    def __post_init__(self) -> None:
        if self.rollout_horizon < 2:
            raise ValueError("representation rollout_horizon must be at least 2")
        if self.max_test_reconstruction_mse <= 0:
            raise ValueError("maximum reconstruction MSE must be positive")
        if not 0 < self.min_alignment_r2 <= 1:
            raise ValueError("minimum alignment R2 must lie in (0,1]")
        if self.max_frequency_relative_error <= 0 or self.min_latent_std <= 0:
            raise ValueError("frequency error and latent std thresholds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_horizon": self.rollout_horizon,
            "max_test_reconstruction_mse": self.max_test_reconstruction_mse,
            "min_alignment_r2": self.min_alignment_r2,
            "max_frequency_relative_error": self.max_frequency_relative_error,
            "min_latent_std": self.min_latent_std,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RepresentationEvaluationConfig:
        allowed = {
            "rollout_horizon",
            "max_test_reconstruction_mse",
            "min_alignment_r2",
            "max_frequency_relative_error",
            "min_latent_std",
        }
        _reject_unknown(data, allowed, "representation evaluation config")
        defaults = cls()
        return cls(
            rollout_horizon=int(data.get("rollout_horizon", defaults.rollout_horizon)),
            max_test_reconstruction_mse=float(
                data.get(
                    "max_test_reconstruction_mse",
                    defaults.max_test_reconstruction_mse,
                )
            ),
            min_alignment_r2=float(data.get("min_alignment_r2", defaults.min_alignment_r2)),
            max_frequency_relative_error=float(
                data.get(
                    "max_frequency_relative_error",
                    defaults.max_frequency_relative_error,
                )
            ),
            min_latent_std=float(data.get("min_latent_std", defaults.min_latent_std)),
        )


@dataclass(frozen=True, slots=True)
class AdvectionDiffusion2DConfig:
    """Analytic periodic 2-D advection-diffusion data settings for V0.5."""

    num_trajectories: int = 12
    num_steps: int = 24
    nx: int = 16
    ny: int = 16
    length_x: float = 6.283185307179586
    length_y: float = 6.283185307179586
    cx: float = 0.7
    cy: float = 0.25
    nu: float = 0.02
    mode_x: int = 1
    mode_y: int = 2
    base_dt: float = 0.05
    variable_dt: bool = True
    dt_jitter: float = 0.15
    amplitude_min: float = 0.7
    amplitude_max: float = 1.3
    mean_min: float = -0.25
    mean_max: float = 0.25

    def __post_init__(self) -> None:
        if self.num_trajectories < 3 or self.num_steps < 2:
            raise ValueError("V0.5 data requires at least 3 trajectories and 2 steps")
        if self.nx < 8 or self.ny < 8:
            raise ValueError("V0.5 periodic grids require nx, ny >= 8")
        if self.length_x <= 0 or self.length_y <= 0 or self.base_dt <= 0:
            raise ValueError("V0.5 lengths and base_dt must be positive")
        if self.nu < 0:
            raise ValueError("V0.5 diffusivity must be non-negative")
        if self.mode_x < 0 or self.mode_y < 0 or self.mode_x + self.mode_y == 0:
            raise ValueError("V0.5 Fourier mode must be non-negative and nonzero")
        if 2 * self.mode_x >= self.nx or 2 * self.mode_y >= self.ny:
            raise ValueError("V0.5 Fourier mode must be below each Nyquist limit")
        if not 0 <= self.dt_jitter < 1:
            raise ValueError("V0.5 dt_jitter must lie in [0,1)")
        if not self.variable_dt and self.dt_jitter != 0:
            raise ValueError("constant-dt V0.5 data requires dt_jitter=0")
        if self.amplitude_min <= 0 or self.amplitude_min > self.amplitude_max:
            raise ValueError("V0.5 amplitude range is invalid")
        if self.mean_min > self.mean_max:
            raise ValueError("V0.5 mean range is invalid")

    def to_dict(self) -> dict[str, Any]:
        integer_fields = {"num_trajectories", "num_steps", "nx", "ny", "mode_x", "mode_y"}
        return {
            name: (
                bool(value)
                if name == "variable_dt"
                else int(value)
                if name in integer_fields
                else float(value)
            )
            for name in self.__dataclass_fields__
            for value in (getattr(self, name),)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AdvectionDiffusion2DConfig:
        defaults = cls()
        allowed = set(defaults.__dataclass_fields__)
        _reject_unknown(data, allowed, "V0.5 advection-diffusion config")
        values = {name: data.get(name, getattr(defaults, name)) for name in allowed}
        for name in {"num_trajectories", "num_steps", "nx", "ny", "mode_x", "mode_y"}:
            values[name] = int(values[name])
        values["variable_dt"] = bool(values["variable_dt"])
        for name in allowed - {
            "num_trajectories",
            "num_steps",
            "nx",
            "ny",
            "mode_x",
            "mode_y",
            "variable_dt",
        }:
            values[name] = float(values[name])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class FieldAutoencoderConfig:
    """Small circular-CNN field autoencoder used by V0.5."""

    input_channels: int = 1
    latent_dim: int = 4
    width: int = 8
    decoder_hidden_dim: int = 32

    def __post_init__(self) -> None:
        if min(self.input_channels, self.latent_dim, self.width, self.decoder_hidden_dim) < 1:
            raise ValueError("V0.5 field autoencoder dimensions must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FieldAutoencoderConfig:
        defaults = cls()
        allowed = set(defaults.__dataclass_fields__)
        _reject_unknown(data, allowed, "V0.5 field autoencoder config")
        return cls(**{name: int(data.get(name, getattr(defaults, name))) for name in allowed})


@dataclass(frozen=True, slots=True)
class FieldLossConfig:
    """Independent V0.5 representation and raw-physics loss weights."""

    lambda_k: float = 1.0
    lambda_generator: float = 1.0
    lambda_multi: float = 1.0
    lambda_rec: float = 1.0
    lambda_forecast: float = 1.0
    lambda_var: float = 0.1
    lambda_stability: float = 0.1
    lambda_physics: float = 0.05
    lambda_mass: float = 0.1
    lambda_operator: float = 0.05
    min_std: float = 0.1

    def __post_init__(self) -> None:
        weights = tuple(
            getattr(self, name) for name in self.__dataclass_fields__ if name.startswith("lambda_")
        )
        if any(value < 0 for value in weights) or sum(weights) <= 0 or self.min_std <= 0:
            raise ValueError("V0.5 loss weights must be non-negative/nonzero and min_std positive")

    def to_dict(self) -> dict[str, Any]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FieldLossConfig:
        defaults = cls()
        allowed = set(defaults.__dataclass_fields__)
        _reject_unknown(data, allowed, "V0.5 field loss config")
        return cls(**{name: float(data.get(name, getattr(defaults, name))) for name in allowed})


@dataclass(frozen=True, slots=True)
class V05TrainingConfig:
    """CPU/GPU-portable V0.5 optimizer and precision controls."""

    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 0.002
    generator_lr_multiplier: float = 5.0
    weight_decay: float = 0.0
    init_scale: float = 0.03
    physics_warmup_epochs: int = 5
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.7
    diagnostic_interval: int = 5
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if (
            self.epochs < 1
            or self.batch_size < 1
            or self.learning_rate <= 0
            or self.generator_lr_multiplier <= 0
        ):
            raise ValueError("V0.5 epochs/batch_size/lr must be positive")
        if self.weight_decay < 0 or self.init_scale < 0:
            raise ValueError("V0.5 weight_decay/init_scale must be non-negative")
        if not 0 <= self.physics_warmup_epochs <= self.epochs:
            raise ValueError("V0.5 physics warmup must lie in [0, epochs]")
        if self.scheduler_step_size < 1 or not 0 < self.scheduler_gamma <= 1:
            raise ValueError("V0.5 scheduler settings are invalid")
        if self.diagnostic_interval < 1:
            raise ValueError("V0.5 diagnostic_interval must be positive")
        if self.precision not in {"fp32", "amp_fp16", "amp_bf16"}:
            raise ValueError("V0.5 precision must be fp32, amp_fp16, or amp_bf16")

    def to_dict(self) -> dict[str, Any]:
        integer_fields = {
            "epochs",
            "batch_size",
            "physics_warmup_epochs",
            "scheduler_step_size",
            "diagnostic_interval",
        }
        return {
            name: (
                str(value)
                if name == "precision"
                else int(value)
                if name in integer_fields
                else float(value)
            )
            for name in self.__dataclass_fields__
            for value in (getattr(self, name),)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V05TrainingConfig:
        defaults = cls()
        allowed = set(defaults.__dataclass_fields__)
        _reject_unknown(data, allowed, "V0.5 training config")
        return cls(
            epochs=int(data.get("epochs", defaults.epochs)),
            batch_size=int(data.get("batch_size", defaults.batch_size)),
            learning_rate=float(data.get("learning_rate", defaults.learning_rate)),
            generator_lr_multiplier=float(
                data.get("generator_lr_multiplier", defaults.generator_lr_multiplier)
            ),
            weight_decay=float(data.get("weight_decay", defaults.weight_decay)),
            init_scale=float(data.get("init_scale", defaults.init_scale)),
            physics_warmup_epochs=int(
                data.get("physics_warmup_epochs", defaults.physics_warmup_epochs)
            ),
            scheduler_step_size=int(data.get("scheduler_step_size", defaults.scheduler_step_size)),
            scheduler_gamma=float(data.get("scheduler_gamma", defaults.scheduler_gamma)),
            diagnostic_interval=int(data.get("diagnostic_interval", defaults.diagnostic_interval)),
            precision=str(data.get("precision", defaults.precision)),
        )


@dataclass(frozen=True, slots=True)
class V05EvaluationConfig:
    """Fixed held-out field-rollout horizons and scientific thresholds."""

    short_horizon: int = 1
    medium_horizon: int = 4
    long_horizon: int = 12
    max_frequency_relative_error: float = 0.05
    max_decay_relative_error: float = 0.2
    max_spectral_abscissa: float = 1.0e-3
    max_relative_mass_drift: float = 0.01
    max_operator_mse: float = 1.0e-4
    max_ablation_skill_degradation: float = 0.05
    max_ablation_constraint_degradation: float = 0.10

    def __post_init__(self) -> None:
        if not 1 <= self.short_horizon <= self.medium_horizon <= self.long_horizon:
            raise ValueError("V0.5 evaluation horizons must be positive and ordered")
        if (
            self.max_frequency_relative_error <= 0
            or self.max_decay_relative_error <= 0
            or self.max_spectral_abscissa < 0
            or self.max_relative_mass_drift <= 0
            or self.max_operator_mse <= 0
            or self.max_ablation_skill_degradation < 0
            or self.max_ablation_constraint_degradation < 0
        ):
            raise ValueError(
                "V0.5 error tolerances must be positive; stability and ablation margins "
                "must be non-negative"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (
                float(getattr(self, name))
                if name
                in {
                    "max_frequency_relative_error",
                    "max_decay_relative_error",
                    "max_spectral_abscissa",
                    "max_relative_mass_drift",
                    "max_operator_mse",
                    "max_ablation_skill_degradation",
                    "max_ablation_constraint_degradation",
                }
                else int(getattr(self, name))
            )
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V05EvaluationConfig:
        defaults = cls()
        allowed = set(defaults.__dataclass_fields__)
        _reject_unknown(data, allowed, "V0.5 evaluation config")
        return cls(
            short_horizon=int(data.get("short_horizon", defaults.short_horizon)),
            medium_horizon=int(data.get("medium_horizon", defaults.medium_horizon)),
            long_horizon=int(data.get("long_horizon", defaults.long_horizon)),
            max_frequency_relative_error=float(
                data.get("max_frequency_relative_error", defaults.max_frequency_relative_error)
            ),
            max_decay_relative_error=float(
                data.get("max_decay_relative_error", defaults.max_decay_relative_error)
            ),
            max_spectral_abscissa=float(
                data.get("max_spectral_abscissa", defaults.max_spectral_abscissa)
            ),
            max_relative_mass_drift=float(
                data.get("max_relative_mass_drift", defaults.max_relative_mass_drift)
            ),
            max_operator_mse=float(data.get("max_operator_mse", defaults.max_operator_mse)),
            max_ablation_skill_degradation=float(
                data.get(
                    "max_ablation_skill_degradation",
                    defaults.max_ablation_skill_degradation,
                )
            ),
            max_ablation_constraint_degradation=float(
                data.get(
                    "max_ablation_constraint_degradation",
                    defaults.max_ablation_constraint_degradation,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class JEPALossConfig:
    """V0.6 JEPA additions to the complete, unchanged V0.5 objective."""

    lambda_one_step: float = 1.0
    lambda_multi_step: float = 1.0

    def __post_init__(self) -> None:
        if self.lambda_one_step < 0 or self.lambda_multi_step < 0:
            raise ValueError("V0.6 JEPA weights must be non-negative")

    @property
    def enabled(self) -> bool:
        return self.lambda_one_step > 0 or self.lambda_multi_step > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lambda_one_step": float(self.lambda_one_step),
            "lambda_multi_step": float(self.lambda_multi_step),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JEPALossConfig:
        defaults = cls()
        allowed = {"lambda_one_step", "lambda_multi_step"}
        _reject_unknown(data, allowed, "V0.6 JEPA loss config")
        return cls(
            lambda_one_step=float(data.get("lambda_one_step", defaults.lambda_one_step)),
            lambda_multi_step=float(data.get("lambda_multi_step", defaults.lambda_multi_step)),
        )


@dataclass(frozen=True, slots=True)
class EMAConfig:
    """Optimizer-step-indexed EMA target schedule for V0.6."""

    start_tau: float = 0.996
    end_tau: float = 1.0
    schedule: str = "linear"

    def __post_init__(self) -> None:
        if not 0 <= self.start_tau <= self.end_tau <= 1:
            raise ValueError("EMA tau must satisfy 0 <= start_tau <= end_tau <= 1")
        if self.schedule not in {"constant", "linear"}:
            raise ValueError("EMA schedule must be 'constant' or 'linear'")
        if self.schedule == "constant" and self.start_tau != self.end_tau:
            raise ValueError("constant EMA schedule requires start_tau == end_tau")

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_tau": float(self.start_tau),
            "end_tau": float(self.end_tau),
            "schedule": self.schedule,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EMAConfig:
        defaults = cls()
        allowed = {"start_tau", "end_tau", "schedule"}
        _reject_unknown(data, allowed, "V0.6 EMA config")
        return cls(
            start_tau=float(data.get("start_tau", defaults.start_tau)),
            end_tau=float(data.get("end_tau", defaults.end_tau)),
            schedule=str(data.get("schedule", defaults.schedule)),
        )


@dataclass(frozen=True, slots=True)
class V06EvaluationConfig:
    """Matched-control scientific gates; final acceptance requires multi-seed GPU evidence."""

    max_long_rollout_degradation: float = 0.05
    max_physics_degradation: float = 0.10
    min_latent_std: float = 0.02
    seeds: tuple[int, ...] = (47, 53, 59)

    def __post_init__(self) -> None:
        if self.max_long_rollout_degradation < 0 or self.max_physics_degradation < 0:
            raise ValueError("V0.6 degradation margins must be non-negative")
        if self.min_latent_std <= 0:
            raise ValueError("V0.6 minimum latent std threshold must be positive")
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("V0.6 scientific comparison requires at least three unique seeds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_long_rollout_degradation": float(self.max_long_rollout_degradation),
            "max_physics_degradation": float(self.max_physics_degradation),
            "min_latent_std": float(self.min_latent_std),
            "seeds": list(self.seeds),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V06EvaluationConfig:
        defaults = cls()
        allowed = {
            "max_long_rollout_degradation",
            "max_physics_degradation",
            "min_latent_std",
            "seeds",
        }
        _reject_unknown(data, allowed, "V0.6 evaluation config")
        return cls(
            max_long_rollout_degradation=float(
                data.get("max_long_rollout_degradation", defaults.max_long_rollout_degradation)
            ),
            max_physics_degradation=float(
                data.get("max_physics_degradation", defaults.max_physics_degradation)
            ),
            min_latent_std=float(data.get("min_latent_std", defaults.min_latent_std)),
            seeds=tuple(int(seed) for seed in data.get("seeds", defaults.seeds)),
        )


@dataclass(frozen=True, slots=True)
class ResidualClosureConfig:
    """V0.7 residual-cache and minimal closure architecture contract."""

    history: int = 4
    hidden_dim: int = 64
    depth: int = 2
    include_parameters: bool = True
    cache_dtype: str = "float32"
    max_acf_lag: int = 12
    variants: tuple[str, ...] = (
        "zero",
        "linear",
        "instantaneous",
        "history",
        "shuffled_history",
    )

    def __post_init__(self) -> None:
        allowed = {"zero", "linear", "instantaneous", "history", "shuffled_history"}
        if self.history < 1 or self.hidden_dim < 1 or self.depth < 1:
            raise ValueError("V0.7 closure history>=1 and positive hidden_dim/depth are required")
        if self.cache_dtype not in {"float32", "float64"}:
            raise ValueError("V0.7 cache_dtype must be float32 or float64")
        if self.max_acf_lag < 1:
            raise ValueError("V0.7 max_acf_lag must be positive")
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise ValueError("V0.7 closure variants must be unique and non-empty")
        unknown = set(self.variants) - allowed
        if unknown:
            raise ValueError(f"unknown V0.7 closure variant(s): {sorted(unknown)!r}")
        if "zero" not in self.variants or "history" not in self.variants:
            raise ValueError("V0.7 requires zero and history closure variants")

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "include_parameters": self.include_parameters,
            "cache_dtype": self.cache_dtype,
            "max_acf_lag": self.max_acf_lag,
            "variants": list(self.variants),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResidualClosureConfig:
        defaults = cls()
        allowed = {
            "history",
            "hidden_dim",
            "depth",
            "include_parameters",
            "cache_dtype",
            "max_acf_lag",
            "variants",
        }
        _reject_unknown(data, allowed, "V0.7 residual closure config")
        return cls(
            history=int(data.get("history", defaults.history)),
            hidden_dim=int(data.get("hidden_dim", defaults.hidden_dim)),
            depth=int(data.get("depth", defaults.depth)),
            include_parameters=bool(data.get("include_parameters", defaults.include_parameters)),
            cache_dtype=str(data.get("cache_dtype", defaults.cache_dtype)),
            max_acf_lag=int(data.get("max_acf_lag", defaults.max_acf_lag)),
            variants=tuple(str(item) for item in data.get("variants", defaults.variants)),
        )


@dataclass(frozen=True, slots=True)
class ResidualTrainingConfig:
    """Optimizer settings for closure-only V0.7 training."""

    epochs: int = 60
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    patience: int = 12
    precision: str = "fp32"
    initialization_seed: int = 101

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("V0.7 closure epochs/batch_size/lr must be positive")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0 or self.patience < 1:
            raise ValueError("invalid V0.7 closure regularization or patience")
        if self.precision not in {"fp32", "amp_fp16", "amp_bf16"}:
            raise ValueError("V0.7 precision must be fp32, amp_fp16, or amp_bf16")
        if self.initialization_seed < 0:
            raise ValueError("V0.7 closure initialization_seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "patience": self.patience,
            "precision": self.precision,
            "initialization_seed": self.initialization_seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResidualTrainingConfig:
        defaults = cls()
        allowed = {
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "patience",
            "precision",
            "initialization_seed",
        }
        _reject_unknown(data, allowed, "V0.7 residual training config")
        return cls(
            epochs=int(data.get("epochs", defaults.epochs)),
            batch_size=int(data.get("batch_size", defaults.batch_size)),
            learning_rate=float(data.get("learning_rate", defaults.learning_rate)),
            weight_decay=float(data.get("weight_decay", defaults.weight_decay)),
            gradient_clip_norm=float(data.get("gradient_clip_norm", defaults.gradient_clip_norm)),
            patience=int(data.get("patience", defaults.patience)),
            precision=str(data.get("precision", defaults.precision)),
            initialization_seed=int(data.get("initialization_seed", defaults.initialization_seed)),
        )


@dataclass(frozen=True, slots=True)
class MemorySweepConfig:
    """Explicit operational thresholds for V0.7 memory characterization."""

    history_lengths: tuple[int, ...] = (1, 2, 4, 8, 16)
    effective_gain_fraction: float = 0.95
    material_relative_gain: float = 0.02
    plateau_relative_gain: float = 0.01
    parameter_match_tolerance: float = 0.05
    seed_consistency_fraction: float = 2.0 / 3.0
    initialization_seeds: tuple[int, ...] = (101, 211, 307)
    strong_r2: float = 0.75
    moderate_r2: float = 0.40
    weak_r2: float = 0.05

    def __post_init__(self) -> None:
        if len(self.history_lengths) < 4 or self.history_lengths[0] != 1:
            raise ValueError("V0.7 memory sweep requires H=1 and at least four history levels")
        if tuple(sorted(set(self.history_lengths))) != self.history_lengths:
            raise ValueError("V0.7 history lengths must be unique and increasing")
        if not 0 < self.effective_gain_fraction <= 1:
            raise ValueError("effective_gain_fraction must lie in (0,1]")
        for name, value in (
            ("material_relative_gain", self.material_relative_gain),
            ("plateau_relative_gain", self.plateau_relative_gain),
            ("parameter_match_tolerance", self.parameter_match_tolerance),
        ):
            if not 0 <= value < 1:
                raise ValueError(f"{name} must lie in [0,1)")
        if not 0.5 <= self.seed_consistency_fraction <= 1:
            raise ValueError("seed_consistency_fraction must lie in [0.5,1]")
        if not self.initialization_seeds or len(set(self.initialization_seeds)) != len(
            self.initialization_seeds
        ):
            raise ValueError("V0.7 closure initialization seeds must be unique and non-empty")
        if any(seed < 0 for seed in self.initialization_seeds):
            raise ValueError("V0.7 closure initialization seeds must be non-negative")
        if not 1 >= self.strong_r2 > self.moderate_r2 > self.weak_r2 >= 0:
            raise ValueError("learnability R2 thresholds must strictly decrease in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_lengths": list(self.history_lengths),
            "effective_gain_fraction": self.effective_gain_fraction,
            "material_relative_gain": self.material_relative_gain,
            "plateau_relative_gain": self.plateau_relative_gain,
            "parameter_match_tolerance": self.parameter_match_tolerance,
            "seed_consistency_fraction": self.seed_consistency_fraction,
            "initialization_seeds": list(self.initialization_seeds),
            "strong_r2": self.strong_r2,
            "moderate_r2": self.moderate_r2,
            "weak_r2": self.weak_r2,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemorySweepConfig:
        defaults = cls()
        allowed = {
            "history_lengths",
            "effective_gain_fraction",
            "material_relative_gain",
            "plateau_relative_gain",
            "parameter_match_tolerance",
            "seed_consistency_fraction",
            "initialization_seeds",
            "strong_r2",
            "moderate_r2",
            "weak_r2",
        }
        _reject_unknown(data, allowed, "V0.7 memory sweep config")
        return cls(
            history_lengths=tuple(
                int(value) for value in data.get("history_lengths", defaults.history_lengths)
            ),
            effective_gain_fraction=float(
                data.get("effective_gain_fraction", defaults.effective_gain_fraction)
            ),
            material_relative_gain=float(
                data.get("material_relative_gain", defaults.material_relative_gain)
            ),
            plateau_relative_gain=float(
                data.get("plateau_relative_gain", defaults.plateau_relative_gain)
            ),
            parameter_match_tolerance=float(
                data.get("parameter_match_tolerance", defaults.parameter_match_tolerance)
            ),
            seed_consistency_fraction=float(
                data.get("seed_consistency_fraction", defaults.seed_consistency_fraction)
            ),
            initialization_seeds=tuple(
                int(value)
                for value in data.get("initialization_seeds", defaults.initialization_seeds)
            ),
            strong_r2=float(data.get("strong_r2", defaults.strong_r2)),
            moderate_r2=float(data.get("moderate_r2", defaults.moderate_r2)),
            weak_r2=float(data.get("weak_r2", defaults.weak_r2)),
        )


@dataclass(frozen=True, slots=True)
class V07EvaluationConfig:
    """Problem-agnostic thresholds for residual structure assessment."""

    rollout_horizons: tuple[int, ...] = (8, 16, 32)
    min_residual_rms: float = 1e-6
    min_residual_significance: float = 0.01
    min_history_r2_gain: float = 0.02
    max_closure_burden: float = 0.25
    max_physics_degradation: float = 0.10
    formal_record_count: int = 144
    seeds: tuple[int, ...] = (47, 53, 59)

    def __post_init__(self) -> None:
        if not self.rollout_horizons or any(value < 1 for value in self.rollout_horizons):
            raise ValueError("V0.7 rollout horizons must be positive")
        if tuple(sorted(set(self.rollout_horizons))) != self.rollout_horizons:
            raise ValueError("V0.7 rollout horizons must be unique and increasing")
        if (
            self.min_residual_rms < 0
            or not 0 <= self.min_residual_significance < 1
            or self.min_history_r2_gain < 0
        ):
            raise ValueError("V0.7 evidence thresholds must be non-negative")
        if self.max_closure_burden <= 0 or self.max_physics_degradation < 0:
            raise ValueError("invalid V0.7 closure/physics threshold")
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("V0.7 scientific comparison requires at least three unique seeds")
        if self.formal_record_count < 1:
            raise ValueError("V0.7 formal_record_count must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_horizons": list(self.rollout_horizons),
            "min_residual_rms": self.min_residual_rms,
            "min_residual_significance": self.min_residual_significance,
            "min_history_r2_gain": self.min_history_r2_gain,
            "max_closure_burden": self.max_closure_burden,
            "max_physics_degradation": self.max_physics_degradation,
            "formal_record_count": self.formal_record_count,
            "seeds": list(self.seeds),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V07EvaluationConfig:
        defaults = cls()
        allowed = {
            "rollout_horizons",
            "min_residual_rms",
            "min_residual_significance",
            "min_history_r2_gain",
            "max_closure_burden",
            "max_physics_degradation",
            "formal_record_count",
            "seeds",
        }
        _reject_unknown(data, allowed, "V0.7 evaluation config")
        return cls(
            rollout_horizons=tuple(
                int(value) for value in data.get("rollout_horizons", defaults.rollout_horizons)
            ),
            min_residual_rms=float(data.get("min_residual_rms", defaults.min_residual_rms)),
            min_residual_significance=float(
                data.get("min_residual_significance", defaults.min_residual_significance)
            ),
            min_history_r2_gain=float(
                data.get("min_history_r2_gain", defaults.min_history_r2_gain)
            ),
            max_closure_burden=float(data.get("max_closure_burden", defaults.max_closure_burden)),
            max_physics_degradation=float(
                data.get("max_physics_degradation", defaults.max_physics_degradation)
            ),
            formal_record_count=int(data.get("formal_record_count", defaults.formal_record_count)),
            seeds=tuple(int(seed) for seed in data.get("seeds", defaults.seeds)),
        )


@dataclass(frozen=True, slots=True)
class CylinderWake2DConfig:
    """Low-Mach D2Q9 cylinder-wake data-generation contract shared by V0.8/V0.9."""

    num_trajectories: int = 12
    num_steps: int = 160
    nx: int = 256
    ny: int = 128
    x_min: float = -8.0
    x_max: float = 16.0
    y_min: float = -6.0
    y_max: float = 6.0
    cylinder_x: float = 0.0
    cylinder_y: float = 0.0
    cylinder_diameter: float = 1.0
    reynolds_number: float = 100.0
    u_infinity: float = 1.0
    lattice_inflow_velocity: float = 0.08
    solver_steps_per_snapshot: int = 20
    perturbation_amplitude: float = 0.015
    time_varying_boundary: bool = False
    dataset_path: str = ""

    def __post_init__(self) -> None:
        if self.num_trajectories < 3 or self.num_steps < 2:
            raise ValueError("V0.8 cylinder data requires at least 3 trajectories and 2 steps")
        if self.nx < 32 or self.ny < 16:
            raise ValueError("V0.8 cylinder grid requires nx>=32 and ny>=16")
        if not self.x_min < self.cylinder_x < self.x_max:
            raise ValueError("cylinder center must lie inside the x domain")
        if not self.y_min < self.cylinder_y < self.y_max:
            raise ValueError("cylinder center must lie inside the y domain")
        if (
            min(
                self.x_max - self.x_min,
                self.y_max - self.y_min,
                self.cylinder_diameter,
                self.reynolds_number,
                self.u_infinity,
                self.lattice_inflow_velocity,
            )
            <= 0
        ):
            raise ValueError("V0.8 cylinder physical scales must be positive")
        if self.lattice_inflow_velocity >= 0.12:
            raise ValueError("D2Q9 low-Mach contract requires lattice inflow velocity < 0.12")
        if self.solver_steps_per_snapshot < 1 or self.perturbation_amplitude < 0:
            raise ValueError("invalid cylinder sampling or perturbation setting")
        radius = 0.5 * self.cylinder_diameter
        if (
            self.cylinder_x - radius <= self.x_min
            or self.cylinder_x + radius >= self.x_max
            or self.cylinder_y - radius <= self.y_min
            or self.cylinder_y + radius >= self.y_max
        ):
            raise ValueError("cylinder must be strictly separated from the domain boundary")
        if self.lattice_relaxation_time <= 0.505:
            raise ValueError("D2Q9 relaxation time is too close to the stability limit 0.5")

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.ny

    @property
    def cylinder_diameter_cells(self) -> float:
        return self.cylinder_diameter / self.dx

    @property
    def lattice_viscosity(self) -> float:
        return self.lattice_inflow_velocity * self.cylinder_diameter_cells / self.reynolds_number

    @property
    def lattice_relaxation_time(self) -> float:
        return 0.5 + 3.0 * self.lattice_viscosity

    @property
    def snapshot_dt(self) -> float:
        return (
            self.solver_steps_per_snapshot
            * self.lattice_inflow_velocity
            / self.cylinder_diameter_cells
        )

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CylinderWake2DConfig:
        defaults = cls()
        allowed = set(defaults.__dataclass_fields__)
        _reject_unknown(data, allowed, "V0.8 cylinder-wake config")
        values = {name: data.get(name, getattr(defaults, name)) for name in allowed}
        for name in {"num_trajectories", "num_steps", "nx", "ny", "solver_steps_per_snapshot"}:
            values[name] = int(values[name])
        values["time_varying_boundary"] = bool(values["time_varying_boundary"])
        values["dataset_path"] = str(values["dataset_path"])
        for name in allowed - {
            "num_trajectories",
            "num_steps",
            "nx",
            "ny",
            "solver_steps_per_snapshot",
            "time_varying_boundary",
            "dataset_path",
        }:
            values[name] = float(values[name])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class V08RoutingConfig:
    """Evidence-owned context-family selection; benchmark names never select a route."""

    mode: str = "auto"
    v0_7_result: str = ""

    def __post_init__(self) -> None:
        if self.mode != "auto":
            raise ValueError("V0.8 formal routing mode must be auto")

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "v0_7_result": self.v0_7_result}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V08RoutingConfig:
        _reject_unknown(data, {"mode", "v0_7_result"}, "V0.8 routing config")
        return cls(mode=str(data.get("mode", "auto")), v0_7_result=str(data.get("v0_7_result", "")))


@dataclass(frozen=True, slots=True)
class V08ContextConfig:
    """Compact R2/R3 context model shared-output contract."""

    family: str = "auto"
    context_dim: int = 8
    history_length: int = 8
    width: int = 64
    layers: int = 2
    heads: int = 4
    ffn_multiplier: int = 2
    dropout: float = 0.0
    include_parameters: bool = True

    def __post_init__(self) -> None:
        if self.family not in {
            "auto",
            "instantaneous",
            "instantaneous_matched",
            "attention",
            "history_mlp",
        }:
            raise ValueError("unknown V0.8 context family")
        if self.context_dim not in {4, 8, 16} or self.history_length < 1:
            raise ValueError("V0.8 requires d_c in {4,8,16} and positive history")
        if not 1 <= self.layers <= 4 or not 1 <= self.width <= 128:
            raise ValueError("V0.8 context depth/width exceeds the compact-model boundary")
        if self.heads < 1 or self.width % self.heads != 0 or self.ffn_multiplier < 1:
            raise ValueError("invalid V0.8 Attention heads or FFN multiplier")
        if not 0 <= self.dropout < 1:
            raise ValueError("V0.8 context dropout must lie in [0,1)")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V08ContextConfig:
        defaults = cls()
        allowed = set(defaults.__dataclass_fields__)
        _reject_unknown(data, allowed, "V0.8 context config")
        return cls(
            family=str(data.get("family", defaults.family)),
            context_dim=int(data.get("context_dim", defaults.context_dim)),
            history_length=int(data.get("history_length", defaults.history_length)),
            width=int(data.get("width", defaults.width)),
            layers=int(data.get("layers", defaults.layers)),
            heads=int(data.get("heads", defaults.heads)),
            ffn_multiplier=int(data.get("ffn_multiplier", defaults.ffn_multiplier)),
            dropout=float(data.get("dropout", defaults.dropout)),
            include_parameters=bool(data.get("include_parameters", defaults.include_parameters)),
        )


@dataclass(frozen=True, slots=True)
class V08TrainingConfig:
    """Context-only optimization and exact-resume settings."""

    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    lambda_adequacy: float = 0.2
    gradient_clip_norm: float = 1.0
    patience: int = 16
    precision: str = "amp_bf16"
    context_initialization_seed: int = 401

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("V0.8 epochs, batch size, and learning rate must be positive")
        if self.weight_decay < 0 or self.lambda_adequacy < 0:
            raise ValueError("V0.8 loss weights must be non-negative")
        if self.gradient_clip_norm <= 0 or self.patience < 1:
            raise ValueError("invalid V0.8 clipping or patience")
        if self.precision not in {"fp32", "amp_fp16", "amp_bf16"}:
            raise ValueError("invalid V0.8 precision")
        if self.context_initialization_seed < 0:
            raise ValueError("V0.8 context seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V08TrainingConfig:
        defaults = cls()
        allowed = set(defaults.__dataclass_fields__)
        _reject_unknown(data, allowed, "V0.8 training config")
        values = {name: data.get(name, getattr(defaults, name)) for name in allowed}
        for name in {"epochs", "batch_size", "patience", "context_initialization_seed"}:
            values[name] = int(values[name])
        for name in {
            "learning_rate",
            "weight_decay",
            "lambda_adequacy",
            "gradient_clip_norm",
        }:
            values[name] = float(values[name])
        values["precision"] = str(values["precision"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class V08EvaluationConfig:
    """Locked scientific acceptance thresholds for dynamic context."""

    rollout_horizons: tuple[int, ...] = (8, 16, 32, 80)
    material_relative_gain: float = 0.02
    min_context_effective_rank: float = 2.0
    max_closure_burden: float = 0.25
    max_physics_degradation: float = 0.10
    max_divergence_mse: float = 0.02
    max_boundary_mse: float = 0.05
    seed_consistency_fraction: float = 2.0 / 3.0
    context_initialization_seeds: tuple[int, ...] = (401, 503, 607)

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.rollout_horizons))) != self.rollout_horizons:
            raise ValueError("V0.8 rollout horizons must be unique and increasing")
        if not self.rollout_horizons or self.rollout_horizons[0] < 1:
            raise ValueError("V0.8 rollout horizons must be positive")
        if not 0 <= self.material_relative_gain < 1:
            raise ValueError("V0.8 material gain must lie in [0,1)")
        if self.min_context_effective_rank <= 1 or self.max_closure_burden <= 0:
            raise ValueError("invalid V0.8 context-rank or burden threshold")
        if (
            self.max_physics_degradation < 0
            or self.max_divergence_mse <= 0
            or self.max_boundary_mse <= 0
            or not 0.5 <= self.seed_consistency_fraction <= 1
        ):
            raise ValueError("invalid V0.8 physics/seed threshold")
        if len(self.context_initialization_seeds) < 3 or len(
            set(self.context_initialization_seeds)
        ) != len(self.context_initialization_seeds):
            raise ValueError("V0.8 formal evaluation requires three unique context seeds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_horizons": list(self.rollout_horizons),
            "material_relative_gain": self.material_relative_gain,
            "min_context_effective_rank": self.min_context_effective_rank,
            "max_closure_burden": self.max_closure_burden,
            "max_physics_degradation": self.max_physics_degradation,
            "max_divergence_mse": self.max_divergence_mse,
            "max_boundary_mse": self.max_boundary_mse,
            "seed_consistency_fraction": self.seed_consistency_fraction,
            "context_initialization_seeds": list(self.context_initialization_seeds),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V08EvaluationConfig:
        defaults = cls()
        allowed = set(defaults.to_dict())
        _reject_unknown(data, allowed, "V0.8 evaluation config")
        return cls(
            rollout_horizons=tuple(
                int(value) for value in data.get("rollout_horizons", defaults.rollout_horizons)
            ),
            material_relative_gain=float(
                data.get("material_relative_gain", defaults.material_relative_gain)
            ),
            min_context_effective_rank=float(
                data.get("min_context_effective_rank", defaults.min_context_effective_rank)
            ),
            max_closure_burden=float(data.get("max_closure_burden", defaults.max_closure_burden)),
            max_physics_degradation=float(
                data.get("max_physics_degradation", defaults.max_physics_degradation)
            ),
            max_divergence_mse=float(data.get("max_divergence_mse", defaults.max_divergence_mse)),
            max_boundary_mse=float(data.get("max_boundary_mse", defaults.max_boundary_mse)),
            seed_consistency_fraction=float(
                data.get("seed_consistency_fraction", defaults.seed_consistency_fraction)
            ),
            context_initialization_seeds=tuple(
                int(value)
                for value in data.get(
                    "context_initialization_seeds", defaults.context_initialization_seeds
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class V09ConditionConfig:
    """Controlled, in-distribution operating-condition schedules for V0.9."""

    schedule_types: tuple[str, ...] = ("smooth", "abrupt")
    reynolds_low: float = 80.0
    reynolds_high: float = 120.0
    transition_start_fraction: float = 0.35
    smooth_duration_fraction: float = 0.25
    known_condition_features: tuple[str, ...] = ("reynolds_number", "u_infinity")

    def __post_init__(self) -> None:
        if tuple(dict.fromkeys(self.schedule_types)) != self.schedule_types:
            raise ValueError("V0.9 condition schedule types must be unique")
        allowed = {
            "smooth",
            "abrupt",
            "smooth_up_slow",
            "smooth_up_medium",
            "smooth_up_fast",
            "abrupt_up",
            "smooth_down_slow",
            "smooth_down_medium",
            "smooth_down_fast",
            "abrupt_down",
            "cyclic_slow_long",
            "cyclic_slow_short",
            "cyclic_fast_long",
            "cyclic_fast_short",
        }
        if not self.schedule_types or set(self.schedule_types) - allowed:
            raise ValueError("V0.9 condition config contains an unsupported schedule")
        if not 0 < self.reynolds_low < self.reynolds_high:
            raise ValueError("V0.9 requires 0 < Re_low < Re_high")
        if not 0 < self.transition_start_fraction < 1:
            raise ValueError("V0.9 transition start fraction must lie in (0,1)")
        if not 0 < self.smooth_duration_fraction < 1:
            raise ValueError("V0.9 smooth duration fraction must lie in (0,1)")
        if self.transition_start_fraction + self.smooth_duration_fraction >= 0.9:
            raise ValueError("V0.9 schedules must retain a post-transition rollout tail")
        if self.known_condition_features != ("reynolds_number", "u_infinity"):
            raise ValueError("V0.9 canonical known-condition features are Re and U_infinity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_types": list(self.schedule_types),
            "reynolds_low": self.reynolds_low,
            "reynolds_high": self.reynolds_high,
            "transition_start_fraction": self.transition_start_fraction,
            "smooth_duration_fraction": self.smooth_duration_fraction,
            "known_condition_features": list(self.known_condition_features),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V09ConditionConfig:
        defaults = cls()
        _reject_unknown(data, set(defaults.to_dict()), "V0.9 condition config")
        return cls(
            schedule_types=tuple(
                str(value) for value in data.get("schedule_types", defaults.schedule_types)
            ),
            reynolds_low=float(data.get("reynolds_low", defaults.reynolds_low)),
            reynolds_high=float(data.get("reynolds_high", defaults.reynolds_high)),
            transition_start_fraction=float(
                data.get("transition_start_fraction", defaults.transition_start_fraction)
            ),
            smooth_duration_fraction=float(
                data.get("smooth_duration_fraction", defaults.smooth_duration_fraction)
            ),
            known_condition_features=tuple(
                str(value)
                for value in data.get("known_condition_features", defaults.known_condition_features)
            ),
        )


@dataclass(frozen=True, slots=True)
class V09AdaptiveConfig:
    """Restricted context-to-generator interface; no additive residual path."""

    condition_mode: str = "latent_inferred"
    rank: int = 4
    rank_candidates: tuple[int, ...] = (1, 2, 4, 8)
    width: int = 64
    condition_embedding_dim: int = 4
    normalize_factors: bool = True
    zero_output_initialization: bool = True
    bounded_coordinates: bool = False
    eta_max: float = 1.0
    trust_gate: bool = False
    trust_gate_bias: float = -1.3862943611198906

    def __post_init__(self) -> None:
        if self.condition_mode not in {"known", "latent_inferred"}:
            raise ValueError("V0.9 condition_mode must be known or latent_inferred")
        if self.rank < 1 or self.rank not in self.rank_candidates:
            raise ValueError("V0.9 rank must be a registered rank candidate")
        if tuple(sorted(set(self.rank_candidates))) != self.rank_candidates:
            raise ValueError("V0.9 rank candidates must be unique and increasing")
        if self.rank_candidates[-1] > 16:
            raise ValueError("V0.9 primary low-rank update is capped at rank 16")
        if self.width < 4 or self.width > 256 or self.condition_embedding_dim < 1:
            raise ValueError("invalid V0.9 adaptive-head dimensions")
        if not self.zero_output_initialization:
            raise ValueError("V0.9 requires exact zero-update initialization")
        if self.eta_max <= 0:
            raise ValueError("V0.9 eta_max must be positive")
        if not -10.0 <= self.trust_gate_bias <= 10.0:
            raise ValueError("V0.9 trust-gate bias is outside the auditable range")

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_mode": self.condition_mode,
            "rank": self.rank,
            "rank_candidates": list(self.rank_candidates),
            "width": self.width,
            "condition_embedding_dim": self.condition_embedding_dim,
            "normalize_factors": self.normalize_factors,
            "zero_output_initialization": self.zero_output_initialization,
            "bounded_coordinates": self.bounded_coordinates,
            "eta_max": self.eta_max,
            "trust_gate": self.trust_gate,
            "trust_gate_bias": self.trust_gate_bias,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V09AdaptiveConfig:
        defaults = cls()
        _reject_unknown(data, set(defaults.to_dict()), "V0.9 adaptive config")
        return cls(
            condition_mode=str(data.get("condition_mode", defaults.condition_mode)),
            rank=int(data.get("rank", defaults.rank)),
            rank_candidates=tuple(
                int(value) for value in data.get("rank_candidates", defaults.rank_candidates)
            ),
            width=int(data.get("width", defaults.width)),
            condition_embedding_dim=int(
                data.get("condition_embedding_dim", defaults.condition_embedding_dim)
            ),
            normalize_factors=bool(data.get("normalize_factors", defaults.normalize_factors)),
            zero_output_initialization=bool(
                data.get("zero_output_initialization", defaults.zero_output_initialization)
            ),
            bounded_coordinates=bool(data.get("bounded_coordinates", defaults.bounded_coordinates)),
            eta_max=float(data.get("eta_max", defaults.eta_max)),
            trust_gate=bool(data.get("trust_gate", defaults.trust_gate)),
            trust_gate_bias=float(data.get("trust_gate_bias", defaults.trust_gate_bias)),
        )


@dataclass(frozen=True, slots=True)
class V09TrainingConfig:
    """Operator-only optimization with compact formal-GPU output."""

    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-5
    lambda_operator_burden: float = 1.0e-3
    lambda_smooth: float = 1.0e-3
    lambda_stability: float = 1.0e-3
    rollout_horizons: tuple[int, ...] = (1,)
    rollout_start_fractions: tuple[float, ...] = (0.0,)
    rollout_weights: tuple[float, ...] = (0.0,)
    rollout_batch_size: int = 32
    rollout_stride: int = 1
    lambda_rollout: float = 0.0
    lambda_propagator_growth: float = 0.0
    propagator_growth_margin: float = 0.02
    operator_burden_target: float = 0.50
    physics_start_fraction: float = 1.0
    physics_ramp_duration_fraction: float = 0.0
    physics_horizon: int = 1
    physics_batch_size: int = 1
    lambda_physics: float = 0.0
    physics_velocity_weight: float = 1.0
    physics_vorticity_weight: float = 0.2
    physics_divergence_weight: float = 0.2
    physics_boundary_weight: float = 0.1
    physics_lift_weight: float = 0.0
    physics_drag_weight: float = 0.0
    observable_names: tuple[str, ...] = ()
    observable_component_weights: tuple[float, ...] = ()
    observable_horizons: tuple[int, ...] = ()
    observable_horizon_weights: tuple[float, ...] = ()
    observable_horizon_probabilities: tuple[float, ...] = ()
    lambda_observable_noninferiority: float = 0.0
    observable_noninferiority_margin: float = 0.10
    observable_noninferiority_floor: float = 1.0e-3
    phase1_enabled: bool = False
    observable_scale_method: str = "mad"
    observable_scale_epsilon: float = 1.0e-6
    observable_scale_max_samples: int = 262_144
    observable_huber_delta: float = 1.0
    force_correlation_weight: float = 0.1
    force_spectrum_weight: float = 0.05
    force_window_stride: int = 4
    augmented_lagrangian_initial_penalty: float = 1.0
    augmented_lagrangian_penalty_growth: float = 2.0
    augmented_lagrangian_max_penalty: float = 100.0
    augmented_lagrangian_improvement_ratio: float = 0.9
    augmented_lagrangian_update_interval: int = 1
    augmented_lagrangian_dual_step_size: float = 1.0
    augmented_lagrangian_max_multiplier: float = 100.0
    gradient_audit_interval: int = 5
    gradient_conflict_method: str = "none"
    gradient_conflict_start_fraction: float = 0.0
    rank_sweep_epochs: int = 40
    gradient_clip_norm: float = 1.0
    patience: int = 16
    precision: str = "amp_bf16"
    operator_initialization_seed: int = 701

    def __post_init__(self) -> None:
        if (
            min(
                self.epochs,
                self.batch_size,
                self.rollout_batch_size,
                self.rollout_stride,
                self.physics_batch_size,
                self.rank_sweep_epochs,
                self.patience,
            )
            < 1
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "V0.9 epochs, batch size, patience, and learning rate must be positive"
            )
        if (
            min(
                self.weight_decay,
                self.lambda_operator_burden,
                self.lambda_smooth,
                self.lambda_stability,
                self.lambda_rollout,
                self.lambda_propagator_growth,
                self.lambda_physics,
                self.physics_velocity_weight,
                self.physics_vorticity_weight,
                self.physics_divergence_weight,
                self.physics_boundary_weight,
                self.physics_lift_weight,
                self.physics_drag_weight,
                self.lambda_observable_noninferiority,
                self.observable_noninferiority_margin,
                self.observable_noninferiority_floor,
                self.observable_scale_epsilon,
                self.observable_huber_delta,
                self.force_correlation_weight,
                self.force_spectrum_weight,
                self.augmented_lagrangian_initial_penalty,
                self.augmented_lagrangian_penalty_growth,
                self.augmented_lagrangian_max_penalty,
                self.augmented_lagrangian_improvement_ratio,
                self.augmented_lagrangian_dual_step_size,
                self.augmented_lagrangian_max_multiplier,
                self.gradient_conflict_start_fraction,
                *self.observable_component_weights,
                *self.observable_horizon_weights,
                *self.observable_horizon_probabilities,
            )
            < 0
        ):
            raise ValueError("V0.9 regularization weights must be non-negative")
        if tuple(sorted(set(self.rollout_horizons))) != self.rollout_horizons:
            raise ValueError("V0.9 training rollout horizons must be unique and increasing")
        if not self.rollout_horizons or self.rollout_horizons[0] < 1:
            raise ValueError("V0.9 training rollout horizons must be positive")
        if not (
            len(self.rollout_horizons)
            == len(self.rollout_start_fractions)
            == len(self.rollout_weights)
        ):
            raise ValueError("V0.9 rollout curriculum arrays must have equal length")
        if any(not 0.0 <= value <= 1.0 for value in self.rollout_start_fractions):
            raise ValueError("V0.9 rollout start fractions must lie in [0,1]")
        if tuple(sorted(self.rollout_start_fractions)) != self.rollout_start_fractions:
            raise ValueError("V0.9 rollout curriculum must be ordered")
        if self.lambda_rollout > 0 and not any(weight > 0 for weight in self.rollout_weights):
            raise ValueError("enabled V0.9 rollout loss requires a positive horizon weight")
        if not 0.0 <= self.physics_start_fraction <= 1.0:
            raise ValueError("V0.9 physics start fraction must lie in [0,1]")
        if not 0.0 <= self.physics_ramp_duration_fraction <= 1.0:
            raise ValueError("V0.9 physics ramp duration must lie in [0,1]")
        if (
            self.physics_ramp_duration_fraction > 0
            and self.physics_start_fraction + self.physics_ramp_duration_fraction > 1
        ):
            raise ValueError("V0.9 physics ramp must finish within training")
        if self.physics_horizon < 1 or self.physics_horizon > self.rollout_horizons[-1]:
            raise ValueError("V0.9 physics horizon must lie inside the training rollout")
        if len(self.observable_names) != len(self.observable_component_weights):
            raise ValueError("V0.9 observable names and component weights must align")
        if len(set(self.observable_names)) != len(self.observable_names):
            raise ValueError("V0.9 observable names must be unique")
        if len(self.observable_horizons) != len(self.observable_horizon_weights):
            raise ValueError("V0.9 observable horizons and weights must align")
        if self.observable_horizon_probabilities and (
            len(self.observable_horizons) != len(self.observable_horizon_probabilities)
        ):
            raise ValueError("V0.9 observable horizons and probabilities must align")
        if self.observable_horizons:
            if tuple(sorted(set(self.observable_horizons))) != self.observable_horizons:
                raise ValueError("V0.9 observable horizons must be unique and increasing")
            if self.observable_horizons[0] < 1:
                raise ValueError("V0.9 observable horizons must be positive")
            if self.observable_horizons[-1] > self.rollout_horizons[-1] and not self.phase1_enabled:
                raise ValueError("V0.9 observables beyond latent rollout require phase1_enabled")
            if not any(weight > 0 for weight in self.observable_horizon_weights):
                raise ValueError("V0.9 observable curriculum requires a positive horizon weight")
            if (
                self.observable_horizon_probabilities
                and not abs(sum(self.observable_horizon_probabilities) - 1.0) <= 1.0e-6
            ):
                raise ValueError("V0.9 observable horizon probabilities must sum to one")
        if self.propagator_growth_margin < 0 or self.operator_burden_target <= 0:
            raise ValueError("invalid V0.9 growth margin or burden target")
        if self.gradient_clip_norm <= 0 or self.operator_initialization_seed < 0:
            raise ValueError("invalid V0.9 clipping or initialization seed")
        if self.observable_scale_method not in {"mad", "rms"}:
            raise ValueError("invalid V0.9 observable scale method")
        if (
            min(
                self.observable_scale_epsilon,
                self.observable_huber_delta,
                self.augmented_lagrangian_initial_penalty,
                self.augmented_lagrangian_improvement_ratio,
            )
            <= 0
        ):
            raise ValueError("invalid V0.9 phase-1 positive hyperparameter")
        if self.augmented_lagrangian_penalty_growth < 1:
            raise ValueError("V0.9 augmented-Lagrangian penalty growth must be >=1")
        if self.augmented_lagrangian_max_penalty < self.augmented_lagrangian_initial_penalty:
            raise ValueError("V0.9 augmented-Lagrangian maximum penalty is too small")
        if not 0 < self.augmented_lagrangian_improvement_ratio <= 1:
            raise ValueError("invalid V0.9 augmented-Lagrangian improvement ratio")
        if self.augmented_lagrangian_dual_step_size <= 0:
            raise ValueError("invalid V0.9 augmented-Lagrangian dual step")
        if self.augmented_lagrangian_max_multiplier <= 0:
            raise ValueError("invalid V0.9 augmented-Lagrangian multiplier cap")
        if (
            min(
                self.observable_scale_max_samples,
                self.force_window_stride,
                self.augmented_lagrangian_update_interval,
            )
            < 1
            or self.gradient_audit_interval < 0
        ):
            raise ValueError("invalid V0.9 phase-1 interval or sample count")
        if self.precision not in {"fp32", "amp_fp16", "amp_bf16"}:
            raise ValueError("invalid V0.9 precision")
        if self.gradient_conflict_method not in {"none", "pcgrad"}:
            raise ValueError("invalid V0.9 gradient conflict method")
        if not 0 <= self.gradient_conflict_start_fraction <= 1:
            raise ValueError("invalid V0.9 gradient conflict start fraction")
        if self.phase1_enabled:
            if self.lambda_physics <= 0 or not self.observable_horizons:
                raise ValueError("phase-1 V0.9 requires physical observable training")
            required_constraints = {"divergence", "boundary"}
            if not required_constraints <= set(self.observable_names):
                raise ValueError("phase-1 V0.9 requires divergence and boundary observables")

    @property
    def active_observable_horizons(self) -> tuple[int, ...]:
        return self.observable_horizons or (self.physics_horizon,)

    @property
    def active_observable_horizon_weights(self) -> tuple[float, ...]:
        return self.observable_horizon_weights or (1.0,)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name in self.__dataclass_fields__
            for value in (getattr(self, name),)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V09TrainingConfig:
        defaults = cls()
        allowed = set(defaults.__dataclass_fields__)
        _reject_unknown(data, allowed, "V0.9 training config")
        values = {name: data.get(name, getattr(defaults, name)) for name in allowed}
        tuple_int_fields = {"rollout_horizons", "observable_horizons"}
        tuple_float_fields = {
            "rollout_start_fractions",
            "rollout_weights",
            "observable_component_weights",
            "observable_horizon_weights",
            "observable_horizon_probabilities",
        }
        tuple_string_fields = {"observable_names"}
        integer_fields = {
            "epochs",
            "batch_size",
            "rollout_batch_size",
            "rollout_stride",
            "physics_horizon",
            "physics_batch_size",
            "rank_sweep_epochs",
            "patience",
            "operator_initialization_seed",
            "observable_scale_max_samples",
            "force_window_stride",
            "augmented_lagrangian_update_interval",
            "gradient_audit_interval",
        }
        for name in integer_fields:
            values[name] = int(values[name])
        for name in tuple_int_fields:
            values[name] = tuple(int(value) for value in values[name])
        for name in tuple_float_fields:
            values[name] = tuple(float(value) for value in values[name])
        for name in tuple_string_fields:
            values[name] = tuple(str(value) for value in values[name])
        for name in allowed - {
            *integer_fields,
            *tuple_int_fields,
            *tuple_float_fields,
            *tuple_string_fields,
            "precision",
            "observable_scale_method",
            "phase1_enabled",
            "gradient_conflict_method",
        }:
            values[name] = float(values[name])
        values["precision"] = str(values["precision"])
        values["observable_scale_method"] = str(values["observable_scale_method"])
        values["gradient_conflict_method"] = str(values["gradient_conflict_method"])
        values["phase1_enabled"] = bool(values["phase1_enabled"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class V09EvaluationConfig:
    """Predeclared adaptive-operator and V1.0 readiness gates."""

    rollout_horizons: tuple[int, ...] = (8, 16, 32, 80)
    material_relative_gain: float = 0.02
    min_operator_explained_fraction: float = 0.02
    min_dynamic_over_static_gain: float = 0.02
    max_operator_burden: float = 0.50
    max_physics_degradation: float = 0.10
    max_divergence_mse: float = 0.02
    max_boundary_mse: float = 0.05
    observable_pair_pass_fraction: float = 1.0
    frequency_resolution_bins: float = 1.0
    scientific_seed_fraction: float = 2.0 / 3.0
    v1_0_readiness_fraction: float = 1.0
    operator_initialization_seeds: tuple[int, ...] = (701, 809, 907)

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.rollout_horizons))) != self.rollout_horizons:
            raise ValueError("V0.9 rollout horizons must be unique and increasing")
        if not self.rollout_horizons or self.rollout_horizons[0] < 1:
            raise ValueError("V0.9 rollout horizons must be positive")
        if not 0 <= self.material_relative_gain < 1:
            raise ValueError("V0.9 material gain must lie in [0,1)")
        if not 0 <= self.min_operator_explained_fraction < 1:
            raise ValueError("V0.9 Gamma_op threshold must lie in [0,1)")
        if not 0 <= self.min_dynamic_over_static_gain < 1:
            raise ValueError("V0.9 dynamic/static threshold must lie in [0,1)")
        if self.max_operator_burden <= 0 or self.max_physics_degradation < 0:
            raise ValueError("invalid V0.9 burden/physics threshold")
        if self.max_divergence_mse <= 0 or self.max_boundary_mse <= 0:
            raise ValueError("invalid V0.9 physical absolute threshold")
        if not 0 < self.observable_pair_pass_fraction <= 1:
            raise ValueError("V0.9 observable pair fraction must lie in (0,1]")
        if self.frequency_resolution_bins < 0:
            raise ValueError("V0.9 frequency resolution bins must be non-negative")
        if not 0.5 <= self.scientific_seed_fraction <= 1:
            raise ValueError("invalid V0.9 scientific seed threshold")
        if self.v1_0_readiness_fraction != 1.0:
            raise ValueError("V1.0 readiness requires all backbone/data seeds")
        if (
            len(self.operator_initialization_seeds) != 3
            or len(set(self.operator_initialization_seeds)) != 3
        ):
            raise ValueError("V0.9 formal evaluation requires three operator seeds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_horizons": list(self.rollout_horizons),
            "material_relative_gain": self.material_relative_gain,
            "min_operator_explained_fraction": self.min_operator_explained_fraction,
            "min_dynamic_over_static_gain": self.min_dynamic_over_static_gain,
            "max_operator_burden": self.max_operator_burden,
            "max_physics_degradation": self.max_physics_degradation,
            "max_divergence_mse": self.max_divergence_mse,
            "max_boundary_mse": self.max_boundary_mse,
            "observable_pair_pass_fraction": self.observable_pair_pass_fraction,
            "frequency_resolution_bins": self.frequency_resolution_bins,
            "scientific_seed_fraction": self.scientific_seed_fraction,
            "v1_0_readiness_fraction": self.v1_0_readiness_fraction,
            "operator_initialization_seeds": list(self.operator_initialization_seeds),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V09EvaluationConfig:
        defaults = cls()
        _reject_unknown(data, set(defaults.to_dict()), "V0.9 evaluation config")
        return cls(
            rollout_horizons=tuple(
                int(value) for value in data.get("rollout_horizons", defaults.rollout_horizons)
            ),
            material_relative_gain=float(
                data.get("material_relative_gain", defaults.material_relative_gain)
            ),
            min_operator_explained_fraction=float(
                data.get(
                    "min_operator_explained_fraction",
                    defaults.min_operator_explained_fraction,
                )
            ),
            min_dynamic_over_static_gain=float(
                data.get(
                    "min_dynamic_over_static_gain",
                    defaults.min_dynamic_over_static_gain,
                )
            ),
            max_operator_burden=float(
                data.get("max_operator_burden", defaults.max_operator_burden)
            ),
            max_physics_degradation=float(
                data.get("max_physics_degradation", defaults.max_physics_degradation)
            ),
            max_divergence_mse=float(data.get("max_divergence_mse", defaults.max_divergence_mse)),
            max_boundary_mse=float(data.get("max_boundary_mse", defaults.max_boundary_mse)),
            observable_pair_pass_fraction=float(
                data.get(
                    "observable_pair_pass_fraction",
                    defaults.observable_pair_pass_fraction,
                )
            ),
            frequency_resolution_bins=float(
                data.get("frequency_resolution_bins", defaults.frequency_resolution_bins)
            ),
            scientific_seed_fraction=float(
                data.get("scientific_seed_fraction", defaults.scientific_seed_fraction)
            ),
            v1_0_readiness_fraction=float(
                data.get("v1_0_readiness_fraction", defaults.v1_0_readiness_fraction)
            ),
            operator_initialization_seeds=tuple(
                int(value)
                for value in data.get(
                    "operator_initialization_seeds", defaults.operator_initialization_seeds
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class V09Phase2Config:
    """Identifiable condition/dynamic factorization over the frozen V0.8 handoff."""

    enabled: bool = False
    static_rank: int = 4
    dynamic_rank: int = 4
    observer_width: int = 64
    observer_depth: int = 2
    observer_warmup_fraction: float = 0.20
    observer_output_limit: float = 16.0
    static_stage_end_fraction: float = 0.30
    dynamic_stage_end_fraction: float = 0.70
    initial_symmetric_delta_budget: float = 0.05
    intermediate_symmetric_delta_budget: float = 0.10
    symmetric_delta_budget: float = 0.15
    lambda_condition_observer: float = 1.0
    lambda_context_residualization: float = 0.10
    lambda_condition_centering: float = 0.10
    lambda_basis_cross_orthogonality: float = 0.10
    conditional_centering_bandwidth: float = 1.0
    max_condition_observer_normalized_rmse: float = 0.50
    min_condition_observer_r2: float = 0.20
    paired_horizon: int = 8
    matched_condition_tolerance: float = 0.20
    matched_latent_tolerance: float = 0.75
    minimum_history_separation: float = 0.25
    minimum_future_separation: float = 0.05
    minimum_identifiable_pairs: int = 8
    min_paired_dynamic_gain: float = 0.02
    schedule_variants: tuple[str, ...] = (
        "smooth_up_slow",
        "smooth_up_fast",
        "abrupt_up",
        "smooth_up_medium",
        "smooth_down_slow",
        "smooth_down_fast",
        "abrupt_down",
        "smooth_down_medium",
        "cyclic_slow_long",
        "cyclic_slow_short",
        "cyclic_fast_long",
        "cyclic_fast_short",
    )

    def __post_init__(self) -> None:
        if min(self.static_rank, self.dynamic_rank, self.observer_width, self.observer_depth) < 1:
            raise ValueError("V0.9 Phase-2 ranks and observer dimensions must be positive")
        if not 0 <= self.observer_warmup_fraction < 1:
            raise ValueError("V0.9 Phase-2 observer warmup must lie in [0,1)")
        if not (0 < self.static_stage_end_fraction < self.dynamic_stage_end_fraction < 1):
            raise ValueError("V0.9 Phase-2 stage boundaries must satisfy 0 < static < dynamic < 1")
        positive = (
            self.lambda_condition_observer,
            self.lambda_context_residualization,
            self.lambda_condition_centering,
            self.lambda_basis_cross_orthogonality,
            self.conditional_centering_bandwidth,
            self.observer_output_limit,
            self.initial_symmetric_delta_budget,
            self.intermediate_symmetric_delta_budget,
            self.symmetric_delta_budget,
            self.max_condition_observer_normalized_rmse,
            self.matched_condition_tolerance,
            self.matched_latent_tolerance,
            self.minimum_history_separation,
            self.minimum_future_separation,
            self.min_paired_dynamic_gain,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("V0.9 Phase-2 losses, scales, and tolerances must be positive")
        if not (
            self.initial_symmetric_delta_budget
            <= self.intermediate_symmetric_delta_budget
            <= self.symmetric_delta_budget
        ):
            raise ValueError("V0.9 Phase-2 stability budgets must be non-decreasing")
        if not -1 < self.min_condition_observer_r2 < 1:
            raise ValueError("V0.9 Phase-2 observer R2 gate must lie in (-1,1)")
        if self.paired_horizon < 1 or self.minimum_identifiable_pairs < 1:
            raise ValueError("V0.9 Phase-2 paired audit dimensions must be positive")
        if len(self.schedule_variants) < 6 or len(set(self.schedule_variants)) != len(
            self.schedule_variants
        ):
            raise ValueError("V0.9 Phase-2 requires unique multi-route schedule variants")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name in self.__dataclass_fields__
            for value in (getattr(self, name),)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V09Phase2Config:
        defaults = cls()
        allowed = set(defaults.__dataclass_fields__)
        _reject_unknown(data, allowed, "V0.9 Phase-2 config")
        values = {name: data.get(name, getattr(defaults, name)) for name in allowed}
        for name in (
            "static_rank",
            "dynamic_rank",
            "observer_width",
            "observer_depth",
            "paired_horizon",
            "minimum_identifiable_pairs",
        ):
            values[name] = int(values[name])
        values["enabled"] = bool(values["enabled"])
        values["schedule_variants"] = tuple(str(value) for value in values["schedule_variants"])
        for name in allowed - {
            "enabled",
            "schedule_variants",
            "static_rank",
            "dynamic_rank",
            "observer_width",
            "observer_depth",
            "paired_horizon",
            "minimum_identifiable_pairs",
        }:
            values[name] = float(values[name])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class V09Phase3Config:
    """Physical-manifold and representation-sufficiency audit before matched retraining."""

    enabled: bool = False
    source_phase2_result: str = ""
    routes: tuple[str, ...] = ("frozen", "joint", "from_scratch")
    primary_decoder_candidate: str = "streamfunction"
    joint_backbone_allowlist: tuple[str, ...] = (
        "online_encoder.projection",
        "training_decoder.refine.2",
    )
    audit_samples_per_trajectory: int = 4
    tangent_epsilon: float = 1.0e-3
    max_roundtrip_nrmse: float = 0.25
    max_reconstruction_physics_degradation: float = 0.10
    max_tangent_divergence: float = 0.10
    representation_learning_rate: float = 5.0e-5
    operator_learning_rate: float = 5.0e-4
    lambda_roundtrip: float = 0.20
    lambda_jepa_consistency: float = 0.20
    lambda_physical_manifold: float = 1.0
    max_normalized_representation_drift: float = 0.10

    def __post_init__(self) -> None:
        if self.enabled and not self.source_phase2_result.strip():
            raise ValueError("V0.9 Phase-3 requires an explicit Phase-2 source result")
        if self.routes != ("frozen", "joint", "from_scratch"):
            raise ValueError("V0.9 Phase-3 routes must be frozen/joint/from_scratch")
        if self.primary_decoder_candidate not in {"streamfunction", "hodge"}:
            raise ValueError("invalid V0.9 Phase-3 physical decoder candidate")
        if not self.joint_backbone_allowlist or any(
            not value.strip() for value in self.joint_backbone_allowlist
        ):
            raise ValueError("V0.9 Phase-3 joint route requires a parameter allow-list")
        if self.audit_samples_per_trajectory < 1:
            raise ValueError("V0.9 Phase-3 audit requires positive sample count")
        positive = (
            self.tangent_epsilon,
            self.max_roundtrip_nrmse,
            self.max_reconstruction_physics_degradation,
            self.max_tangent_divergence,
            self.representation_learning_rate,
            self.operator_learning_rate,
            self.lambda_roundtrip,
            self.lambda_jepa_consistency,
            self.lambda_physical_manifold,
            self.max_normalized_representation_drift,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("V0.9 Phase-3 scales and tolerances must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name in self.__dataclass_fields__
            for value in (getattr(self, name),)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V09Phase3Config:
        defaults = cls()
        allowed = set(defaults.__dataclass_fields__)
        _reject_unknown(data, allowed, "V0.9 Phase-3 config")
        values = {name: data.get(name, getattr(defaults, name)) for name in allowed}
        values["enabled"] = bool(values["enabled"])
        values["source_phase2_result"] = str(values["source_phase2_result"])
        values["routes"] = tuple(str(value) for value in values["routes"])
        values["joint_backbone_allowlist"] = tuple(
            str(value) for value in values["joint_backbone_allowlist"]
        )
        values["primary_decoder_candidate"] = str(values["primary_decoder_candidate"])
        values["audit_samples_per_trajectory"] = int(values["audit_samples_per_trajectory"])
        for name in allowed - {
            "enabled",
            "source_phase2_result",
            "routes",
            "joint_backbone_allowlist",
            "primary_decoder_candidate",
            "audit_samples_per_trajectory",
        }:
            values[name] = float(values[name])
        return cls(**values)


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
            split=SplitConfig.from_dict(_ensure_mapping(data.get("split", {}), "split config")),
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
    """Resolved configuration with optional versioned experiment sections."""

    architecture: ArchitectureConfig
    training: TrainingConfig
    data: DataConfig
    koopman: KoopmanConfig | None = None
    oscillator: DampedOscillatorConfig | None = None
    duffing: DuffingConfig | None = None
    identification: DirectIdentificationConfig | None = None
    evaluation: KoopmanEvaluationConfig | None = None
    known_latent: KnownLatentConfig | None = None
    autoencoder: KoopmanAutoencoderConfig | None = None
    representation_loss: RepresentationLossConfig | None = None
    representation_training: RepresentationTrainingConfig | None = None
    representation_evaluation: RepresentationEvaluationConfig | None = None
    advection_diffusion_2d: AdvectionDiffusion2DConfig | None = None
    field_autoencoder: FieldAutoencoderConfig | None = None
    field_loss: FieldLossConfig | None = None
    v0_5_training: V05TrainingConfig | None = None
    v0_5_evaluation: V05EvaluationConfig | None = None
    jepa_loss: JEPALossConfig | None = None
    ema: EMAConfig | None = None
    v0_6_evaluation: V06EvaluationConfig | None = None
    residual_closure: ResidualClosureConfig | None = None
    residual_training: ResidualTrainingConfig | None = None
    memory_sweep: MemorySweepConfig | None = None
    v0_7_evaluation: V07EvaluationConfig | None = None
    cylinder_wake_2d: CylinderWake2DConfig | None = None
    v0_8_routing: V08RoutingConfig | None = None
    v0_8_context: V08ContextConfig | None = None
    v0_8_training: V08TrainingConfig | None = None
    v0_8_evaluation: V08EvaluationConfig | None = None
    v0_9_condition: V09ConditionConfig | None = None
    v0_9_adaptive: V09AdaptiveConfig | None = None
    v0_9_training: V09TrainingConfig | None = None
    v0_9_evaluation: V09EvaluationConfig | None = None
    v0_9_phase2: V09Phase2Config | None = None
    v0_9_phase3: V09Phase3Config | None = None
    project_version: str = PROJECT_VERSION
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.project_version not in SUPPORTED_CONFIG_PROJECT_VERSIONS:
            raise ValueError(
                f"config project version {self.project_version!r} is not supported; "
                f"expected one of {sorted(SUPPORTED_CONFIG_PROJECT_VERSIONS)!r}"
            )
        v0_3_sections = (self.oscillator, self.identification, self.evaluation)
        if any(section is not None for section in v0_3_sections) and not all(
            section is not None for section in v0_3_sections
        ):
            raise ValueError("V0.3 config must provide oscillator/identification/evaluation")
        v0_4_sections = (
            self.known_latent,
            self.autoencoder,
            self.representation_loss,
            self.representation_training,
            self.representation_evaluation,
        )
        if any(section is not None for section in v0_4_sections) and not all(
            section is not None for section in v0_4_sections
        ):
            raise ValueError("V0.4 config must provide all representation sections together")
        field_learning_sections = (
            self.field_autoencoder,
            self.field_loss,
            self.v0_5_training,
            self.v0_5_evaluation,
        )
        field_problem_sections = (self.advection_diffusion_2d, self.cylinder_wake_2d)
        if any(section is not None for section in field_learning_sections + field_problem_sections):
            if not all(section is not None for section in field_learning_sections):
                raise ValueError("field-learning configs must provide all architecture sections")
            if sum(section is not None for section in field_problem_sections) != 1:
                raise ValueError("field learning requires exactly one physical-problem config")
        v0_6_sections = (self.jepa_loss, self.ema, self.v0_6_evaluation)
        if any(section is not None for section in v0_6_sections) and not all(
            section is not None for section in v0_6_sections
        ):
            raise ValueError("V0.6 config must provide JEPA loss, EMA, and evaluation together")
        if any(section is not None for section in v0_6_sections) and not all(
            section is not None for section in field_learning_sections
        ):
            raise ValueError("V0.6 must inherit every V0.5 field-learning section")
        if any(section is not None for section in v0_6_sections) and self.project_version not in {
            V0_6_PROJECT_VERSION,
            V0_7_PROJECT_VERSION,
            V0_8_PROJECT_VERSION,
            PROJECT_VERSION,
        }:
            raise ValueError(
                "V0.6 sections require project_version 0.6.0 or a later inheriting version"
            )
        v0_7_sections = (
            self.residual_closure,
            self.residual_training,
            self.memory_sweep,
            self.v0_7_evaluation,
        )
        if any(section is not None for section in v0_7_sections) and not all(
            section is not None for section in v0_7_sections
        ):
            raise ValueError("V0.7 config must provide closure, training, and evaluation together")
        if any(section is not None for section in v0_7_sections) and not all(
            section is not None for section in v0_6_sections
        ):
            raise ValueError("V0.7 must inherit every V0.6 section")
        if any(section is not None for section in v0_7_sections) and self.project_version not in {
            V0_7_PROJECT_VERSION,
            V0_8_PROJECT_VERSION,
            PROJECT_VERSION,
        }:
            raise ValueError("V0.7 sections require project_version 0.7.0 or later")
        if self.memory_sweep is not None and self.advection_diffusion_2d is not None:
            if self.memory_sweep.history_lengths[-1] >= self.advection_diffusion_2d.num_steps:
                raise ValueError("V0.7 maximum history must be shorter than each trajectory")
        v0_8_sections = (
            self.cylinder_wake_2d,
            self.v0_8_routing,
            self.v0_8_context,
            self.v0_8_training,
            self.v0_8_evaluation,
        )
        if any(section is not None for section in v0_8_sections) and not all(
            section is not None for section in v0_8_sections
        ):
            raise ValueError(
                "V0.8 config must provide cylinder, routing, context, training, and evaluation"
            )
        if any(section is not None for section in v0_8_sections):
            if self.project_version not in {V0_8_PROJECT_VERSION, PROJECT_VERSION}:
                raise ValueError("V0.8 sections require project_version 0.8.0 or later")
            if not all(section is not None for section in v0_7_sections):
                raise ValueError("V0.8 must inherit the complete V0.7 residual assessment contract")
            assert self.cylinder_wake_2d and self.v0_8_context
            if self.data.problem_name != "cylinder_wake_2d":
                raise ValueError("V0.8 data problem_name must be cylinder_wake_2d")
            if self.data.action_dim != 0 or self.data.parameter_dim != 3:
                raise ValueError("V0.8 cylinder data requires action_dim=0/parameter_dim=3")
            if self.data.dt_mode is not DtMode.CONSTANT:
                raise ValueError("V0.8 cylinder benchmark uses fixed snapshot dt")
            if (
                self.data.constant_dt is None
                or abs(self.data.constant_dt - self.cylinder_wake_2d.snapshot_dt) > 1e-9
            ):
                raise ValueError("V0.8 data.constant_dt must match cylinder snapshot_dt")
            if self.field_autoencoder is None or self.field_autoencoder.input_channels != 3:
                raise ValueError("V0.8 cylinder model state is [u,v,p] with three channels")
            if self.koopman is None or self.koopman.state_dim != self.field_autoencoder.latent_dim:
                raise ValueError("V0.8 Koopman state_dim must equal the field latent dimension")
            if not self.koopman.trainable or self.koopman.dtype != "float32":
                raise ValueError("V0.8 reuses the trainable float32 V0.6 Koopman contract")
            if self.data.normalization.kind != "standard" or self.data.horizon < 2:
                raise ValueError("V0.8 requires train-only standardization and multi-step training")
            if self.v0_5_evaluation and (
                self.v0_5_evaluation.long_horizon > self.cylinder_wake_2d.num_steps
            ):
                raise ValueError("V0.8 field rollout horizon exceeds cylinder trajectory length")
            if self.memory_sweep and (
                self.memory_sweep.history_lengths[-1] >= self.cylinder_wake_2d.num_steps
            ):
                raise ValueError("V0.8 cylinder trajectories are too short for the V0.7 sweep")
            if self.v0_8_context.history_length >= self.cylinder_wake_2d.num_steps:
                raise ValueError("V0.8 context history must be shorter than each trajectory")
        v0_9_sections = (
            self.v0_9_condition,
            self.v0_9_adaptive,
            self.v0_9_training,
            self.v0_9_evaluation,
        )
        if any(section is not None for section in v0_9_sections) and not all(
            section is not None for section in v0_9_sections
        ):
            raise ValueError("V0.9 config must provide condition/adaptive/training/evaluation")
        if any(section is not None for section in v0_9_sections):
            if self.project_version != PROJECT_VERSION:
                raise ValueError("V0.9 sections require project_version 0.9.0")
            if not all(section is not None for section in v0_8_sections):
                raise ValueError("V0.9 must inherit the complete V0.8 context contract")
            if self.training.stage is not TrainStage.ADAPTIVE:
                raise ValueError("V0.9 training.stage must be adaptive")
            assert self.cylinder_wake_2d and self.koopman and self.v0_9_adaptive
            if not self.cylinder_wake_2d.time_varying_boundary:
                raise ValueError("V0.9 requires time_varying_boundary=true")
            if self.v0_9_adaptive.rank >= self.koopman.state_dim:
                raise ValueError("V0.9 requires strict low rank r < d_K")
            if self.v0_9_phase2 is not None and self.v0_9_phase2.enabled:
                assert self.v0_9_condition is not None
                if (
                    self.v0_9_phase2.static_rank + self.v0_9_phase2.dynamic_rank
                    != self.v0_9_adaptive.rank
                ):
                    raise ValueError("V0.9 Phase-2 static/dynamic ranks must sum to adaptive rank")
                if self.v0_9_condition.schedule_types != self.v0_9_phase2.schedule_variants:
                    raise ValueError(
                        "V0.9 Phase-2 schedule variants must match the condition contract"
                    )
                if (
                    self.cylinder_wake_2d.num_trajectories < len(self.v0_9_phase2.schedule_variants)
                    or self.cylinder_wake_2d.num_trajectories
                    % len(self.v0_9_phase2.schedule_variants)
                    != 0
                ):
                    raise ValueError(
                        "V0.9 Phase-2 requires balanced coverage of every schedule variant"
                    )
                if self.v0_9_phase2.paired_horizon > self.v0_9_training.rollout_horizons[-1]:
                    raise ValueError("V0.9 Phase-2 paired horizon exceeds training rollout")
            if self.v0_9_phase3 is not None and self.v0_9_phase3.enabled:
                if self.v0_9_phase2 is None or not self.v0_9_phase2.enabled:
                    raise ValueError("V0.9 Phase-3 requires the Phase-2 diagnostic contract")
        elif self.cylinder_wake_2d is not None and self.cylinder_wake_2d.time_varying_boundary:
            raise ValueError("time-varying cylinder boundaries require the complete V0.9 contract")
        if (
            any(section is not None for section in v0_3_sections)
            or any(section is not None for section in v0_4_sections)
            or any(section is not None for section in field_learning_sections)
        ) and self.koopman is None:
            raise ValueError("versioned Koopman experiments require a Koopman config")
        if self.oscillator is not None:
            assert self.koopman is not None
            if self.koopman.state_dim != 2:
                raise ValueError("V0.3 oscillator experiments require Koopman state_dim=2")
            expected_mode = DtMode.VARIABLE if self.oscillator.variable_dt else DtMode.CONSTANT
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
        if self.known_latent is not None:
            assert self.koopman is not None
            assert self.autoencoder is not None
            assert self.representation_evaluation is not None
            if self.koopman.state_dim != self.autoencoder.latent_dim:
                raise ValueError("Koopman state_dim must equal autoencoder latent_dim")
            if not self.koopman.trainable:
                raise ValueError("V0.4 representation learning requires koopman.trainable=true")
            if self.autoencoder.observation_dim != 5:
                raise ValueError("V0.4 nonlinear observation requires observation_dim=5")
            if self.data.problem_name != "known_latent_nonlinear_observation":
                raise ValueError("V0.4 data problem_name is incompatible")
            if self.data.action_dim != 0 or self.data.parameter_dim != 2:
                raise ValueError("V0.4 known-latent data requires action_dim=0/parameter_dim=2")
            expected_mode = DtMode.VARIABLE if self.known_latent.variable_dt else DtMode.CONSTANT
            if self.data.dt_mode is not expected_mode:
                raise ValueError("V0.4 data dt_mode must match known_latent variable_dt")
            if expected_mode is DtMode.CONSTANT:
                assert self.data.constant_dt is not None
                if abs(self.data.constant_dt - self.known_latent.base_dt) > 1e-12:
                    raise ValueError("V0.4 constant_dt must match known_latent base_dt")
            if self.data.normalization.kind != "standard":
                raise ValueError(
                    "V0.4 known-latent data requires train-only standard normalization"
                )
            if self.data.horizon < 2:
                raise ValueError("V0.4 multi-step training requires data horizon > 1")
            if self.known_latent.num_steps < self.data.history + self.data.horizon - 1:
                raise ValueError("V0.4 trajectories are too short for history/horizon")
            if self.representation_evaluation.rollout_horizon > self.known_latent.num_steps:
                raise ValueError("V0.4 rollout horizon exceeds known-latent trajectory length")
        if self.advection_diffusion_2d is not None:
            assert self.koopman is not None
            assert self.field_autoencoder is not None
            assert self.v0_5_evaluation is not None
            pde = self.advection_diffusion_2d
            if self.data.problem_name != "periodic_advection_diffusion_2d":
                raise ValueError("V0.5 data problem_name is incompatible")
            if self.data.action_dim != 0 or self.data.parameter_dim != 3:
                raise ValueError("V0.5 data requires action_dim=0/parameter_dim=3")
            if self.field_autoencoder.input_channels != 1:
                raise ValueError("V0.5 scalar PDE requires one field channel")
            if self.koopman.state_dim != self.field_autoencoder.latent_dim:
                raise ValueError("V0.5 Koopman state_dim must equal field latent_dim")
            if not self.koopman.trainable or self.koopman.dtype != "float32":
                raise ValueError("V0.5 requires a trainable float32 Koopman core")
            expected_mode = DtMode.VARIABLE if pde.variable_dt else DtMode.CONSTANT
            if self.data.dt_mode is not expected_mode:
                raise ValueError("V0.5 data dt_mode must match PDE variable_dt")
            if expected_mode is DtMode.CONSTANT:
                assert self.data.constant_dt is not None
                if abs(self.data.constant_dt - pde.base_dt) > 1e-12:
                    raise ValueError("V0.5 constant_dt must match PDE base_dt")
            if self.data.normalization.kind != "standard":
                raise ValueError("V0.5 requires train-only standard normalization")
            if self.data.horizon < 2:
                raise ValueError("V0.5 multi-step training requires horizon > 1")
            if pde.num_steps < self.data.history + self.data.horizon - 1:
                raise ValueError("V0.5 trajectories are too short for history/horizon")
            if self.v0_5_evaluation.long_horizon > pde.num_steps:
                raise ValueError("V0.5 long evaluation horizon exceeds trajectory length")

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
            "known_latent": (None if self.known_latent is None else self.known_latent.to_dict()),
            "autoencoder": None if self.autoencoder is None else self.autoencoder.to_dict(),
            "representation_loss": (
                None if self.representation_loss is None else self.representation_loss.to_dict()
            ),
            "representation_training": (
                None
                if self.representation_training is None
                else self.representation_training.to_dict()
            ),
            "representation_evaluation": (
                None
                if self.representation_evaluation is None
                else self.representation_evaluation.to_dict()
            ),
            "advection_diffusion_2d": (
                None
                if self.advection_diffusion_2d is None
                else self.advection_diffusion_2d.to_dict()
            ),
            "field_autoencoder": (
                None if self.field_autoencoder is None else self.field_autoencoder.to_dict()
            ),
            "field_loss": None if self.field_loss is None else self.field_loss.to_dict(),
            "v0_5_training": None if self.v0_5_training is None else self.v0_5_training.to_dict(),
            "v0_5_evaluation": None
            if self.v0_5_evaluation is None
            else self.v0_5_evaluation.to_dict(),
            "jepa_loss": None if self.jepa_loss is None else self.jepa_loss.to_dict(),
            "ema": None if self.ema is None else self.ema.to_dict(),
            "v0_6_evaluation": None
            if self.v0_6_evaluation is None
            else self.v0_6_evaluation.to_dict(),
            "residual_closure": None
            if self.residual_closure is None
            else self.residual_closure.to_dict(),
            "residual_training": None
            if self.residual_training is None
            else self.residual_training.to_dict(),
            "memory_sweep": None if self.memory_sweep is None else self.memory_sweep.to_dict(),
            "v0_7_evaluation": None
            if self.v0_7_evaluation is None
            else self.v0_7_evaluation.to_dict(),
            "cylinder_wake_2d": None
            if self.cylinder_wake_2d is None
            else self.cylinder_wake_2d.to_dict(),
            "v0_8_routing": None if self.v0_8_routing is None else self.v0_8_routing.to_dict(),
            "v0_8_context": None if self.v0_8_context is None else self.v0_8_context.to_dict(),
            "v0_8_training": None if self.v0_8_training is None else self.v0_8_training.to_dict(),
            "v0_8_evaluation": None
            if self.v0_8_evaluation is None
            else self.v0_8_evaluation.to_dict(),
            "v0_9_condition": None
            if self.v0_9_condition is None
            else self.v0_9_condition.to_dict(),
            "v0_9_adaptive": None if self.v0_9_adaptive is None else self.v0_9_adaptive.to_dict(),
            "v0_9_training": None if self.v0_9_training is None else self.v0_9_training.to_dict(),
            "v0_9_evaluation": None
            if self.v0_9_evaluation is None
            else self.v0_9_evaluation.to_dict(),
            "v0_9_phase2": None if self.v0_9_phase2 is None else self.v0_9_phase2.to_dict(),
            "v0_9_phase3": None if self.v0_9_phase3 is None else self.v0_9_phase3.to_dict(),
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
                "known_latent",
                "autoencoder",
                "representation_loss",
                "representation_training",
                "representation_evaluation",
                "advection_diffusion_2d",
                "field_autoencoder",
                "field_loss",
                "v0_5_training",
                "v0_5_evaluation",
                "jepa_loss",
                "ema",
                "v0_6_evaluation",
                "residual_closure",
                "residual_training",
                "memory_sweep",
                "v0_7_evaluation",
                "cylinder_wake_2d",
                "v0_8_routing",
                "v0_8_context",
                "v0_8_training",
                "v0_8_evaluation",
                "v0_9_condition",
                "v0_9_adaptive",
                "v0_9_training",
                "v0_9_evaluation",
                "v0_9_phase2",
                "v0_9_phase3",
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
            known_latent=(
                None
                if data.get("known_latent") is None
                else KnownLatentConfig.from_dict(
                    _ensure_mapping(data["known_latent"], "known-latent config")
                )
            ),
            autoencoder=(
                None
                if data.get("autoencoder") is None
                else KoopmanAutoencoderConfig.from_dict(
                    _ensure_mapping(data["autoencoder"], "Koopman autoencoder config")
                )
            ),
            representation_loss=(
                None
                if data.get("representation_loss") is None
                else RepresentationLossConfig.from_dict(
                    _ensure_mapping(data["representation_loss"], "representation loss config")
                )
            ),
            representation_training=(
                None
                if data.get("representation_training") is None
                else RepresentationTrainingConfig.from_dict(
                    _ensure_mapping(
                        data["representation_training"], "representation training config"
                    )
                )
            ),
            representation_evaluation=(
                None
                if data.get("representation_evaluation") is None
                else RepresentationEvaluationConfig.from_dict(
                    _ensure_mapping(
                        data["representation_evaluation"], "representation evaluation config"
                    )
                )
            ),
            advection_diffusion_2d=(
                None
                if data.get("advection_diffusion_2d") is None
                else AdvectionDiffusion2DConfig.from_dict(
                    _ensure_mapping(
                        data["advection_diffusion_2d"], "V0.5 advection-diffusion config"
                    )
                )
            ),
            field_autoencoder=(
                None
                if data.get("field_autoencoder") is None
                else FieldAutoencoderConfig.from_dict(
                    _ensure_mapping(data["field_autoencoder"], "V0.5 field autoencoder config")
                )
            ),
            field_loss=(
                None
                if data.get("field_loss") is None
                else FieldLossConfig.from_dict(
                    _ensure_mapping(data["field_loss"], "V0.5 field loss config")
                )
            ),
            v0_5_training=(
                None
                if data.get("v0_5_training") is None
                else V05TrainingConfig.from_dict(
                    _ensure_mapping(data["v0_5_training"], "V0.5 training config")
                )
            ),
            v0_5_evaluation=(
                None
                if data.get("v0_5_evaluation") is None
                else V05EvaluationConfig.from_dict(
                    _ensure_mapping(data["v0_5_evaluation"], "V0.5 evaluation config")
                )
            ),
            jepa_loss=(
                None
                if data.get("jepa_loss") is None
                else JEPALossConfig.from_dict(
                    _ensure_mapping(data["jepa_loss"], "V0.6 JEPA loss config")
                )
            ),
            ema=(
                None
                if data.get("ema") is None
                else EMAConfig.from_dict(_ensure_mapping(data["ema"], "V0.6 EMA config"))
            ),
            v0_6_evaluation=(
                None
                if data.get("v0_6_evaluation") is None
                else V06EvaluationConfig.from_dict(
                    _ensure_mapping(data["v0_6_evaluation"], "V0.6 evaluation config")
                )
            ),
            residual_closure=(
                None
                if data.get("residual_closure") is None
                else ResidualClosureConfig.from_dict(
                    _ensure_mapping(data["residual_closure"], "V0.7 residual closure config")
                )
            ),
            residual_training=(
                None
                if data.get("residual_training") is None
                else ResidualTrainingConfig.from_dict(
                    _ensure_mapping(data["residual_training"], "V0.7 residual training config")
                )
            ),
            memory_sweep=(
                None
                if data.get("memory_sweep") is None
                else MemorySweepConfig.from_dict(
                    _ensure_mapping(data["memory_sweep"], "V0.7 memory sweep config")
                )
            ),
            v0_7_evaluation=(
                None
                if data.get("v0_7_evaluation") is None
                else V07EvaluationConfig.from_dict(
                    _ensure_mapping(data["v0_7_evaluation"], "V0.7 evaluation config")
                )
            ),
            cylinder_wake_2d=(
                None
                if data.get("cylinder_wake_2d") is None
                else CylinderWake2DConfig.from_dict(
                    _ensure_mapping(data["cylinder_wake_2d"], "V0.8 cylinder-wake config")
                )
            ),
            v0_8_routing=(
                None
                if data.get("v0_8_routing") is None
                else V08RoutingConfig.from_dict(
                    _ensure_mapping(data["v0_8_routing"], "V0.8 routing config")
                )
            ),
            v0_8_context=(
                None
                if data.get("v0_8_context") is None
                else V08ContextConfig.from_dict(
                    _ensure_mapping(data["v0_8_context"], "V0.8 context config")
                )
            ),
            v0_8_training=(
                None
                if data.get("v0_8_training") is None
                else V08TrainingConfig.from_dict(
                    _ensure_mapping(data["v0_8_training"], "V0.8 training config")
                )
            ),
            v0_8_evaluation=(
                None
                if data.get("v0_8_evaluation") is None
                else V08EvaluationConfig.from_dict(
                    _ensure_mapping(data["v0_8_evaluation"], "V0.8 evaluation config")
                )
            ),
            v0_9_condition=(
                None
                if data.get("v0_9_condition") is None
                else V09ConditionConfig.from_dict(
                    _ensure_mapping(data["v0_9_condition"], "V0.9 condition config")
                )
            ),
            v0_9_adaptive=(
                None
                if data.get("v0_9_adaptive") is None
                else V09AdaptiveConfig.from_dict(
                    _ensure_mapping(data["v0_9_adaptive"], "V0.9 adaptive config")
                )
            ),
            v0_9_training=(
                None
                if data.get("v0_9_training") is None
                else V09TrainingConfig.from_dict(
                    _ensure_mapping(data["v0_9_training"], "V0.9 training config")
                )
            ),
            v0_9_evaluation=(
                None
                if data.get("v0_9_evaluation") is None
                else V09EvaluationConfig.from_dict(
                    _ensure_mapping(data["v0_9_evaluation"], "V0.9 evaluation config")
                )
            ),
            v0_9_phase2=(
                None
                if data.get("v0_9_phase2") is None
                else V09Phase2Config.from_dict(
                    _ensure_mapping(data["v0_9_phase2"], "V0.9 Phase-2 config")
                )
            ),
            v0_9_phase3=(
                None
                if data.get("v0_9_phase3") is None
                else V09Phase3Config.from_dict(
                    _ensure_mapping(data["v0_9_phase3"], "V0.9 Phase-3 config")
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
