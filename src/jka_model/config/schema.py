"""Small strict dataclass configuration system with stable hashing."""

from __future__ import annotations

import hashlib
import json
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
        v0_5_sections = (
            self.advection_diffusion_2d,
            self.field_autoencoder,
            self.field_loss,
            self.v0_5_training,
            self.v0_5_evaluation,
        )
        if any(section is not None for section in v0_5_sections) and not all(
            section is not None for section in v0_5_sections
        ):
            raise ValueError("V0.5 config must provide all field-learning sections together")
        v0_6_sections = (self.jepa_loss, self.ema, self.v0_6_evaluation)
        if any(section is not None for section in v0_6_sections) and not all(
            section is not None for section in v0_6_sections
        ):
            raise ValueError("V0.6 config must provide JEPA loss, EMA, and evaluation together")
        if any(section is not None for section in v0_6_sections) and not all(
            section is not None for section in v0_5_sections
        ):
            raise ValueError("V0.6 must inherit every V0.5 field-learning section")
        if any(section is not None for section in v0_6_sections) and self.project_version not in {
            V0_6_PROJECT_VERSION,
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
        if (
            any(section is not None for section in v0_7_sections)
            and self.project_version != PROJECT_VERSION
        ):
            raise ValueError("V0.7 sections require project_version 0.7.0")
        if self.memory_sweep is not None and self.advection_diffusion_2d is not None:
            if self.memory_sweep.history_lengths[-1] >= self.advection_diffusion_2d.num_steps:
                raise ValueError("V0.7 maximum history must be shorter than each trajectory")
        if (
            any(section is not None for section in v0_3_sections)
            or any(section is not None for section in v0_4_sections)
            or any(section is not None for section in v0_5_sections)
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
