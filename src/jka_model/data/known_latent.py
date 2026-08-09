"""Known-latent synthetic dynamics with nonlinear observations for V0.4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor

from jka_model.config import KnownLatentConfig
from jka_model.contracts import (
    BoundarySpec,
    ChannelSpec,
    DtMode,
    GeometrySpec,
    GridSpec,
    NormalizationSpec,
    ProblemSpec,
)
from jka_model.data.datasets import TrajectoryDataset, TrajectoryRecord
from jka_model.data.toy_oscillators import rotation_decay_transition


@dataclass(frozen=True, slots=True)
class KnownLatentDataset:
    """Public observations plus evaluation-only hidden states keyed by trajectory ID."""

    records: TrajectoryDataset
    problem_spec: ProblemSpec
    true_latents: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        identifiers = {record.trajectory_id for record in self.records}
        if set(self.true_latents) != identifiers:
            raise ValueError("true-latent keys must exactly match trajectory IDs")
        frozen: dict[str, Tensor] = {}
        for identifier, latent in self.true_latents.items():
            record = next(
                item for item in self.records if item.trajectory_id == identifier
            )
            if latent.shape != (record.num_steps + 1, 2):
                raise ValueError("true latent must have shape [T+1,2]")
            if latent.requires_grad or not torch.isfinite(latent).all():
                raise ValueError("true latent must be finite and detached")
            frozen[identifier] = latent.detach().clone()
        object.__setattr__(self, "true_latents", MappingProxyType(frozen))

    def latent(self, trajectory_id: str) -> Tensor:
        """Return an evaluation-only hidden trajectory; never part of ProblemBatch."""
        try:
            return self.true_latents[trajectory_id]
        except KeyError as error:
            raise KeyError(f"unknown trajectory ID: {trajectory_id}") from error


def hidden_rotation_decay_generator(
    alpha: float,
    omega: float,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Return the hidden continuous generator ``[[-a,-w],[w,-a]]``."""
    if alpha <= 0 or omega <= 0:
        raise ValueError("hidden alpha and omega must be positive")
    return torch.tensor([[-alpha, -omega], [omega, -alpha]], dtype=dtype)


def nonlinear_observation_map(hidden_state: Tensor) -> Tensor:
    """Map ``[...,2]`` hidden states to ``[q,p,q^2,qp,p^2]`` observations."""
    if hidden_state.ndim < 1 or hidden_state.shape[-1] != 2:
        raise ValueError("hidden_state must end with dimension 2")
    if not torch.is_floating_point(hidden_state) or not torch.isfinite(hidden_state).all():
        raise ValueError("hidden_state must be finite floating point")
    q, p = hidden_state[..., 0], hidden_state[..., 1]
    return torch.stack((q, p, q.square(), q * p, p.square()), dim=-1)


def linear_observation_map(hidden_state: Tensor) -> Tensor:
    """Identity observation used only for the V0.4 linear sanity experiment."""
    if hidden_state.ndim < 1 or hidden_state.shape[-1] != 2:
        raise ValueError("hidden_state must end with dimension 2")
    if not torch.is_floating_point(hidden_state) or not torch.isfinite(hidden_state).all():
        raise ValueError("hidden_state must be finite floating point")
    return hidden_state.clone()


def make_known_latent_problem_spec(
    config: KnownLatentConfig,
    *,
    nonlinear_observation: bool = True,
) -> ProblemSpec:
    names = (
        ("q", "p", "q_squared", "q_times_p", "p_squared")
        if nonlinear_observation
        else ("q", "p")
    )
    return ProblemSpec(
        name=(
            "known_latent_nonlinear_observation"
            if nonlinear_observation
            else "known_latent_linear_observation"
        ),
        channels=tuple(ChannelSpec(name, "1") for name in names),
        spatial_dim=0,
        grid=GridSpec(layout="channels_first", shape=()),
        boundary=BoundarySpec("none"),
        action_dim=0,
        parameter_dim=2,
        dt_mode=DtMode.VARIABLE if config.variable_dt else DtMode.CONSTANT,
        constant_dt=None if config.variable_dt else config.base_dt,
        normalization=NormalizationSpec("standard", {"fit_scope": "train_only"}),
        geometry=GeometrySpec(mask_required=False),
        observable_requirements=names,
        metadata={
            "hidden_dynamics": "rotation_decay",
            "hidden_state_is_evaluation_only": True,
        },
    )


def generate_known_latent_trajectories(
    config: KnownLatentConfig,
    *,
    seed: int,
    dtype: torch.dtype = torch.float64,
    nonlinear_observation: bool = True,
) -> KnownLatentDataset:
    """Generate exact hidden dynamics and expose only observations as trajectories."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random = torch.Generator(device="cpu").manual_seed(seed)
    records: list[TrajectoryRecord] = []
    true_latents: dict[str, Tensor] = {}
    observation_map = (
        nonlinear_observation_map if nonlinear_observation else linear_observation_map
    )
    for index in range(config.num_trajectories):
        initial = -1.25 + 2.5 * torch.rand(2, generator=random, dtype=dtype)
        if config.variable_dt:
            multipliers = 1.0 + config.dt_jitter * (
                2.0 * torch.rand(config.num_steps, generator=random, dtype=dtype) - 1.0
            )
            dts = config.base_dt * multipliers
        else:
            dts = torch.full((config.num_steps,), config.base_dt, dtype=dtype)
        hidden_states = [initial]
        for dt in dts:
            transition = rotation_decay_transition(
                config.alpha,
                config.omega,
                float(dt),
                dtype=dtype,
            )
            hidden_states.append(transition @ hidden_states[-1])
        hidden = torch.stack(hidden_states).detach()
        observations = observation_map(hidden)
        identifier = f"known-latent-{index:04d}"
        records.append(
            TrajectoryRecord(
                trajectory_id=identifier,
                states_raw=observations,
                dts=dts,
                mu_static=torch.tensor([config.alpha, config.omega], dtype=dtype),
                metadata={
                    "system": "known_latent_rotation_decay",
                    "observation": "nonlinear" if nonlinear_observation else "linear",
                    "seed": seed,
                },
            )
        )
        true_latents[identifier] = hidden
    spec = make_known_latent_problem_spec(
        config, nonlinear_observation=nonlinear_observation
    )
    return KnownLatentDataset(TrajectoryDataset(records), spec, true_latents)
