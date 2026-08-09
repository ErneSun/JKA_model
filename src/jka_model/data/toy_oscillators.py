"""Independent deterministic oscillator references for V0.3 identification."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from jka_model.config import DampedOscillatorConfig, DuffingConfig
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


def damped_oscillator_generator_matrix(
    omega0: float,
    gamma: float,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Return ``[[0,1],[-omega0^2,-2 gamma]]``."""
    if omega0 <= 0 or not 0 < gamma < omega0:
        raise ValueError("damped oscillator requires omega0 > gamma > 0")
    return torch.tensor(
        [[0.0, 1.0], [-(omega0**2), -2.0 * gamma]],
        dtype=dtype,
    )


def rotation_decay_transition(
    alpha: float,
    omega: float,
    dt: float,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Closed form of ``exp([[-a,-w],[w,-a]] dt)`` without ``matrix_exp``."""
    if alpha < 0 or omega < 0 or dt < 0:
        raise ValueError("rotation-decay alpha, omega, and dt must be non-negative")
    angle = omega * dt
    decay = math.exp(-alpha * dt)
    return decay * torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=dtype,
    )


def damped_oscillator_analytic_state(
    initial_state: Tensor,
    times: Tensor,
    *,
    omega0: float,
    gamma: float,
) -> Tensor:
    """Independent underdamped analytical ``[x(t),v(t)]`` reference."""
    if initial_state.shape != (2,) or times.ndim != 1:
        raise ValueError("initial_state must be [2] and times must be [T+1]")
    if initial_state.dtype != times.dtype or initial_state.device != times.device:
        raise ValueError("initial_state and times must share dtype/device")
    if omega0 <= 0 or not 0 < gamma < omega0:
        raise ValueError("damped oscillator requires omega0 > gamma > 0")
    x0, v0 = initial_state[0], initial_state[1]
    omega_d = math.sqrt(omega0**2 - gamma**2)
    cosine = torch.cos(omega_d * times)
    sine = torch.sin(omega_d * times)
    coefficient = (v0 + gamma * x0) / omega_d
    envelope = torch.exp(-gamma * times)
    carrier = x0 * cosine + coefficient * sine
    position = envelope * carrier
    velocity = envelope * (
        -gamma * carrier - x0 * omega_d * sine + coefficient * omega_d * cosine
    )
    return torch.stack((position, velocity), dim=-1)


def damped_oscillator_analytic_transition(
    omega0: float,
    gamma: float,
    dt: float,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Independent closed form for the physical ``[x,v]`` state transition."""
    if omega0 <= 0 or not 0 < gamma < omega0:
        raise ValueError("damped oscillator requires omega0 > gamma > 0")
    if dt < 0 or not math.isfinite(dt):
        raise ValueError("damped oscillator dt must be finite and non-negative")
    omega_d = math.sqrt(omega0**2 - gamma**2)
    angle = omega_d * dt
    sine = math.sin(angle)
    cosine = math.cos(angle)
    envelope = math.exp(-gamma * dt)
    return envelope * torch.tensor(
        [
            [cosine + gamma * sine / omega_d, sine / omega_d],
            [-(omega0**2) * sine / omega_d, cosine - gamma * sine / omega_d],
        ],
        dtype=dtype,
    )


def make_damped_oscillator_problem_spec(config: DampedOscillatorConfig) -> ProblemSpec:
    return ProblemSpec(
        name="damped_harmonic_oscillator",
        channels=(ChannelSpec("position", "1"), ChannelSpec("velocity", "1/s")),
        spatial_dim=0,
        grid=GridSpec(layout="channels_first", shape=()),
        boundary=BoundarySpec("none"),
        action_dim=0,
        parameter_dim=2,
        dt_mode=DtMode.VARIABLE if config.variable_dt else DtMode.CONSTANT,
        constant_dt=None if config.variable_dt else config.base_dt,
        normalization=NormalizationSpec("external", {"direct_state": True}),
        geometry=GeometrySpec(mask_required=False),
        observable_requirements=("position", "velocity"),
        metadata={"equation": "x_dot=v; v_dot=-omega0^2*x-2*gamma*v"},
    )


def generate_damped_oscillator_trajectories(
    config: DampedOscillatorConfig,
    *,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> tuple[TrajectoryDataset, ProblemSpec]:
    """Generate analytical trajectories without using ``torch.matrix_exp``."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    records: list[TrajectoryRecord] = []
    for index in range(config.num_trajectories):
        initial_state = -1.25 + 2.5 * torch.rand(2, generator=generator, dtype=dtype)
        if config.variable_dt:
            multipliers = 1.0 + config.dt_jitter * (
                2.0 * torch.rand(config.num_steps, generator=generator, dtype=dtype) - 1.0
            )
            dts = config.base_dt * multipliers
        else:
            dts = torch.full((config.num_steps,), config.base_dt, dtype=dtype)
        times = torch.cat((torch.zeros(1, dtype=dtype), torch.cumsum(dts, dim=0)))
        states = damped_oscillator_analytic_state(
            initial_state,
            times,
            omega0=config.omega0,
            gamma=config.gamma,
        )
        records.append(
            TrajectoryRecord(
                trajectory_id=f"damped-{index:04d}",
                states_raw=states,
                dts=dts,
                mu_static=torch.tensor([config.omega0, config.gamma], dtype=dtype),
                metadata={"system": "damped_oscillator", "analytic": True, "seed": seed},
            )
        )
    return TrajectoryDataset(records), make_damped_oscillator_problem_spec(config)


def _duffing_rhs(state: Tensor, config: DuffingConfig) -> Tensor:
    position, velocity = state[0], state[1]
    acceleration = (
        -config.delta * velocity
        - config.alpha * position
        - config.beta * position**3
    )
    return torch.stack((velocity, acceleration))


def _duffing_rk4_step(state: Tensor, dt: float, config: DuffingConfig) -> Tensor:
    sub_dt = dt / config.rk4_substeps
    result = state
    for _ in range(config.rk4_substeps):
        k1 = _duffing_rhs(result, config)
        k2 = _duffing_rhs(result + 0.5 * sub_dt * k1, config)
        k3 = _duffing_rhs(result + 0.5 * sub_dt * k2, config)
        k4 = _duffing_rhs(result + sub_dt * k3, config)
        result = result + (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return result


def make_duffing_problem_spec(config: DuffingConfig) -> ProblemSpec:
    return ProblemSpec(
        name="unforced_duffing_oscillator",
        channels=(ChannelSpec("position", "1"), ChannelSpec("velocity", "1/s")),
        spatial_dim=0,
        grid=GridSpec(layout="channels_first", shape=()),
        boundary=BoundarySpec("none"),
        action_dim=0,
        parameter_dim=3,
        dt_mode=DtMode.CONSTANT,
        constant_dt=config.dt,
        normalization=NormalizationSpec("external", {"direct_state": True}),
        geometry=GeometrySpec(mask_required=False),
        metadata={"equation": "x_dot=v; v_dot=-delta*v-alpha*x-beta*x^3"},
    )


def generate_duffing_trajectories(
    config: DuffingConfig,
    *,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> tuple[TrajectoryDataset, ProblemSpec]:
    """Generate unforced Duffing trajectories using a small reference-only RK4."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    records: list[TrajectoryRecord] = []
    for index in range(config.num_trajectories):
        initial_state = -1.2 + 2.4 * torch.rand(2, generator=generator, dtype=dtype)
        states = [initial_state]
        for _ in range(config.num_steps):
            states.append(_duffing_rk4_step(states[-1], config.dt, config))
        records.append(
            TrajectoryRecord(
                trajectory_id=f"duffing-{index:04d}",
                states_raw=torch.stack(states),
                dts=torch.full((config.num_steps,), config.dt, dtype=dtype),
                mu_static=torch.tensor(
                    [config.delta, config.alpha, config.beta], dtype=dtype
                ),
                metadata={"system": "duffing", "integrator": "rk4", "seed": seed},
            )
        )
    return TrajectoryDataset(records), make_duffing_problem_spec(config)


def trajectory_transition_tensors(
    records: TrajectoryDataset,
) -> tuple[Tensor, Tensor, Tensor]:
    """Flatten trajectory-local ``(z_i,z_{i+1},dt_i)`` pairs without crossing IDs."""
    states = torch.cat([record.states_raw[:-1] for record in records], dim=0)
    targets = torch.cat([record.states_raw[1:] for record in records], dim=0)
    dts = torch.cat([record.dts for record in records], dim=0)
    return states, targets, dts
