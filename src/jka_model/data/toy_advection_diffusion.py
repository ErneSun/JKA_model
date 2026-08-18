"""Deterministic analytic 1D periodic advection-diffusion trajectories."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from jka_model.config import ToyAdvectionDiffusionConfig
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


def make_advection_diffusion_problem_spec(
    config: ToyAdvectionDiffusionConfig,
) -> ProblemSpec:
    """Create the static contract for ``u_t + c u_x = nu u_xx``."""
    dx = config.length / (config.nx - 1)
    return ProblemSpec(
        name="toy_periodic_advection_diffusion_1d",
        channels=(ChannelSpec("concentration", "1"),),
        spatial_dim=1,
        grid=GridSpec(
            layout="channels_first",
            shape=(config.nx,),
            spacing=(dx,),
            coordinates_required=True,
            cell_weights_required=True,
            metadata={"endpoint": "duplicated_periodic"},
        ),
        boundary=BoundarySpec("periodic", {"axis": "x"}),
        action_dim=0,
        parameter_dim=2,
        dt_mode=DtMode.VARIABLE if config.variable_dt else DtMode.CONSTANT,
        constant_dt=None if config.variable_dt else config.base_dt,
        normalization=NormalizationSpec("standard", {"fit_split": "train"}),
        geometry=GeometrySpec(mask_required=True),
        observable_requirements=("channel_mean", "channel_rms", "mass"),
        metadata={
            "equation": "u_t + c*u_x = nu*u_xx",
            "parameters": ["advection_speed_c", "diffusivity_nu"],
        },
    )


def _uniform(
    generator: torch.Generator,
    low: float,
    high: float,
    *,
    dtype: torch.dtype,
) -> Tensor:
    return low + (high - low) * torch.rand((), generator=generator, dtype=dtype)


def generate_advection_diffusion_trajectories(
    config: ToyAdvectionDiffusionConfig,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[TrajectoryDataset, ProblemSpec]:
    """Generate exact Fourier-mode solutions at constant or variable time steps."""
    if seed < 0:
        raise ValueError("toy data seed must be non-negative")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.linspace(0.0, config.length, config.nx, dtype=dtype)
    dx = config.length / (config.nx - 1)
    weights = torch.full((config.nx,), dx, dtype=dtype)
    weights[[0, -1]] *= 0.5
    valid_mask = torch.ones((1, config.nx), dtype=torch.bool)
    records: list[TrajectoryRecord] = []

    for trajectory_index in range(config.num_trajectories):
        c = _uniform(generator, config.c_min, config.c_max, dtype=dtype)
        nu = _uniform(generator, config.nu_min, config.nu_max, dtype=dtype)
        if config.variable_dt:
            multipliers = 0.7 + 0.6 * torch.rand(config.num_steps, generator=generator, dtype=dtype)
            dts = config.base_dt * multipliers
        else:
            dts = torch.full((config.num_steps,), config.base_dt, dtype=dtype)
        times = torch.cat((torch.zeros(1, dtype=dtype), torch.cumsum(dts, dim=0)))
        offset = _uniform(generator, 0.8, 1.2, dtype=dtype)
        amplitudes = 0.08 + 0.12 * torch.rand(config.modes, generator=generator, dtype=dtype)
        phases = 2.0 * math.pi * torch.rand(config.modes, generator=generator, dtype=dtype)
        state = torch.full((config.num_steps + 1, config.nx), offset, dtype=dtype)
        for mode_index in range(config.modes):
            wave_number = 2.0 * math.pi * (mode_index + 1) / config.length
            phase = wave_number * (x[None, :] - c * times[:, None]) + phases[mode_index]
            decay = torch.exp(-nu * wave_number**2 * times)[:, None]
            state += amplitudes[mode_index] * decay * torch.sin(phase)
        state[:, -1] = state[:, 0]
        records.append(
            TrajectoryRecord(
                trajectory_id=f"trajectory-{trajectory_index:04d}",
                states_raw=state[:, None, :],
                dts=dts,
                mu_static=torch.stack((c, nu)),
                coordinates=x.clone(),
                cell_weights=weights.clone(),
                valid_mask=valid_mask.clone(),
                metadata={
                    "seed": seed,
                    "analytic": True,
                    "offset_b": float(offset),
                },
            )
        )
    return TrajectoryDataset(records), make_advection_diffusion_problem_spec(config)
