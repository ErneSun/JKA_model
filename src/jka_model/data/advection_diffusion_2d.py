"""Analytic periodic 2-D advection-diffusion trajectories for V0.5."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from jka_model.config import AdvectionDiffusion2DConfig
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


@dataclass(frozen=True, slots=True)
class AdvectionDiffusion2DReference:
    """Exact spectral quantities shared by every trajectory."""

    wave_number_x: float
    wave_number_y: float
    angular_frequency: float
    decay_rate: float
    domain_area: float


@dataclass(frozen=True, slots=True)
class AdvectionDiffusion2DDataset:
    records: TrajectoryDataset
    problem_spec: ProblemSpec
    reference: AdvectionDiffusion2DReference


def make_advection_diffusion_2d_problem_spec(
    config: AdvectionDiffusion2DConfig,
) -> ProblemSpec:
    dx, dy = config.length_x / config.nx, config.length_y / config.ny
    return ProblemSpec(
        name="periodic_advection_diffusion_2d",
        channels=(ChannelSpec("scalar", "1"),),
        spatial_dim=2,
        grid=GridSpec(
            layout="channels_first",
            shape=(config.nx, config.ny),
            spacing=(dx, dy),
            coordinates_required=True,
            cell_weights_required=True,
            metadata={"endpoint": False},
        ),
        boundary=BoundarySpec("periodic"),
        action_dim=0,
        parameter_dim=3,
        dt_mode=DtMode.VARIABLE if config.variable_dt else DtMode.CONSTANT,
        constant_dt=None if config.variable_dt else config.base_dt,
        normalization=NormalizationSpec("standard", {"fit_scope": "train_only"}),
        geometry=GeometrySpec(mask_required=False),
        observable_requirements=("scalar",),
        metadata={
            "equation": "u_t + cx*u_x + cy*u_y = nu*(u_xx+u_yy)",
            "analytical_reference": "single_fourier_mode_plus_mean",
        },
    )


def analytical_advection_diffusion_2d(
    x: Tensor,
    y: Tensor,
    t: Tensor,
    *,
    mean: Tensor,
    amplitude: Tensor,
    phase: Tensor,
    config: AdvectionDiffusion2DConfig,
) -> Tensor:
    """Evaluate the exact scalar field using broadcasting and physical units."""
    kx = 2.0 * torch.pi * config.mode_x / config.length_x
    ky = 2.0 * torch.pi * config.mode_y / config.length_y
    omega = config.cx * kx + config.cy * ky
    decay = config.nu * (kx * kx + ky * ky)
    return mean + amplitude * torch.exp(-decay * t) * torch.sin(kx * x + ky * y - omega * t + phase)


def generate_advection_diffusion_2d_trajectories(
    config: AdvectionDiffusion2DConfig,
    *,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> AdvectionDiffusion2DDataset:
    """Generate exact endpoint-free periodic trajectories on the CPU."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.arange(config.nx, dtype=dtype) * (config.length_x / config.nx)
    y = torch.arange(config.ny, dtype=dtype) * (config.length_y / config.ny)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    coordinates = torch.stack((xx, yy))
    cell_weights = torch.full(
        (config.nx, config.ny),
        (config.length_x / config.nx) * (config.length_y / config.ny),
        dtype=dtype,
    )
    records: list[TrajectoryRecord] = []
    for index in range(config.num_trajectories):
        if config.variable_dt:
            jitter = 2.0 * torch.rand(config.num_steps, generator=random, dtype=dtype) - 1.0
            dts = config.base_dt * (1.0 + config.dt_jitter * jitter)
        else:
            dts = torch.full((config.num_steps,), config.base_dt, dtype=dtype)
        times = torch.cat((torch.zeros(1, dtype=dtype), dts.cumsum(0)))
        amplitude = config.amplitude_min + (
            config.amplitude_max - config.amplitude_min
        ) * torch.rand((), generator=random, dtype=dtype)
        mean = config.mean_min + (config.mean_max - config.mean_min) * torch.rand(
            (), generator=random, dtype=dtype
        )
        phase = 2.0 * torch.pi * torch.rand((), generator=random, dtype=dtype)
        fields = torch.stack(
            [
                analytical_advection_diffusion_2d(
                    xx, yy, time, mean=mean, amplitude=amplitude, phase=phase, config=config
                )
                for time in times
            ]
        ).unsqueeze(1)
        records.append(
            TrajectoryRecord(
                trajectory_id=f"advection-diffusion-2d-{index:04d}",
                states_raw=fields,
                dts=dts,
                mu_static=torch.tensor([config.cx, config.cy, config.nu], dtype=dtype),
                coordinates=coordinates,
                cell_weights=cell_weights,
                metadata={
                    "phase": float(phase),
                    "amplitude": float(amplitude),
                    "mean": float(mean),
                    "seed": seed,
                },
            )
        )
    kx = 2.0 * torch.pi * config.mode_x / config.length_x
    ky = 2.0 * torch.pi * config.mode_y / config.length_y
    reference = AdvectionDiffusion2DReference(
        wave_number_x=float(kx),
        wave_number_y=float(ky),
        angular_frequency=float(config.cx * kx + config.cy * ky),
        decay_rate=float(config.nu * (kx * kx + ky * ky)),
        domain_area=config.length_x * config.length_y,
    )
    return AdvectionDiffusion2DDataset(
        TrajectoryDataset(records), make_advection_diffusion_2d_problem_spec(config), reference
    )
