from __future__ import annotations

import torch

from jka_model.config import ToyAdvectionDiffusionConfig
from jka_model.data import generate_advection_diffusion_trajectories
from jka_model.physics import (
    periodic_first_derivative,
    periodic_second_derivative,
    weighted_integral,
)


def test_analytic_toy_is_deterministic_periodic_and_mass_conserving() -> None:
    config = ToyAdvectionDiffusionConfig(num_trajectories=3, variable_dt=True)
    first, spec = generate_advection_diffusion_trajectories(
        config, seed=9, dtype=torch.float64
    )
    second, _ = generate_advection_diffusion_trajectories(
        config, seed=9, dtype=torch.float64
    )
    torch.testing.assert_close(first[0].states_raw, second[0].states_raw)
    assert spec.dt_mode.value == "variable"
    for record in first:
        torch.testing.assert_close(record.states_raw[..., 0], record.states_raw[..., -1])
        assert record.cell_weights is not None
        mass = weighted_integral(record.states_raw, record.cell_weights)
        torch.testing.assert_close(mass, mass[:1].expand_as(mass), atol=1e-12, rtol=1e-12)


def test_toy_pipeline_defaults_to_float32() -> None:
    records, _ = generate_advection_diffusion_trajectories(
        ToyAdvectionDiffusionConfig(num_trajectories=1), seed=9
    )
    assert records[0].states_raw.dtype is torch.float32
    assert records[0].dts.dtype is torch.float32


def test_periodic_finite_difference_operators_are_accurate() -> None:
    nx = 129
    x = torch.linspace(0.0, 2.0 * torch.pi, nx, dtype=torch.float64)
    state = torch.sin(2.0 * x)[None, None, :]
    dx = float(x[1] - x[0])
    first = periodic_first_derivative(state, dx)
    second = periodic_second_derivative(state, dx)
    assert (first - 2.0 * torch.cos(2.0 * x)).abs().max() < 0.01
    assert (second + 4.0 * torch.sin(2.0 * x)).abs().max() < 0.02
