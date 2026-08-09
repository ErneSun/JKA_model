from __future__ import annotations

import math

import torch

from jka_model.physics import periodic_first_derivative, periodic_second_derivative


def _periodic_sine(nx: int, wave_number: float) -> tuple[torch.Tensor, float]:
    x = torch.linspace(0.0, 2.0 * math.pi, nx, dtype=torch.float64)
    state = torch.sin(wave_number * x)[None, None, :]
    return state, float(x[1] - x[0])


def test_first_derivative_against_analytic() -> None:
    wave_number = 3.0
    state, dx = _periodic_sine(513, wave_number)
    x = torch.linspace(0.0, 2.0 * math.pi, 513, dtype=torch.float64)
    numerical = periodic_first_derivative(state, dx)[0, 0]
    analytic = wave_number * torch.cos(wave_number * x)
    error = (numerical - analytic).abs().max()
    assert error < 1e-3


def test_second_derivative_against_analytic() -> None:
    wave_number = 3.0
    state, dx = _periodic_sine(513, wave_number)
    x = torch.linspace(0.0, 2.0 * math.pi, 513, dtype=torch.float64)
    numerical = periodic_second_derivative(state, dx)[0, 0]
    analytic = -(wave_number**2) * torch.sin(wave_number * x)
    error = (numerical - analytic).abs().max()
    assert error < 2e-3


def test_second_order_grid_convergence() -> None:
    wave_number = 2.0
    errors: dict[int, float] = {}
    for nx in (32, 64, 128):
        state, dx = _periodic_sine(nx, wave_number)
        x = torch.linspace(0.0, 2.0 * math.pi, nx, dtype=torch.float64)
        numerical = periodic_first_derivative(state, dx)[0, 0, :-1]
        analytic = wave_number * torch.cos(wave_number * x[:-1])
        errors[nx] = float((numerical - analytic).square().mean().sqrt())

    assert errors[128] < errors[64] < errors[32]
    order_32_to_64 = math.log(errors[32] / errors[64]) / math.log(2.0)
    order_64_to_128 = math.log(errors[64] / errors[128]) / math.log(2.0)
    print(
        "grid convergence diagnostics: "
        f"E32={errors[32]:.8e}, E64={errors[64]:.8e}, E128={errors[128]:.8e}, "
        f"p32->64={order_32_to_64:.6f}, p64->128={order_64_to_128:.6f}"
    )
    assert 1.7 < order_32_to_64 < 2.3, f"observed order={order_32_to_64:.4f}"
    assert 1.7 < order_64_to_128 < 2.3, f"observed order={order_64_to_128:.4f}"
