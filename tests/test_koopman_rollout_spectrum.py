from __future__ import annotations

import math

import pytest
import torch

from jka_model.metrics import (
    dominant_oscillatory_mode,
    relative_frequency_error,
    spectral_growth_rate,
)
from jka_model.models import ContinuousKoopmanCore
from jka_model.rollout import koopman_rollout


def _core() -> ContinuousKoopmanCore:
    generator = torch.tensor([[0.0, 1.0], [-4.0, -0.4]], dtype=torch.float64)
    return ContinuousKoopmanCore(2, generator=generator, trainable=False, dtype=torch.float64)


def test_rollout_constant_dt() -> None:
    core = _core()
    state = torch.tensor([1.0, 0.0], dtype=torch.float64)
    rollout = core.rollout(state, 0.05, horizon=5)
    assert rollout.shape == (6, 2)
    torch.testing.assert_close(rollout[-1], core.step(state, 0.25), atol=1e-12, rtol=1e-12)


def test_rollout_variable_dt() -> None:
    core = _core()
    state = torch.tensor([1.0, 0.0], dtype=torch.float64)
    dts = torch.tensor([0.01, 0.03, 0.02, 0.05], dtype=torch.float64)
    rollout = core.rollout(state, dts)
    torch.testing.assert_close(
        rollout[-1], core.step(state, dts.sum()), atol=1e-11, rtol=1e-11
    )


def test_rollout_includes_initial_state() -> None:
    core = _core()
    state = torch.tensor([0.25, -0.75], dtype=torch.float64)
    rollout = koopman_rollout(core, state, torch.tensor([0.1, 0.2], dtype=torch.float64))
    torch.testing.assert_close(rollout[0], state)


def test_rollout_batch_shape() -> None:
    core = _core()
    states = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    shared = core.rollout(states, torch.tensor([0.1, 0.2, 0.1], dtype=torch.float64))
    schedules = torch.tensor([[0.1, 0.2, 0.1], [0.05, 0.1, 0.2]], dtype=torch.float64)
    per_sample = core.rollout(states, schedules)
    assert shared.shape == (2, 4, 2)
    assert per_sample.shape == (2, 4, 2)
    expected_shared = torch.stack(
        [core.rollout(state, torch.tensor([0.1, 0.2, 0.1])) for state in states]
    )
    expected_per_sample = torch.stack(
        [core.rollout(state, schedule) for state, schedule in zip(states, schedules, strict=True)]
    )
    torch.testing.assert_close(shared, expected_shared)
    torch.testing.assert_close(per_sample, expected_per_sample)


def test_rollout_rejects_invalid_schedule() -> None:
    core = _core()
    states = torch.zeros((2, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="first dimension"):
        core.rollout(states, torch.full((3, 4), 0.1, dtype=torch.float64))
    with pytest.raises(ValueError, match="horizon must match"):
        core.rollout(states, torch.full((4,), 0.1, dtype=torch.float64), horizon=3)
    with pytest.raises(ValueError, match="positive horizon"):
        core.rollout(states, 0.1, horizon=0)


def test_rollout_matches_repeated_step() -> None:
    core = _core()
    state = torch.tensor([1.0, 0.2], dtype=torch.float64)
    dts = torch.tensor([0.02, 0.05, 0.03], dtype=torch.float64)
    expected = [state]
    for dt in dts:
        expected.append(core.step(expected[-1], dt))
    torch.testing.assert_close(core.rollout(state, dts), torch.stack(expected))


def test_spectrum_matches_known_generator() -> None:
    gamma, omega0 = 0.2, 2.0
    core = _core()
    spectrum = core.spectrum()
    expected_omega = math.sqrt(omega0**2 - gamma**2)
    torch.testing.assert_close(
        spectrum.growth_rates,
        torch.full_like(spectrum.growth_rates, -gamma),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        spectrum.angular_frequencies,
        torch.full_like(spectrum.angular_frequencies, expected_omega),
        atol=1e-12,
        rtol=1e-12,
    )


def test_frequency_extraction() -> None:
    spectrum = _core().spectrum()
    growth, omega = dominant_oscillatory_mode(spectrum)
    assert abs(growth + 0.2) < 1e-12
    assert abs(omega - math.sqrt(4.0 - 0.04)) < 1e-12
    torch.testing.assert_close(
        spectrum.frequencies_hz,
        spectrum.angular_frequencies / (2.0 * math.pi),
    )
    assert relative_frequency_error(omega, math.sqrt(4.0 - 0.04)) < 1e-12
    assert spectral_growth_rate(spectrum) == pytest.approx(-0.2)


def test_frequency_metrics_reject_invalid_references() -> None:
    with pytest.raises(ValueError, match="reference"):
        relative_frequency_error(1.0, 0.0)
    with pytest.raises(ValueError, match="estimated"):
        relative_frequency_error(float("nan"), 1.0)


def test_spectrum_is_detached() -> None:
    trainable = ContinuousKoopmanCore(2, generator=_core().A, trainable=True, dtype=torch.float64)
    spectrum = trainable.spectrum()
    assert all(not tensor.requires_grad for tensor in spectrum.as_tuple())
    assert all(tensor.grad_fn is None for tensor in spectrum.as_tuple())
