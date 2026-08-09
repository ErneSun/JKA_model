from __future__ import annotations

import torch

from jka_model.config import DampedOscillatorConfig, DuffingConfig
from jka_model.data import (
    damped_oscillator_analytic_state,
    damped_oscillator_analytic_transition,
    damped_oscillator_generator_matrix,
    generate_damped_oscillator_trajectories,
    generate_duffing_trajectories,
    rotation_decay_transition,
    trajectory_transition_tensors,
    validate_trajectories_against_spec,
)
from jka_model.models import ContinuousKoopmanCore


def test_damped_oscillator_reference() -> None:
    config = DampedOscillatorConfig(
        omega0=2.0,
        gamma=0.15,
        base_dt=0.04,
        variable_dt=True,
        dt_jitter=0.2,
        num_steps=20,
        num_trajectories=3,
    )
    records, spec = generate_damped_oscillator_trajectories(
        config, seed=9, dtype=torch.float64
    )
    validate_trajectories_against_spec(records, spec)
    core = ContinuousKoopmanCore(
        2,
        generator=damped_oscillator_generator_matrix(config.omega0, config.gamma),
        trainable=False,
        dtype=torch.float64,
    )
    record = records[0]
    predicted = core.rollout(record.states_raw[0], record.dts)
    torch.testing.assert_close(predicted, record.states_raw, atol=2e-11, rtol=2e-11)


def test_damped_analytic_formula_at_zero_and_positive_time() -> None:
    initial = torch.tensor([0.7, -0.3], dtype=torch.float64)
    times = torch.tensor([0.0, 0.1, 0.35], dtype=torch.float64)
    states = damped_oscillator_analytic_state(
        initial, times, omega0=1.8, gamma=0.12
    )
    torch.testing.assert_close(states[0], initial)
    assert states.shape == (3, 2)
    assert torch.isfinite(states).all()


def test_physical_oscillator_transition_matches_analytic_closed_form() -> None:
    omega0, gamma, dt = 2.0, 0.15, 0.04
    core = ContinuousKoopmanCore(
        2,
        generator=damped_oscillator_generator_matrix(omega0, gamma),
        trainable=False,
        dtype=torch.float64,
    )
    expected = damped_oscillator_analytic_transition(omega0, gamma, dt)
    torch.testing.assert_close(
        core.transition_matrix(dt), expected, atol=1e-12, rtol=1e-12
    )


def test_rotation_decay_reference_is_independent_closed_form() -> None:
    transition = rotation_decay_transition(0.2, 1.5, 0.3)
    assert transition.shape == (2, 2)
    assert torch.isfinite(transition).all()


def test_duffing_reference_and_transition_pairs() -> None:
    config = DuffingConfig(num_steps=30, num_trajectories=3, rk4_substeps=2)
    records, spec = generate_duffing_trajectories(config, seed=5, dtype=torch.float64)
    validate_trajectories_against_spec(records, spec)
    states, targets, dts = trajectory_transition_tensors(records)
    assert states.shape == targets.shape == (90, 2)
    assert dts.shape == (90,)
    assert torch.isfinite(states).all() and torch.isfinite(targets).all()
    assert torch.all(dts > 0)
