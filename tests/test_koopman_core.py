from __future__ import annotations

import math

import pytest
import torch

from jka_model.models import ContinuousKoopmanCore


def _rotation_decay_generator(alpha: float, omega: float) -> torch.Tensor:
    return torch.tensor(
        [[-alpha, -omega], [omega, -alpha]],
        dtype=torch.float64,
    )


def test_matrix_exp_against_closed_form() -> None:
    alpha, omega, dt = 0.2, 1.7, 0.35
    generator = _rotation_decay_generator(alpha, omega)
    core = ContinuousKoopmanCore(2, generator=generator, trainable=False, dtype=torch.float64)
    angle = omega * dt
    expected = math.exp(-alpha * dt) * torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(core.transition_matrix(dt), expected, atol=1e-12, rtol=1e-12)


def test_zero_dt_is_identity() -> None:
    core = ContinuousKoopmanCore(
        2,
        generator=_rotation_decay_generator(0.2, 1.7),
        trainable=False,
        dtype=torch.float64,
    )
    state = torch.tensor([1.2, -0.4], dtype=torch.float64)
    torch.testing.assert_close(core.transition_matrix(0.0), torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(core.step(state, 0.0), state)


def test_semigroup_property() -> None:
    core = ContinuousKoopmanCore(
        2,
        generator=_rotation_decay_generator(0.1, 2.1),
        trainable=False,
        dtype=torch.float64,
    )
    dt1, dt2 = 0.07, 0.13
    combined = core.transition_matrix(dt1 + dt2)
    composed = core.transition_matrix(dt2) @ core.transition_matrix(dt1)
    torch.testing.assert_close(combined, composed, atol=1e-12, rtol=1e-12)


def test_step_single_state_shape() -> None:
    core = ContinuousKoopmanCore(2, dtype=torch.float32)
    output = core.step(torch.tensor([1.0, 0.0]), 0.1)
    assert output.shape == (2,)


def test_step_batch_shared_dt() -> None:
    generator = _rotation_decay_generator(0.1, 1.0)
    core = ContinuousKoopmanCore(2, generator=generator, trainable=False, dtype=torch.float64)
    states = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.25]], dtype=torch.float64)
    output = core.step(states, 0.2)
    expected = torch.stack([core.step(state, 0.2) for state in states])
    assert output.shape == (3, 2)
    torch.testing.assert_close(output, expected)


def test_step_batch_per_sample_dt() -> None:
    generator = _rotation_decay_generator(0.1, 1.0)
    core = ContinuousKoopmanCore(2, generator=generator, trainable=False, dtype=torch.float64)
    states = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.25]], dtype=torch.float64)
    dts = torch.tensor([0.05, 0.1, 0.2], dtype=torch.float64)
    output = core.step(states, dts)
    expected = torch.stack([core.step(state, dt) for state, dt in zip(states, dts, strict=True)])
    assert output.shape == (3, 2)
    torch.testing.assert_close(output, expected)


def test_reject_negative_dt() -> None:
    core = ContinuousKoopmanCore(2)
    with pytest.raises(ValueError, match="negative dt"):
        core.step(torch.zeros(2), -0.1)
    with pytest.raises(ValueError, match="negative dt"):
        core.step(torch.zeros(2, 2), torch.tensor([0.1, -0.1]))


def test_reject_nonfinite_dt_and_state() -> None:
    core = ContinuousKoopmanCore(2)
    with pytest.raises(ValueError, match="finite"):
        core.step(torch.zeros(2), float("nan"))
    with pytest.raises(ValueError, match="finite"):
        core.step(torch.zeros(2, 2), torch.tensor([0.1, float("inf")]))
    with pytest.raises(ValueError, match="finite"):
        core.step(torch.tensor([0.0, float("nan")]), 0.1)


def test_reject_wrong_dt_shape_or_batch_length() -> None:
    core = ContinuousKoopmanCore(2)
    with pytest.raises(ValueError, match="scalar or have shape"):
        core.step(torch.zeros(2, 2), torch.full((2, 1), 0.1))
    with pytest.raises(ValueError, match="single state requires scalar"):
        core.step(torch.zeros(2), torch.tensor([0.1, 0.2]))
    with pytest.raises(ValueError, match="batch dt length"):
        core.step(torch.zeros(2, 2), torch.tensor([0.1, 0.2, 0.3]))


def test_matrix_exp_gradient_wrt_A() -> None:
    core = ContinuousKoopmanCore(
        2,
        generator=torch.tensor([[0.0, 1.0], [-2.0, -0.2]]),
        trainable=True,
    )
    state = torch.tensor([[1.0, 0.0], [0.5, -0.5]])
    target = torch.tensor([[0.9, -0.2], [0.4, -0.6]])
    loss = (core.step(state, torch.tensor([0.1, 0.2])) - target).square().mean()
    loss.backward()
    assert core.A.grad is not None
    assert torch.isfinite(core.A.grad).all()
    assert torch.any(core.A.grad != 0)


def test_float32_float64_consistency() -> None:
    generator64 = _rotation_decay_generator(0.1, 1.3)
    core64 = ContinuousKoopmanCore(2, generator=generator64, trainable=False, dtype=torch.float64)
    core32 = ContinuousKoopmanCore(
        2, generator=generator64.float(), trainable=False, dtype=torch.float32
    )
    state64 = torch.tensor([0.5, -0.7], dtype=torch.float64)
    output64 = core64.step(state64, 0.15)
    output32 = core32.step(state64.float(), 0.15).double()
    torch.testing.assert_close(output32, output64, atol=2e-6, rtol=2e-6)


def test_device_transfer_cpu() -> None:
    core = ContinuousKoopmanCore(2).to("cpu")
    output = core.step(torch.tensor([1.0, 2.0]), 0.1)
    assert output.device.type == "cpu"


def test_matrix_exp_cpu_autocast_precision_island() -> None:
    core = ContinuousKoopmanCore(
        2,
        generator=_rotation_decay_generator(0.1, 1.3).float(),
        trainable=False,
        dtype=torch.float32,
    )
    reference = core.transition_matrix(0.2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        under_autocast = core.transition_matrix(0.2)
    assert under_autocast.dtype is torch.float32
    torch.testing.assert_close(under_autocast, reference)
