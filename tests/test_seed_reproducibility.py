from __future__ import annotations

import random

import numpy as np
import torch

from jka_model.utils import capture_rng_state, restore_rng_state, set_global_seed


def _draw() -> tuple[float, np.ndarray, torch.Tensor]:
    return random.random(), np.random.rand(3), torch.rand(3)


def test_seed_reproducibility() -> None:
    set_global_seed(123)
    first = _draw()
    set_global_seed(123)
    second = _draw()
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


def test_rng_capture_restore() -> None:
    set_global_seed(321)
    state = capture_rng_state()
    expected = _draw()
    _draw()
    restore_rng_state(state)
    actual = _draw()
    assert expected[0] == actual[0]
    np.testing.assert_array_equal(expected[1], actual[1])
    torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

