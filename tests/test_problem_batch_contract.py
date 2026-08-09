from __future__ import annotations

import pytest
import torch

from jka_model.contracts import ProblemBatch, validate_trajectory_alignment


def make_batch() -> ProblemBatch:
    raw = torch.arange(2 * 5 * 1 * 4, dtype=torch.float32).reshape(2, 5, 1, 4)
    model = raw / 10.0
    return ProblemBatch(
        context_states_raw=raw[:, :3],
        future_states_raw=raw[:, 3:],
        context_states_model=model[:, :3],
        future_states_model=model[:, 3:],
        history_actions=torch.zeros(2, 2, 1),
        future_actions=torch.zeros(2, 2, 1),
        history_dts=torch.full((2, 2), 0.1),
        future_dts=torch.full((2, 2), 0.1),
        mu_static=torch.ones(2, 1),
        valid_mask=torch.ones(2, 1, 4, dtype=torch.bool),
        trajectory_id=["a", "b"],
    )


def test_problem_batch_contract_uses_canonical_names_only() -> None:
    batch = make_batch()
    assert batch.context_states_raw.shape == (2, 3, 1, 4)
    assert batch.future_states_raw.shape == (2, 2, 1, 4)
    assert batch.history_actions is not None and batch.history_actions.shape == (2, 2, 1)
    assert batch.future_dts.shape == (2, 2)
    assert not hasattr(batch, "parameters")
    assert not hasattr(batch, "mask")
    assert not hasattr(batch, "states_raw")


def test_problem_batch_rejects_misaligned_dts() -> None:
    batch = make_batch()
    with pytest.raises(ValueError, match="future_dts"):
        ProblemBatch(
            context_states_raw=batch.context_states_raw,
            future_states_raw=batch.future_states_raw,
            context_states_model=batch.context_states_model,
            future_states_model=batch.future_states_model,
            history_dts=batch.history_dts,
            future_dts=torch.ones(2, 1),
        )


def test_problem_batch_rejects_one_sided_actions() -> None:
    batch = make_batch()
    with pytest.raises(ValueError, match="both be present"):
        ProblemBatch(
            context_states_raw=batch.context_states_raw,
            future_states_raw=batch.future_states_raw,
            context_states_model=batch.context_states_model,
            future_states_model=batch.future_states_model,
            history_dts=batch.history_dts,
            future_dts=batch.future_dts,
            history_actions=batch.history_actions,
        )


def test_trajectory_alignment_requires_t_plus_one_states() -> None:
    validate_trajectory_alignment(torch.zeros(5, 2), torch.zeros(4, 1), torch.ones(4))
    with pytest.raises(ValueError, match="actions length"):
        validate_trajectory_alignment(torch.zeros(5, 2), torch.zeros(3, 1), torch.ones(4))
    with pytest.raises(ValueError, match="dts length"):
        validate_trajectory_alignment(torch.zeros(5, 2), torch.zeros(4, 1), torch.ones(3))


def test_problem_batch_to_returns_new_batch_without_unit_conversion() -> None:
    batch = make_batch()
    moved = batch.to(dtype=torch.float64)
    assert moved is not batch
    assert moved.context_states_raw.dtype is torch.float64
    assert moved.trajectory_id == batch.trajectory_id
    assert torch.equal(moved.context_states_raw, batch.context_states_raw.to(torch.float64))
