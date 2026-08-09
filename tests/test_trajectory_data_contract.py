from __future__ import annotations

import pytest
import torch

from jka_model.data import TrajectoryRecord


def test_trajectory_record_enforces_t_plus_one_alignment() -> None:
    record = TrajectoryRecord(
        trajectory_id="a",
        states_raw=torch.zeros(5, 1, 8),
        actions=torch.zeros(4, 2),
        dts=torch.full((4,), 0.1),
    )
    assert record.num_steps == 4
    with pytest.raises(ValueError, match="dts length"):
        TrajectoryRecord(
            trajectory_id="bad",
            states_raw=torch.zeros(5, 1, 8),
            dts=torch.full((3,), 0.1),
        )


def test_trajectory_record_rejects_nonpositive_dt() -> None:
    with pytest.raises(ValueError, match="positive"):
        TrajectoryRecord(
            trajectory_id="bad-dt",
            states_raw=torch.zeros(3, 1, 8),
            dts=torch.tensor([0.1, 0.0]),
        )


def test_reject_negative_dt() -> None:
    with pytest.raises(ValueError, match="positive"):
        TrajectoryRecord(
            trajectory_id="negative-dt",
            states_raw=torch.zeros(3, 1),
            dts=torch.tensor([0.1, -0.1]),
        )


def test_reject_zero_dt() -> None:
    with pytest.raises(ValueError, match="positive"):
        TrajectoryRecord(
            trajectory_id="zero-dt",
            states_raw=torch.zeros(3, 1),
            dts=torch.tensor([0.1, 0.0]),
        )


def test_reject_wrong_dt_length() -> None:
    with pytest.raises(ValueError, match="dts length"):
        TrajectoryRecord(
            trajectory_id="wrong-dt-length",
            states_raw=torch.zeros(5, 1),
            dts=torch.ones(3),
        )


def test_reject_wrong_action_length() -> None:
    with pytest.raises(ValueError, match="actions length"):
        TrajectoryRecord(
            trajectory_id="wrong-action-length",
            states_raw=torch.zeros(5, 1),
            actions=torch.zeros(3, 1),
            dts=torch.ones(4),
        )
