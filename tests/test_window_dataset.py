from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from jka_model.config import ToyAdvectionDiffusionConfig
from jka_model.data import (
    ChannelStandardizer,
    SplitManifest,
    TrajectoryRecord,
    TrajectoryWindowDataset,
    collate_problem_batches,
    generate_advection_diffusion_trajectories,
)


def test_window_alignment_is_exact() -> None:
    _, spec = generate_advection_diffusion_trajectories(
        ToyAdvectionDiffusionConfig(num_trajectories=1), seed=4
    )
    states = torch.arange(7, dtype=torch.float64).reshape(7, 1, 1)
    actions = torch.arange(10, 16, dtype=torch.float64).reshape(6, 1)
    record = TrajectoryRecord(
        "one", states, torch.arange(1, 7, dtype=torch.float64), actions=actions
    )
    manifest = SplitManifest(
        train=("one",), validation=(), test=(), seed=0, ratios=(1.0, 0.0, 0.0)
    )
    normalizer = ChannelStandardizer().fit([record], manifest, spec)
    dataset = TrajectoryWindowDataset(
        [record], history=3, horizon=2, normalizer=normalizer
    )
    first = dataset[0]
    assert len(dataset) == 3
    assert first.context_states_raw[:, :, 0, 0].tolist() == [[0.0, 1.0, 2.0]]
    assert first.future_states_raw[:, :, 0, 0].tolist() == [[3.0, 4.0]]
    assert first.history_dts.tolist() == [[1.0, 2.0]]
    assert first.future_dts.tolist() == [[3.0, 4.0]]
    assert first.history_actions is not None
    assert first.future_actions is not None
    assert first.history_actions[:, :, 0].tolist() == [[10.0, 11.0]]
    assert first.future_actions[:, :, 0].tolist() == [[12.0, 13.0]]


def test_variable_dt_alignment_at_later_window() -> None:
    _, spec = generate_advection_diffusion_trajectories(
        ToyAdvectionDiffusionConfig(num_trajectories=1), seed=6
    )
    states = torch.arange(8, dtype=torch.float64).reshape(8, 1, 1)
    dts = torch.tensor([0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64])
    record = TrajectoryRecord("variable", states, dts)
    manifest = SplitManifest(
        train=("variable",), validation=(), test=(), seed=0, ratios=(1.0, 0.0, 0.0)
    )
    normalizer = ChannelStandardizer().fit([record], manifest, spec)
    dataset = TrajectoryWindowDataset(
        [record], history=3, horizon=2, normalizer=normalizer
    )
    second = dataset[1]
    assert second.context_states_raw[:, :, 0, 0].tolist() == [[1.0, 2.0, 3.0]]
    torch.testing.assert_close(second.history_dts, torch.tensor([[0.02, 0.04]]))
    torch.testing.assert_close(second.future_dts, torch.tensor([[0.08, 0.16]]))


def test_window_dataset_rejects_trajectory_too_short_for_requested_window() -> None:
    _, spec = generate_advection_diffusion_trajectories(
        ToyAdvectionDiffusionConfig(num_trajectories=1), seed=10
    )
    record = TrajectoryRecord(
        "short", torch.zeros(4, 1, 2), torch.ones(3, dtype=torch.float32)
    )
    manifest = SplitManifest(
        train=("short",), validation=(), test=(), seed=0, ratios=(1.0, 0.0, 0.0)
    )
    normalizer = ChannelStandardizer().fit([record], manifest, spec)
    with pytest.raises(ValueError, match="too short"):
        TrajectoryWindowDataset([record], history=3, horizon=2, normalizer=normalizer)


def test_windows_never_cross_trajectory_and_collate_canonical_fields() -> None:
    records, spec = generate_advection_diffusion_trajectories(
        ToyAdvectionDiffusionConfig(num_trajectories=3, num_steps=8), seed=8
    )
    ids = tuple(record.trajectory_id for record in records)
    manifest = SplitManifest(train=ids, validation=(), test=(), seed=0, ratios=(1.0, 0.0, 0.0))
    normalizer = ChannelStandardizer().fit(records, manifest, spec)
    dataset = TrajectoryWindowDataset(records, history=3, horizon=2, normalizer=normalizer)
    for item in dataset:
        assert isinstance(item.trajectory_id, list) and len(item.trajectory_id) == 1
    batch = next(
        iter(DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_problem_batches))
    )
    assert batch.context_states_raw.shape == (4, 3, 1, 65)
    assert batch.future_states_model.shape == (4, 2, 1, 65)
    assert len(batch.trajectory_id) == 4
