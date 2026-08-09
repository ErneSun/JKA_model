from __future__ import annotations

import torch

from jka_model.config import KnownLatentConfig, SplitConfig
from jka_model.data import (
    ChannelStandardizer,
    TrajectoryWindowDataset,
    generate_known_latent_trajectories,
    make_split_manifest,
    nonlinear_observation_map,
    rotation_decay_transition,
    select_split,
    validate_trajectories_against_spec,
)


def _config() -> KnownLatentConfig:
    return KnownLatentConfig(
        alpha=0.05,
        omega=1.2,
        base_dt=0.1,
        variable_dt=True,
        dt_jitter=0.2,
        num_steps=12,
        num_trajectories=6,
    )


def test_nonlinear_observation_generator_shapes() -> None:
    dataset = generate_known_latent_trajectories(_config(), seed=4)
    validate_trajectories_against_spec(dataset.records, dataset.problem_spec)
    assert len(dataset.records) == 6
    assert dataset.records[0].states_raw.shape == (13, 5)
    assert dataset.records[0].dts.shape == (12,)
    assert dataset.latent(dataset.records[0].trajectory_id).shape == (13, 2)


def test_hidden_latent_obeys_true_linear_dynamics() -> None:
    config = _config()
    dataset = generate_known_latent_trajectories(config, seed=5)
    record = dataset.records[0]
    hidden = dataset.latent(record.trajectory_id)
    expected = rotation_decay_transition(
        config.alpha, config.omega, float(record.dts[0])
    ) @ hidden[0]
    torch.testing.assert_close(hidden[1], expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(record.states_raw, nonlinear_observation_map(hidden))


def test_observation_map_deterministic() -> None:
    first = generate_known_latent_trajectories(_config(), seed=7)
    second = generate_known_latent_trajectories(_config(), seed=7)
    for left, right in zip(first.records, second.records, strict=True):
        torch.testing.assert_close(left.states_raw, right.states_raw)
        torch.testing.assert_close(left.dts, right.dts)
        torch.testing.assert_close(
            first.latent(left.trajectory_id), second.latent(right.trajectory_id)
        )


def test_true_latent_not_in_model_input() -> None:
    dataset = generate_known_latent_trajectories(_config(), seed=8)
    manifest = make_split_manifest(dataset.records, config=_split_config())
    normalizer = ChannelStandardizer().fit(
        dataset.records, manifest, dataset.problem_spec
    )
    windows = TrajectoryWindowDataset(
        select_split(dataset.records, manifest, "train"),
        history=2,
        horizon=2,
        normalizer=normalizer,
    )
    batch = windows[0]
    assert not hasattr(batch, "true_latent_s")
    assert batch.context_states_model.shape[-1] == 5
    assert batch.mu_static is not None and batch.mu_static.shape[-1] == 2


def test_linear_observation_sanity_preserves_hidden_state() -> None:
    dataset = generate_known_latent_trajectories(
        _config(), seed=9, nonlinear_observation=False
    )
    record = dataset.records[0]
    assert record.states_raw.shape == (13, 2)
    torch.testing.assert_close(record.states_raw, dataset.latent(record.trajectory_id))


def _split_config() -> SplitConfig:
    return SplitConfig(train=0.5, validation=0.25, test=0.25, seed=3)
