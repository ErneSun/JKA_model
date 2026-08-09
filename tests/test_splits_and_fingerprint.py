from __future__ import annotations

from dataclasses import replace

from jka_model.config import SplitConfig, ToyAdvectionDiffusionConfig
from jka_model.data import (
    ChannelStandardizer,
    TrajectoryWindowDataset,
    data_fingerprint,
    generate_advection_diffusion_trajectories,
    make_split_manifest,
    select_split,
)


def test_split_is_deterministic_disjoint_and_order_independent(tmp_path) -> None:
    records, _ = generate_advection_diffusion_trajectories(
        ToyAdvectionDiffusionConfig(num_trajectories=12), seed=3
    )
    config = SplitConfig(seed=19)
    first = make_split_manifest(list(records), config)
    second = make_split_manifest(list(reversed(records)), config)
    assert first == second
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.validation).isdisjoint(first.test)
    destination = tmp_path / "split.json"
    first.save(destination)
    assert type(first).load(destination) == first


def test_fingerprint_is_stable_and_content_sensitive() -> None:
    records, spec = generate_advection_diffusion_trajectories(
        ToyAdvectionDiffusionConfig(num_trajectories=4), seed=5
    )
    original = data_fingerprint(list(records), spec)
    assert original == data_fingerprint(list(reversed(records)), spec)
    modified = list(records)
    changed_state = modified[0].states_raw.clone()
    changed_state[0, 0, 1] += 1e-3
    modified[0] = replace(modified[0], states_raw=changed_state)
    assert data_fingerprint(modified, spec) != original
    metadata_modified = list(records)
    metadata_modified[0] = replace(
        metadata_modified[0], metadata={"seed": 5, "analytic": True, "revision": 2}
    )
    assert data_fingerprint(metadata_modified, spec) != original
    changed_spec = replace(spec, metadata={"equation": "changed-for-fingerprint-test"})
    assert data_fingerprint(records, changed_spec) != original
    assert original.startswith("sha256:") and len(original) == 71


def test_split_by_trajectory_precedes_window_generation() -> None:
    records, spec = generate_advection_diffusion_trajectories(
        ToyAdvectionDiffusionConfig(num_trajectories=9, num_steps=8), seed=17
    )
    manifest = make_split_manifest(records, SplitConfig(seed=4))
    normalizer = ChannelStandardizer().fit(records, manifest, spec)
    for split_name in ("train", "validation", "test"):
        split_records = select_split(records, manifest, split_name)
        windows = TrajectoryWindowDataset(
            split_records, history=3, horizon=2, normalizer=normalizer
        )
        expected_ids = set(getattr(manifest, split_name))
        observed_ids = {
            str(window.trajectory_id[0])
            for window in windows
            if isinstance(window.trajectory_id, list)
        }
        assert observed_ids == expected_ids
