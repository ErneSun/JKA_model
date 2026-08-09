from __future__ import annotations

import torch

from jka_model.config import ToyAdvectionDiffusionConfig
from jka_model.data import (
    ChannelStandardizer,
    SplitManifest,
    TrajectoryRecord,
    generate_advection_diffusion_trajectories,
)


def test_normalizer_fits_train_ids_only_and_roundtrips() -> None:
    _, spec = generate_advection_diffusion_trajectories(
        ToyAdvectionDiffusionConfig(num_trajectories=1), seed=1
    )
    train = TrajectoryRecord(
        "train", torch.tensor([[[0.0, 2.0]], [[2.0, 4.0]]]), torch.ones(1)
    )
    validation = TrajectoryRecord(
        "validation", torch.full((2, 1, 2), 1e6), torch.ones(1)
    )
    manifest = SplitManifest(
        train=("train",), validation=("validation",), test=(), seed=0, ratios=(0.5, 0.5, 0.0)
    )
    normalizer = ChannelStandardizer().fit([train, validation], manifest, spec)
    assert normalizer.fitted_trajectory_ids == ("train",)
    torch.testing.assert_close(normalizer.mean, torch.tensor([2.0], dtype=torch.float64))
    raw_before = train.states_raw.clone()
    transformed = normalizer.transform(train.states_raw)
    torch.testing.assert_close(normalizer.inverse_transform(transformed), train.states_raw)
    torch.testing.assert_close(train.states_raw, raw_before)
    assert transformed.data_ptr() != train.states_raw.data_ptr()


def test_normalizer_state_roundtrip() -> None:
    records, spec = generate_advection_diffusion_trajectories(
        ToyAdvectionDiffusionConfig(num_trajectories=3), seed=2
    )
    manifest = SplitManifest(
        train=(records[0].trajectory_id,),
        validation=(records[1].trajectory_id,),
        test=(records[2].trajectory_id,),
        seed=0,
        ratios=(1 / 3, 1 / 3, 1 / 3),
    )
    fitted = ChannelStandardizer().fit(records, manifest, spec)
    restored = ChannelStandardizer()
    restored.load_state_dict(fitted.state_dict())
    torch.testing.assert_close(
        restored.transform(records[2].states_raw), fitted.transform(records[2].states_raw)
    )
