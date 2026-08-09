from __future__ import annotations

import torch

from jka_model.config import load_config
from jka_model.data import ChannelStandardizer
from jka_model.training.koopman_representation import initialize_koopman_autoencoder
from jka_model.utils import Checkpoint, capture_rng_state, load_checkpoint, save_checkpoint


def test_v0_4_checkpoint_roundtrip(tmp_path) -> None:
    config = load_config("configs/v0_4_smoke.yaml")
    assert config.autoencoder is not None
    assert config.representation_training is not None
    model = initialize_koopman_autoencoder(
        config.autoencoder,
        seed=config.training.seed,
        init_scale=config.representation_training.init_scale,
        dtype=torch.float64,
    )
    normalizer = ChannelStandardizer()
    normalizer.load_state_dict(
        {
            "kind": "channel_standardizer",
            "eps": 1e-6,
            "mean": torch.zeros(5, dtype=torch.float64),
            "scale": torch.ones(5, dtype=torch.float64),
            "spatial_dim": 0,
            "layout": "channels_first",
            "fitted_trajectory_ids": ["train-0"],
        }
    )
    state = torch.randn((4, 5), generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    dts = torch.tensor([0.07, 0.11, 0.09], dtype=torch.float64)
    encoded_before = model.encode(state)
    stepped_before = model.core.step(encoded_before[:3], dts)
    decoded_before = model.decode(encoded_before)
    predicted_before = model.step(state[:3], dts)
    spectrum_before = model.core.spectrum()
    destination = tmp_path / "v0_4.pt"
    save_checkpoint(
        Checkpoint(
            train_stage=config.training.stage,
            epoch=3,
            global_step=7,
            online_model_state=model.state_dict(),
            optimizer_state={"state": {}, "param_groups": []},
            rng_state=capture_rng_state(),
            normalizer_state=normalizer.state_dict(),
            config=config,
            data_fingerprint="sha256:v0.4-test",
            split_manifest={"train": ["train-0"], "validation": [], "test": []},
            physics_constraint_spec=[],
        ),
        destination,
    )
    restored_checkpoint = load_checkpoint(destination)
    restored = initialize_koopman_autoencoder(
        config.autoencoder,
        seed=99,
        init_scale=config.representation_training.init_scale,
        dtype=torch.float64,
    )
    assert restored_checkpoint.online_model_state is not None
    restored.load_state_dict(restored_checkpoint.online_model_state)
    restored_normalizer = ChannelStandardizer()
    assert restored_checkpoint.normalizer_state is not None
    restored_normalizer.load_state_dict(restored_checkpoint.normalizer_state)
    torch.testing.assert_close(restored.encode(state), encoded_before)
    torch.testing.assert_close(restored.core.step(encoded_before[:3], dts), stepped_before)
    torch.testing.assert_close(restored.decode(encoded_before), decoded_before)
    torch.testing.assert_close(restored.step(state[:3], dts), predicted_before)
    torch.testing.assert_close(restored_normalizer.transform(state), state)
    torch.testing.assert_close(
        restored.core.spectrum().growth_rates.sort().values,
        spectrum_before.growth_rates.sort().values,
    )
    torch.testing.assert_close(
        restored.core.spectrum().angular_frequencies.sort().values,
        spectrum_before.angular_frequencies.sort().values,
    )
    assert restored_checkpoint.optimizer_state is not None
    assert restored_checkpoint.rng_state is not None
    assert restored_checkpoint.config == config
