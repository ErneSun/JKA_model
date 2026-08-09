from __future__ import annotations

import torch

from jka_model.config import (
    KnownLatentConfig,
    KoopmanAutoencoderConfig,
    RepresentationLossConfig,
    RepresentationTrainingConfig,
    SplitConfig,
)
from jka_model.data import (
    ChannelStandardizer,
    TrajectoryWindowDataset,
    generate_known_latent_trajectories,
    make_split_manifest,
    select_split,
)
from jka_model.evaluation import evaluate_learned_trajectory
from jka_model.metrics import latent_diagnostics
from jka_model.training.koopman_representation import (
    initialize_koopman_autoencoder,
    train_koopman_representation,
)


def test_learned_lifting_on_known_latent_system() -> None:
    data = generate_known_latent_trajectories(
        KnownLatentConfig(
            alpha=0.05,
            omega=1.2,
            base_dt=0.1,
            num_steps=30,
            num_trajectories=8,
        ),
        seed=4,
    )
    assert all(not latent.requires_grad for latent in data.true_latents.values())
    manifest = make_split_manifest(
        data.records,
        SplitConfig(train=0.625, validation=0.125, test=0.25, seed=3),
    )
    normalizer = ChannelStandardizer().fit(
        data.records, manifest, data.problem_spec
    )
    assert not hasattr(normalizer, "parameters")
    windows = TrajectoryWindowDataset(
        select_split(data.records, manifest, "train"),
        history=2,
        horizon=4,
        normalizer=normalizer,
    )
    model = initialize_koopman_autoencoder(
        KoopmanAutoencoderConfig(
            observation_dim=5,
            latent_dim=2,
            hidden_dim=24,
            encoder_hidden_layers=0,
        ),
        seed=4,
        init_scale=0.05,
        dtype=torch.float64,
    )
    training = train_koopman_representation(
        model,
        windows,
        RepresentationLossConfig(
            lambda_k=10.0,
            lambda_multi=50.0,
            lambda_rec=10.0,
            lambda_var=10.0,
            lambda_spec=0.01,
            min_std=0.2,
        ),
        RepresentationTrainingConfig(
            epochs=300,
            batch_size=256,
            learning_rate=0.003,
            init_scale=0.05,
        ),
        seed=4,
    )
    test_record = select_split(data.records, manifest, "test")[0]
    test_state = normalizer.transform(test_record.states_raw)
    diagnostics = latent_diagnostics(model.encode(test_state))
    _, decoded = model.rollout_decoded(
        test_state[0].unsqueeze(0), test_record.dts[:10]
    )
    heldout, _, _, _ = evaluate_learned_trajectory(
        model, test_record, normalizer, horizon=10
    )
    assert training.final_losses["reconstruction_loss"] < training.initial_losses[
        "reconstruction_loss"
    ]
    assert training.diagnostic_history[0].epoch == 0
    assert training.diagnostic_history[-1].epoch == training.epochs
    assert len(training.diagnostic_history) == 4
    assert all(
        snapshot.minimum_latent_std <= snapshot.maximum_latent_std
        for snapshot in training.diagnostic_history
    )
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert diagnostics.minimum_std > 1e-3
    assert torch.isfinite(decoded).all()
    assert heldout.finite
    assert heldout.one_step_latent_mse >= 0
    assert heldout.multi_step_latent_mse >= 0
    assert heldout.decoded_model_mse >= 0
    assert heldout.decoded_raw_mse >= 0
