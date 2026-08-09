"""Secondary V0.4 learned-lifting diagnostic on the non-closed Duffing system."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from jka_model.config import DirectIdentificationConfig, ProjectConfig
from jka_model.data import (
    ChannelStandardizer,
    TrajectoryDataset,
    TrajectoryWindowDataset,
    generate_duffing_trajectories,
    make_split_manifest,
    select_split,
    trajectory_transition_tensors,
)
from jka_model.evaluation.dynamics import evaluate_rollout
from jka_model.evaluation.representation import evaluate_learned_trajectory
from jka_model.training import initialize_direct_koopman, train_direct_koopman
from jka_model.training.koopman_representation import (
    initialize_koopman_autoencoder,
    train_koopman_representation,
)
from jka_model.utils import set_global_seed


@dataclass(frozen=True, slots=True)
class DuffingLiftingDiagnostic:
    direct_state_rollout_mse: float
    learned_lifting_rollout_mse: float
    direct_final_one_step_mse: float
    learned_final_loss: float
    finite: bool


def run_duffing_lifting_diagnostic(config: ProjectConfig) -> DuffingLiftingDiagnostic:
    """Compare V0.3 direct state with one small learned finite-dimensional lifting."""
    required = (
        config.duffing,
        config.autoencoder,
        config.representation_loss,
        config.representation_training,
        config.representation_evaluation,
        config.koopman,
    )
    if any(section is None for section in required):
        raise ValueError("Duffing lifting diagnostic requires V0.4 and Duffing sections")
    assert config.duffing is not None
    assert config.autoencoder is not None
    assert config.representation_loss is not None
    assert config.representation_training is not None
    assert config.representation_evaluation is not None
    assert config.koopman is not None
    set_global_seed(config.training.seed + 101, deterministic=config.training.deterministic)
    dtype = torch.float64 if config.koopman.dtype == "float64" else torch.float32
    records, spec = generate_duffing_trajectories(
        config.duffing, seed=config.training.seed + 101, dtype=dtype
    )
    manifest = make_split_manifest(records, config.data.split)
    train_records = select_split(records, manifest, "train")
    test_records = select_split(records, manifest, "test")
    if not test_records:
        test_records = select_split(records, manifest, "validation")
    direct_states, direct_targets, direct_dts = trajectory_transition_tensors(
        TrajectoryDataset(train_records)
    )
    direct_config = DirectIdentificationConfig(
        epochs=config.representation_training.duffing_epochs,
        learning_rate=config.representation_training.learning_rate,
        init_scale=config.representation_training.init_scale,
        weight_decay=config.representation_training.weight_decay,
    )
    direct = initialize_direct_koopman(
        2,
        seed=config.training.seed + 101,
        init_scale=direct_config.init_scale,
        dtype=dtype,
    )
    direct_fit = train_direct_koopman(
        direct, direct_states, direct_targets, direct_dts, direct_config
    )
    normalizer = ChannelStandardizer(eps=config.data.normalization.eps).fit(
        records, manifest, spec
    )
    windows = TrajectoryWindowDataset(
        train_records,
        history=config.data.history,
        horizon=config.data.horizon,
        normalizer=normalizer,
    )
    lifted_architecture = replace(
        config.autoencoder,
        observation_dim=2,
        latent_dim=4,
        encoder_hidden_layers=1,
    )
    lifted = initialize_koopman_autoencoder(
        lifted_architecture,
        seed=config.training.seed + 102,
        init_scale=config.representation_training.init_scale,
        dtype=dtype,
    )
    lifted_fit = train_koopman_representation(
        lifted,
        windows,
        config.representation_loss,
        config.representation_training,
        seed=config.training.seed + 102,
        epochs=config.representation_training.duffing_epochs,
    )
    horizon = min(
        config.representation_evaluation.rollout_horizon,
        min(record.num_steps for record in test_records),
    )
    direct_errors: list[float] = []
    lifted_errors: list[float] = []
    finite = True
    for record in test_records:
        direct_prediction = direct.rollout(record.states_raw[0], record.dts[:horizon])
        direct_metrics = evaluate_rollout(
            direct_prediction, record.states_raw[: horizon + 1]
        )
        lifted_metrics, _, _, _ = evaluate_learned_trajectory(
            lifted, record, normalizer, horizon=horizon
        )
        direct_errors.append(direct_metrics.rollout_mse)
        lifted_errors.append(lifted_metrics.decoded_raw_mse)
        finite = finite and direct_metrics.finite and lifted_metrics.finite
    direct_mean = sum(direct_errors) / len(direct_errors)
    lifted_mean = sum(lifted_errors) / len(lifted_errors)
    finite = finite and all(
        torch.isfinite(torch.tensor(value))
        for value in (
            direct_fit.final_loss,
            lifted_fit.final_losses["total_loss"],
            direct_mean,
            lifted_mean,
        )
    )
    return DuffingLiftingDiagnostic(
        direct_state_rollout_mse=direct_mean,
        learned_lifting_rollout_mse=lifted_mean,
        direct_final_one_step_mse=direct_fit.final_loss,
        learned_final_loss=lifted_fit.final_losses["total_loss"],
        finite=bool(finite),
    )
