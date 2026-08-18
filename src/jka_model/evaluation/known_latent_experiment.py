"""Reusable deterministic V0.4 known-latent experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from jka_model.config import ProjectConfig, RepresentationLossConfig
from jka_model.data import (
    ChannelStandardizer,
    KnownLatentDataset,
    SplitManifest,
    TrajectoryRecord,
    TrajectoryWindowDataset,
    generate_known_latent_trajectories,
    hidden_rotation_decay_generator,
    make_split_manifest,
    select_split,
)
from jka_model.evaluation.representation import (
    LearnedRolloutMetrics,
    encode_records_for_alignment,
    evaluate_learned_trajectory,
)
from jka_model.metrics import (
    AffineLatentAlignment,
    LatentAlignmentMetrics,
    LatentDiagnostics,
    SpectrumDiagnostics,
    continuous_spectrum,
    evaluate_affine_latent_alignment,
    fit_affine_latent_alignment,
    latent_diagnostics,
)
from jka_model.models import KoopmanAutoencoder
from jka_model.training.koopman_representation import (
    RepresentationTrainingResult,
    initialize_koopman_autoencoder,
    train_koopman_representation,
)
from jka_model.utils import set_global_seed


@dataclass(frozen=True, slots=True)
class KnownLatentExperimentResult:
    dataset: KnownLatentDataset
    split_manifest: SplitManifest
    train_records: tuple[TrajectoryRecord, ...]
    validation_records: tuple[TrajectoryRecord, ...]
    test_records: tuple[TrajectoryRecord, ...]
    normalizer: ChannelStandardizer
    model: KoopmanAutoencoder
    training: RepresentationTrainingResult
    alignment: AffineLatentAlignment
    alignment_metrics: LatentAlignmentMetrics
    latent_diagnostics: LatentDiagnostics
    true_spectrum: SpectrumDiagnostics
    learned_spectrum: SpectrumDiagnostics
    test_reconstruction_model_mse: float
    test_reconstruction_raw_mse: float
    test_one_step_latent_mse: float
    test_multi_step_latent_mse: float
    rollout_decoded_model_mse: float
    rollout_decoded_raw_mse: float
    persistence_model_mse: float
    persistence_raw_mse: float
    rollout_finite: bool


def _mean_rollout_metrics(metrics: list[LearnedRolloutMetrics]) -> dict[str, float | bool]:
    if not metrics:
        raise ValueError("at least one held-out rollout metric is required")
    scalar_names = (
        "reconstruction_model_mse",
        "reconstruction_raw_mse",
        "one_step_latent_mse",
        "multi_step_latent_mse",
        "decoded_model_mse",
        "decoded_raw_mse",
        "persistence_model_mse",
        "persistence_raw_mse",
    )
    output: dict[str, float | bool] = {
        name: sum(getattr(item, name) for item in metrics) / len(metrics) for name in scalar_names
    }
    output["finite"] = all(item.finite for item in metrics)
    return output


def run_known_latent_experiment(
    config: ProjectConfig,
    *,
    epochs: int | None = None,
    loss_override: RepresentationLossConfig | None = None,
) -> KnownLatentExperimentResult:
    """Run train-only fitting and held-out V0.4 representation evaluation."""
    required = (
        config.koopman,
        config.known_latent,
        config.autoencoder,
        config.representation_loss,
        config.representation_training,
        config.representation_evaluation,
    )
    if any(section is None for section in required):
        raise ValueError("known-latent experiment requires all V0.4 config sections")
    assert config.koopman is not None
    assert config.known_latent is not None
    assert config.autoencoder is not None
    assert config.representation_loss is not None
    assert config.representation_training is not None
    assert config.representation_evaluation is not None
    if not config.koopman.trainable:
        raise ValueError("V0.4 learned experiment requires koopman.trainable=true")
    set_global_seed(config.training.seed, deterministic=config.training.deterministic)
    dtype = torch.float64 if config.koopman.dtype == "float64" else torch.float32
    dataset = generate_known_latent_trajectories(
        config.known_latent,
        seed=config.training.seed,
        dtype=dtype,
        nonlinear_observation=True,
    )
    manifest = make_split_manifest(dataset.records, config.data.split)
    train_records = select_split(dataset.records, manifest, "train")
    validation_records = select_split(dataset.records, manifest, "validation")
    test_records = select_split(dataset.records, manifest, "test")
    if not validation_records or not test_records:
        raise ValueError("V0.4 acceptance requires non-empty validation and test splits")
    normalizer = ChannelStandardizer(eps=config.data.normalization.eps).fit(
        dataset.records, manifest, dataset.problem_spec
    )
    train_windows = TrajectoryWindowDataset(
        train_records,
        history=config.data.history,
        horizon=config.data.horizon,
        normalizer=normalizer,
    )
    model = initialize_koopman_autoencoder(
        config.autoencoder,
        seed=config.training.seed,
        init_scale=config.representation_training.init_scale,
        dtype=dtype,
    )
    active_loss = config.representation_loss if loss_override is None else loss_override
    training = train_koopman_representation(
        model,
        train_windows,
        active_loss,
        config.representation_training,
        seed=config.training.seed,
        epochs=epochs,
    )
    model.eval()
    z_train, hidden_train = encode_records_for_alignment(
        model, train_records, normalizer, dataset.true_latents
    )
    z_test, hidden_test = encode_records_for_alignment(
        model, test_records, normalizer, dataset.true_latents
    )
    alignment = fit_affine_latent_alignment(z_train, hidden_train)
    alignment_metrics = evaluate_affine_latent_alignment(alignment, z_test, hidden_test)
    diagnostics = latent_diagnostics(z_test)
    horizon = config.representation_evaluation.rollout_horizon
    rollout_items = [
        evaluate_learned_trajectory(
            model,
            record,
            normalizer,
            horizon=horizon,
        )[0]
        for record in test_records
    ]
    mean_metrics = _mean_rollout_metrics(rollout_items)
    true_generator = hidden_rotation_decay_generator(
        config.known_latent.alpha,
        config.known_latent.omega,
        dtype=dtype,
    )
    return KnownLatentExperimentResult(
        dataset=dataset,
        split_manifest=manifest,
        train_records=train_records,
        validation_records=validation_records,
        test_records=test_records,
        normalizer=normalizer,
        model=model,
        training=training,
        alignment=alignment,
        alignment_metrics=alignment_metrics,
        latent_diagnostics=diagnostics,
        true_spectrum=continuous_spectrum(true_generator),
        learned_spectrum=model.core.spectrum(),
        test_reconstruction_model_mse=float(mean_metrics["reconstruction_model_mse"]),
        test_reconstruction_raw_mse=float(mean_metrics["reconstruction_raw_mse"]),
        test_one_step_latent_mse=float(mean_metrics["one_step_latent_mse"]),
        test_multi_step_latent_mse=float(mean_metrics["multi_step_latent_mse"]),
        rollout_decoded_model_mse=float(mean_metrics["decoded_model_mse"]),
        rollout_decoded_raw_mse=float(mean_metrics["decoded_raw_mse"]),
        persistence_model_mse=float(mean_metrics["persistence_model_mse"]),
        persistence_raw_mse=float(mean_metrics["persistence_raw_mse"]),
        rollout_finite=bool(mean_metrics["finite"]),
    )


def without_reconstruction(config: RepresentationLossConfig) -> RepresentationLossConfig:
    return replace(config, lambda_rec=0.0)


def without_multi_step(config: RepresentationLossConfig) -> RepresentationLossConfig:
    return replace(config, lambda_multi=0.0)
