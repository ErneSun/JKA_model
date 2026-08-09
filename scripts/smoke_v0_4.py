#!/usr/bin/env python3
"""CPU-only end-to-end V0.4 learned Koopman representation acceptance test."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch

from jka_model.config import load_config, save_config
from jka_model.data import ChannelStandardizer, data_fingerprint
from jka_model.evaluation import (
    run_duffing_lifting_diagnostic,
    run_known_latent_experiment,
    without_multi_step,
    without_reconstruction,
)
from jka_model.metrics import dominant_oscillatory_mode, relative_frequency_error
from jka_model.training.koopman_representation import initialize_koopman_autoencoder
from jka_model.utils import (
    Checkpoint,
    capture_rng_state,
    create_run_directory,
    get_git_commit,
    load_checkpoint,
    save_checkpoint,
)


def _maximum_spectrum_error(left, right) -> float:
    return float(
        torch.maximum(
            (
                left.growth_rates.sort().values - right.growth_rates.sort().values
            ).abs().max(),
            (
                left.angular_frequencies.sort().values
                - right.angular_frequencies.sort().values
            ).abs().max(),
        ).item()
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "v0_4_smoke.yaml")
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    required = (
        config.koopman,
        config.known_latent,
        config.autoencoder,
        config.representation_loss,
        config.representation_training,
        config.representation_evaluation,
        config.duffing,
    )
    if any(section is None for section in required):
        raise RuntimeError("V0.4 smoke requires every representation and Duffing section")
    assert config.koopman is not None
    assert config.known_latent is not None
    assert config.autoencoder is not None
    assert config.representation_loss is not None
    assert config.representation_training is not None
    assert config.representation_evaluation is not None
    dtype = torch.float64 if config.koopman.dtype == "float64" else torch.float32
    run = create_run_directory(
        root / config.training.run_root,
        seed=config.training.seed,
        config_hash=config.stable_hash,
        train_stage=config.training.stage,
        git_commit=get_git_commit(root),
    )
    save_config(config, run.run_dir / "resolved_config.yaml")
    logger = logging.getLogger(f"jka_model.run.{run.run_id}")

    result = run_known_latent_experiment(config)
    true_growth, true_omega = dominant_oscillatory_mode(result.true_spectrum)
    learned_growth, learned_omega = dominant_oscillatory_mode(result.learned_spectrum)
    frequency_error = relative_frequency_error(learned_omega, true_omega)

    no_reconstruction = run_known_latent_experiment(
        config,
        epochs=config.representation_training.ablation_epochs,
        loss_override=without_reconstruction(config.representation_loss),
    )
    no_multi_step = run_known_latent_experiment(
        config,
        epochs=config.representation_training.ablation_epochs,
        loss_override=without_multi_step(config.representation_loss),
    )

    output_path = (
        run.run_dir / "checkpoint.pt"
        if arguments.checkpoint_output is None
        else arguments.checkpoint_output
    )
    checkpoint = Checkpoint(
        train_stage=config.training.stage,
        epoch=result.training.epochs,
        global_step=result.training.global_step,
        online_model_state=result.model.state_dict(),
        optimizer_state=result.training.optimizer_state,
        rng_state=capture_rng_state(),
        normalizer_state=result.normalizer.state_dict(),
        problem_spec=result.dataset.problem_spec,
        config=config,
        data_fingerprint=data_fingerprint(
            result.dataset.records, result.dataset.problem_spec
        ),
        split_manifest=result.split_manifest.to_dict(),
        physics_constraint_spec=[],
        git_commit=run.git_commit,
    )
    save_checkpoint(checkpoint, output_path)
    restored_checkpoint = load_checkpoint(output_path)
    restored = initialize_koopman_autoencoder(
        config.autoencoder,
        seed=config.training.seed + 999,
        init_scale=config.representation_training.init_scale,
        dtype=dtype,
    )
    assert restored_checkpoint.online_model_state is not None
    restored.load_state_dict(restored_checkpoint.online_model_state)
    restored_normalizer = ChannelStandardizer()
    assert restored_checkpoint.normalizer_state is not None
    restored_normalizer.load_state_dict(restored_checkpoint.normalizer_state)
    probe_raw = result.test_records[0].states_raw[:4]
    probe_model = result.normalizer.transform(probe_raw)
    probe_dt = result.test_records[0].dts[:3]
    before_z = result.model.encode(probe_model)
    after_z = restored.encode(restored_normalizer.transform(probe_raw))
    before_next = result.model.core.step(before_z[:3], probe_dt)
    after_next = restored.core.step(after_z[:3], probe_dt)
    before_decoded = result.model.decode(before_z)
    after_decoded = restored.decode(after_z)
    before_prediction = result.model.step(probe_model[:3], probe_dt)
    after_prediction = restored.step(
        restored_normalizer.transform(probe_raw[:3]), probe_dt
    )
    encoder_reload_error = float((before_z - after_z).abs().max().item())
    core_reload_error = float((before_next - after_next).abs().max().item())
    decoder_reload_error = float((before_decoded - after_decoded).abs().max().item())
    prediction_reload_error = float(
        (before_prediction - after_prediction).abs().max().item()
    )
    generator_reload_error = float((result.model.core.A - restored.core.A).abs().max().item())
    spectrum_reload_error = _maximum_spectrum_error(
        result.model.core.spectrum(), restored.core.spectrum()
    )
    metadata_consistent = (
        restored_checkpoint.optimizer_state is not None
        and restored_checkpoint.rng_state is not None
        and restored_checkpoint.config == config
        and restored_checkpoint.epoch == result.training.epochs
        and restored_checkpoint.global_step == result.training.global_step
        and restored_checkpoint.split_manifest == result.split_manifest.to_dict()
    )
    checkpoint_consistent = (
        max(
            encoder_reload_error,
            core_reload_error,
            decoder_reload_error,
            prediction_reload_error,
            generator_reload_error,
            spectrum_reload_error,
        )
        < 1e-12
        and metadata_consistent
    )
    duffing = run_duffing_lifting_diagnostic(config)
    thresholds = config.representation_evaluation
    gates = {
        "test_reconstruction": (
            result.test_reconstruction_model_mse
            < thresholds.max_test_reconstruction_mse
        ),
        "latent_noncollapse": (
            result.latent_diagnostics.minimum_std > thresholds.min_latent_std
        ),
        "train_fit_test_apply_alignment": (
            result.alignment_metrics.r2 > thresholds.min_alignment_r2
        ),
        "frequency": frequency_error < thresholds.max_frequency_relative_error,
        "stable_decay": learned_growth < 0,
        "rollout_finite": result.rollout_finite,
        "beats_persistence": (
            result.rollout_decoded_model_mse < result.persistence_model_mse
        ),
        "checkpoint_reload": checkpoint_consistent,
        "duffing_finite": duffing.finite,
        "ablations_finite": all(
            math.isfinite(value)
            for value in (
                no_reconstruction.test_reconstruction_model_mse,
                no_reconstruction.latent_diagnostics.minimum_std,
                no_multi_step.rollout_decoded_model_mse,
            )
        ),
    }

    print("=== V0.4 Learned Koopman Representation ===")
    print("\nDevice:\nCPU")
    print(f"\nArchitecture revision:\n{config.architecture.revision}")
    print(f"\nProject version:\n{config.project_version}")
    print(f"\nRun directory:\n{run.run_dir}")
    print("\n[Dataset]")
    print(f"train trajectories: {len(result.train_records)}")
    print(f"val trajectories: {len(result.validation_records)}")
    print(f"test trajectories: {len(result.test_records)}")
    print("\n[Representation]")
    print(f"latent dimension: {config.autoencoder.latent_dim}")
    print(f"latent mean: {result.latent_diagnostics.mean.tolist()}")
    print(f"latent std: {result.latent_diagnostics.std.tolist()}")
    print(f"minimum latent std: {result.latent_diagnostics.minimum_std:.9e}")
    print(f"covariance condition: {result.latent_diagnostics.covariance_condition:.9e}")
    print("\n[Training diagnostics]")
    print(f"snapshots: {len(result.training.diagnostic_history)}")
    print(
        "epochs: "
        f"{[snapshot.epoch for snapshot in result.training.diagnostic_history]}"
    )
    print(
        "initial/final total loss: "
        f"{result.training.initial_losses['total_loss']:.9e} / "
        f"{result.training.final_losses['total_loss']:.9e}"
    )
    print("\n[Reconstruction]")
    print(
        "train MSE: "
        f"{result.training.final_losses['reconstruction_loss']:.9e}"
    )
    print(f"test MSE: {result.test_reconstruction_model_mse:.9e}")
    print("\n[Prediction diagnostics]")
    print(f"one-step latent MSE: {result.test_one_step_latent_mse:.9e}")
    print(f"multi-step latent rollout MSE: {result.test_multi_step_latent_mse:.9e}")
    print(f"decoded model-space MSE: {result.rollout_decoded_model_mse:.9e}")
    print(f"decoded raw-space MSE: {result.rollout_decoded_raw_mse:.9e}")
    print("\n[Latent alignment]")
    print(f"test R2: {result.alignment_metrics.r2:.12f}")
    print(f"test MSE: {result.alignment_metrics.mse:.9e}")
    print("\n[Spectrum]")
    print(f"true eigenvalues: {result.true_spectrum.eigenvalues.tolist()}")
    print(f"learned eigenvalues: {result.learned_spectrum.eigenvalues.tolist()}")
    print(f"learned growth rates: {result.learned_spectrum.growth_rates.tolist()}")
    print(
        "learned angular frequencies: "
        f"{result.learned_spectrum.angular_frequencies.tolist()}"
    )
    print(f"learned frequencies Hz: {result.learned_spectrum.frequencies_hz.tolist()}")
    print(f"true frequency: {true_omega / (2.0 * math.pi):.9f} Hz")
    print(f"learned frequency: {learned_omega / (2.0 * math.pi):.9f} Hz")
    print(f"relative frequency error: {frequency_error:.9e}")
    print(f"true decay: {-true_growth:.9f}")
    print(f"learned decay: {-learned_growth:.9f}")
    print("\n[Rollout]")
    print(f"Koopman decoded rollout MSE: {result.rollout_decoded_model_mse:.9e}")
    print(f"persistence MSE: {result.persistence_model_mse:.9e}")
    print("\n[Ablations]")
    print(
        "without reconstruction — test reconstruction/min std: "
        f"{no_reconstruction.test_reconstruction_model_mse:.9e} / "
        f"{no_reconstruction.latent_diagnostics.minimum_std:.9e}"
    )
    print(
        "without multi-step — decoded rollout MSE: "
        f"{no_multi_step.rollout_decoded_model_mse:.9e}"
    )
    print("\n[Checkpoint]")
    print(f"path: {output_path}")
    print(
        "encoder/core/decoder/A/prediction/spectrum errors: "
        f"{encoder_reload_error:.3e} / {core_reload_error:.3e} / "
        f"{decoder_reload_error:.3e} / {generator_reload_error:.3e} / "
        f"{prediction_reload_error:.3e} / {spectrum_reload_error:.3e}"
    )
    print(f"reload consistency: {'PASS' if checkpoint_consistent else 'FAIL'}")
    print("\n[Duffing]")
    print(f"direct-state baseline error: {duffing.direct_state_rollout_mse:.9e}")
    print(f"learned-lifting error: {duffing.learned_lifting_rollout_mse:.9e}")
    print(f"finite: {'YES' if duffing.finite else 'NO'}")
    print("\nV0.4 mandatory gates:")
    for name, passed in gates.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print("PASS" if all(gates.values()) else "FAIL")
    logger.info(
        "v0_4_smoke metrics=%s",
        json.dumps(
            {
                "test_reconstruction_mse": result.test_reconstruction_model_mse,
                "alignment_r2": result.alignment_metrics.r2,
                "test_one_step_latent_mse": result.test_one_step_latent_mse,
                "test_multi_step_latent_mse": result.test_multi_step_latent_mse,
                "frequency_error": frequency_error,
                "rollout_mse": result.rollout_decoded_model_mse,
                "persistence_mse": result.persistence_model_mse,
                "duffing_direct_mse": duffing.direct_state_rollout_mse,
                "duffing_lifted_mse": duffing.learned_lifting_rollout_mse,
                "training_diagnostics": [
                    {
                        "epoch": snapshot.epoch,
                        "losses": snapshot.losses,
                        "latent_mean": snapshot.latent_mean,
                        "latent_std": snapshot.latent_std,
                        "minimum_latent_std": snapshot.minimum_latent_std,
                        "maximum_latent_std": snapshot.maximum_latent_std,
                    }
                    for snapshot in result.training.diagnostic_history
                ],
                "gates": gates,
            },
            sort_keys=True,
        ),
    )
    for handler in logger.handlers:
        handler.flush()
    if not all(gates.values()):
        failed = ", ".join(name for name, passed in gates.items() if not passed)
        raise RuntimeError(f"V0.4 smoke gate(s) failed: {failed}")


if __name__ == "__main__":
    main()
