#!/usr/bin/env python3
"""Inspect a saved V0.4 learned latent representation without plotting."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from jka_model.data import (
    ChannelStandardizer,
    generate_known_latent_trajectories,
    make_split_manifest,
    select_split,
)
from jka_model.evaluation import encode_records_for_alignment, evaluate_learned_trajectory
from jka_model.metrics import (
    evaluate_affine_latent_alignment,
    fit_affine_latent_alignment,
    latent_diagnostics,
)
from jka_model.training.koopman_representation import initialize_koopman_autoencoder
from jka_model.utils import load_checkpoint


def _latest_v0_4_checkpoint(root: Path) -> Path:
    candidates = sorted(
        (root / "runs").glob("*/checkpoint.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            checkpoint = load_checkpoint(candidate)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        config = checkpoint.config
        if (
            config is not None
            and config.known_latent is not None
            and config.autoencoder is not None
            and config.representation_training is not None
        ):
            return candidate
    raise FileNotFoundError("no compatible V0.4 checkpoint found; run smoke_v0_4.py first")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    arguments = parser.parse_args()
    checkpoint_path = (
        _latest_v0_4_checkpoint(root)
        if arguments.checkpoint is None
        else arguments.checkpoint
    )
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint.config
    if (
        config is None
        or config.koopman is None
        or config.known_latent is None
        or config.autoencoder is None
        or config.representation_training is None
        or config.representation_evaluation is None
    ):
        raise RuntimeError("checkpoint is not a complete V0.4 learned-representation artifact")
    dtype = torch.float64 if config.koopman.dtype == "float64" else torch.float32
    model = initialize_koopman_autoencoder(
        config.autoencoder,
        seed=config.training.seed,
        init_scale=config.representation_training.init_scale,
        dtype=dtype,
    )
    if checkpoint.online_model_state is None or checkpoint.normalizer_state is None:
        raise RuntimeError("V0.4 checkpoint lacks model or normalizer state")
    model.load_state_dict(checkpoint.online_model_state)
    model.eval()
    normalizer = ChannelStandardizer()
    normalizer.load_state_dict(checkpoint.normalizer_state)
    dataset = generate_known_latent_trajectories(
        config.known_latent, seed=config.training.seed, dtype=dtype
    )
    manifest = make_split_manifest(dataset.records, config.data.split)
    train_records = select_split(dataset.records, manifest, "train")
    test_records = select_split(dataset.records, manifest, "test")
    z_train, s_train = encode_records_for_alignment(
        model, train_records, normalizer, dataset.true_latents
    )
    z_test, s_test = encode_records_for_alignment(
        model, test_records, normalizer, dataset.true_latents
    )
    alignment = fit_affine_latent_alignment(z_train, s_train)
    metrics = evaluate_affine_latent_alignment(alignment, z_test, s_test)
    diagnostics = latent_diagnostics(z_test)
    rollout_items = [
        evaluate_learned_trajectory(
            model,
            record,
            normalizer,
            horizon=min(
                config.representation_evaluation.rollout_horizon, record.num_steps
            ),
        )[0]
        for record in test_records
    ]
    reconstruction = sum(item.reconstruction_model_mse for item in rollout_items) / len(
        rollout_items
    )
    reconstruction_raw = sum(item.reconstruction_raw_mse for item in rollout_items) / len(
        rollout_items
    )
    one_step_latent = sum(item.one_step_latent_mse for item in rollout_items) / len(
        rollout_items
    )
    multi_step_latent = sum(item.multi_step_latent_mse for item in rollout_items) / len(
        rollout_items
    )
    rollout = sum(item.decoded_model_mse for item in rollout_items) / len(rollout_items)
    rollout_raw = sum(item.decoded_raw_mse for item in rollout_items) / len(rollout_items)
    print("=== V0.4 Latent Analysis ===")
    print(f"checkpoint: {checkpoint_path}")
    print(f"latent mean: {diagnostics.mean.tolist()}")
    print(f"latent std: {diagnostics.std.tolist()}")
    print(f"minimum std: {diagnostics.minimum_std:.9e}")
    print(f"covariance condition: {diagnostics.covariance_condition:.9e}")
    print(f"linear alignment R2: {metrics.r2:.12f}")
    print(f"linear alignment MSE: {metrics.mse:.9e}")
    print(f"learned spectrum: {model.core.spectrum().eigenvalues.tolist()}")
    print(f"test reconstruction MSE: {reconstruction:.9e}")
    print(f"test raw reconstruction MSE: {reconstruction_raw:.9e}")
    print(f"one-step latent MSE: {one_step_latent:.9e}")
    print(f"multi-step latent rollout MSE: {multi_step_latent:.9e}")
    print(f"decoded model-space rollout MSE: {rollout:.9e}")
    print(f"decoded raw-space rollout MSE: {rollout_raw:.9e}")


if __name__ == "__main__":
    main()
