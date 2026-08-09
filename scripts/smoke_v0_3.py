#!/usr/bin/env python3
"""CPU-only end-to-end V0.3 direct-state Koopman smoke test."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch

from jka_model.config import load_config, save_config
from jka_model.data import (
    TrajectoryDataset,
    damped_oscillator_analytic_transition,
    damped_oscillator_generator_matrix,
    data_fingerprint,
    generate_damped_oscillator_trajectories,
    generate_duffing_trajectories,
    make_split_manifest,
    select_split,
    trajectory_transition_tensors,
)
from jka_model.evaluation import evaluate_rollout
from jka_model.metrics import (
    dominant_oscillatory_mode,
    relative_frequency_error,
    spectral_growth_rate,
)
from jka_model.models import ContinuousKoopmanCore
from jka_model.training import initialize_direct_koopman, train_direct_koopman
from jka_model.utils import (
    Checkpoint,
    capture_rng_state,
    create_run_directory,
    get_git_commit,
    load_checkpoint,
    save_checkpoint,
    set_global_seed,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "v0_3_smoke.yaml")
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if any(
        section is None
        for section in (
            config.koopman,
            config.oscillator,
            config.duffing,
            config.identification,
            config.evaluation,
        )
    ):
        raise RuntimeError("V0.3 smoke requires all direct-state config sections")
    assert config.koopman is not None
    assert config.oscillator is not None
    assert config.duffing is not None
    assert config.identification is not None
    assert config.evaluation is not None
    if not config.koopman.trainable:
        raise RuntimeError("V0.3 identification config requires koopman.trainable=true")
    set_global_seed(config.training.seed, deterministic=config.training.deterministic)
    dtype = torch.float64 if config.koopman.dtype == "float64" else torch.float32
    device = torch.device("cpu")
    state_dim = config.koopman.state_dim
    run = create_run_directory(
        root / config.training.run_root,
        seed=config.training.seed,
        config_hash=config.stable_hash,
        train_stage=config.training.stage,
        git_commit=get_git_commit(root),
    )
    save_config(config, run.run_dir / "resolved_config.yaml")
    logger = logging.getLogger(f"jka_model.run.{run.run_id}")

    true_omega = math.sqrt(config.oscillator.omega0**2 - config.oscillator.gamma**2)
    verification_generator = damped_oscillator_generator_matrix(
        config.oscillator.omega0,
        config.oscillator.gamma,
        dtype=dtype,
    )
    fixed = ContinuousKoopmanCore(
        state_dim,
        generator=verification_generator,
        trainable=False,
        dtype=dtype,
        device=device,
    )
    verification_dt = config.oscillator.base_dt
    closed_form = damped_oscillator_analytic_transition(
        config.oscillator.omega0,
        config.oscillator.gamma,
        verification_dt,
        dtype=dtype,
    )
    matrix_error = float((fixed.transition_matrix(verification_dt) - closed_form).abs().max())
    dt1, dt2 = 0.7 * verification_dt, 1.3 * verification_dt
    semigroup_error = float(
        (
            fixed.transition_matrix(dt1 + dt2)
            - fixed.transition_matrix(dt2) @ fixed.transition_matrix(dt1)
        )
        .abs()
        .max()
    )
    zero_error = float(
        (fixed.transition_matrix(0.0) - torch.eye(state_dim, dtype=dtype)).abs().max()
    )

    records, spec = generate_damped_oscillator_trajectories(
        config.oscillator, seed=config.training.seed, dtype=dtype
    )
    split_manifest = make_split_manifest(records, config.data.split)
    train_records = TrajectoryDataset(select_split(records, split_manifest, "train"))
    evaluation_split = "test" if split_manifest.test else "validation"
    evaluation_records = select_split(records, split_manifest, evaluation_split)
    states, targets, dts = trajectory_transition_tensors(train_records)
    learned = initialize_direct_koopman(
        config.koopman.state_dim,
        seed=config.training.seed,
        init_scale=config.identification.init_scale,
        dtype=dtype,
        device=device,
    )
    identification = train_direct_koopman(
        learned, states, targets, dts, config.identification
    )
    learned_growth, learned_omega = dominant_oscillatory_mode(learned.spectrum())
    frequency_error = relative_frequency_error(learned_omega, true_omega)
    true_frequency = true_omega / (2.0 * math.pi)
    learned_frequency = learned_omega / (2.0 * math.pi)
    learned_damping = -learned_growth
    damping_error = abs(learned_damping - config.oscillator.gamma) / config.oscillator.gamma

    horizon = config.evaluation.rollout_horizon
    evaluation_record = evaluation_records[0]
    prediction = learned.rollout(
        evaluation_record.states_raw[0], evaluation_record.dts[:horizon]
    )
    rollout_metrics = evaluate_rollout(prediction, evaluation_record.states_raw[: horizon + 1])

    output_path = (
        run.run_dir / "checkpoint.pt"
        if arguments.checkpoint_output is None
        else arguments.checkpoint_output
    )
    checkpoint = Checkpoint(
        train_stage=config.training.stage,
        epoch=identification.epochs,
        global_step=identification.global_step,
        online_model_state=learned.state_dict(),
        optimizer_state=identification.optimizer_state,
        rng_state=capture_rng_state(),
        problem_spec=spec,
        config=config,
        data_fingerprint=data_fingerprint(records, spec),
        split_manifest=split_manifest.to_dict(),
        physics_constraint_spec=[],
        git_commit=run.git_commit,
    )
    save_checkpoint(checkpoint, output_path)
    restored_checkpoint = load_checkpoint(output_path)
    restored = ContinuousKoopmanCore(
        state_dim, trainable=True, dtype=dtype, device=device
    )
    assert restored_checkpoint.online_model_state is not None
    restored.load_state_dict(restored_checkpoint.online_model_state)
    before = learned.step(states[:4], dts[:4])
    after = restored.step(states[:4], dts[:4])
    reload_error = float((before - after).abs().max().detach())
    before_spectrum = learned.spectrum()
    after_spectrum = restored.spectrum()
    reload_spectrum_error = float(
        torch.maximum(
            (
                before_spectrum.growth_rates.sort().values
                - after_spectrum.growth_rates.sort().values
            ).abs().max(),
            (
                before_spectrum.angular_frequencies.sort().values
                - after_spectrum.angular_frequencies.sort().values
            ).abs().max(),
        )
    )
    checkpoint_metadata_consistent = (
        restored_checkpoint.optimizer_state is not None
        and restored_checkpoint.rng_state is not None
        and restored_checkpoint.config == config
        and restored_checkpoint.epoch == identification.epochs
        and restored_checkpoint.global_step == identification.global_step
        and restored_checkpoint.split_manifest == split_manifest.to_dict()
    )

    duffing_records, _ = generate_duffing_trajectories(
        config.duffing, seed=config.training.seed + 1, dtype=dtype
    )
    duffing_manifest = make_split_manifest(duffing_records, config.data.split)
    duffing_train_records = TrajectoryDataset(
        select_split(duffing_records, duffing_manifest, "train")
    )
    duffing_evaluation_split = "test" if duffing_manifest.test else "validation"
    duffing_evaluation_records = select_split(
        duffing_records, duffing_manifest, duffing_evaluation_split
    )
    duffing_states, duffing_targets, duffing_dts = trajectory_transition_tensors(
        duffing_train_records
    )
    duffing_core = initialize_direct_koopman(
        state_dim,
        seed=config.training.seed + 1,
        init_scale=config.identification.init_scale,
        dtype=dtype,
        device=device,
    )
    duffing_fit = train_direct_koopman(
        duffing_core,
        duffing_states,
        duffing_targets,
        duffing_dts,
        config.identification,
    )
    duffing_horizon = min(horizon, config.duffing.num_steps)
    duffing_evaluation_record = duffing_evaluation_records[0]
    duffing_prediction = duffing_core.rollout(
        duffing_evaluation_record.states_raw[0],
        duffing_evaluation_record.dts[:duffing_horizon],
    )
    duffing_metrics = evaluate_rollout(
        duffing_prediction,
        duffing_evaluation_record.states_raw[: duffing_horizon + 1],
    )

    gates = {
        "fixed_matrix_exp": matrix_error < 1e-10,
        "semigroup": semigroup_error < 1e-10,
        "zero_dt": zero_error < 1e-12,
        "frequency_error_below_1_percent": frequency_error < 0.01,
        "learned_growth_stable": spectral_growth_rate(learned.spectrum()) < 0,
        "rollout_finite": rollout_metrics.finite,
        "beats_persistence": rollout_metrics.rollout_mse < rollout_metrics.persistence_mse,
        "checkpoint_reload": (
            reload_error < 1e-12
            and reload_spectrum_error < 1e-12
            and checkpoint_metadata_consistent
        ),
        "duffing_finite": duffing_metrics.finite and math.isfinite(duffing_fit.final_loss),
    }

    print("=== V0.3 Direct-State Koopman Smoke Test ===")
    print("\nDevice:\nCPU")
    print(f"\nArchitecture revision:\n{config.architecture.revision}")
    print(f"\nProject version:\n{config.project_version}")
    print(f"\nRun directory:\n{run.run_dir}")
    print("\n[Fixed-A verification]")
    print(f"matrix exponential error: {matrix_error:.6e}")
    print(f"semigroup error: {semigroup_error:.6e}")
    print(f"zero-dt error: {zero_error:.6e}")
    print("\n[Identification]")
    print(
        "trajectory split train/validation/test: "
        f"{len(split_manifest.train)}/{len(split_manifest.validation)}/{len(split_manifest.test)}"
    )
    print(
        f"initial/final loss: {identification.initial_loss:.6e} / "
        f"{identification.final_loss:.6e}"
    )
    print(f"true frequency: {true_frequency:.9f} Hz")
    print(f"learned frequency: {learned_frequency:.9f} Hz")
    print(f"relative frequency error: {frequency_error:.6e}")
    print(f"true damping: {config.oscillator.gamma:.9f}")
    print(f"learned damping: {learned_damping:.9f}")
    print(f"relative damping error: {damping_error:.6e}")
    print("\n[Rollout]")
    print(f"100-step Koopman MSE: {rollout_metrics.rollout_mse:.6e}")
    print(f"100-step persistence MSE: {rollout_metrics.persistence_mse:.6e}")
    print(f"finite: {'YES' if rollout_metrics.finite else 'NO'}")
    print("\n[Spectrum]")
    print(f"continuous eigenvalues: {learned.spectrum().eigenvalues.tolist()}")
    print("\n[Checkpoint]")
    print(f"path: {output_path}")
    print(f"state/spectrum reload error: {reload_error:.6e} / {reload_spectrum_error:.6e}")
    print(f"metadata consistency: {'PASS' if checkpoint_metadata_consistent else 'FAIL'}")
    print(f"reload consistency: {'PASS' if gates['checkpoint_reload'] else 'FAIL'}")
    print("\n[Duffing]")
    print(f"one-step error: {duffing_fit.final_loss:.6e}")
    print(f"rollout error: {duffing_metrics.rollout_mse:.6e}")
    print(f"finite: {'YES' if duffing_metrics.finite else 'NO'}")
    print("\nV0.3 mandatory gates:")
    print("PASS" if all(gates.values()) else "FAIL")
    logger.info(
        "v0_3_smoke metrics=%s",
        json.dumps(
            {
                "matrix_exp_error": matrix_error,
                "frequency_error": frequency_error,
                "rollout_mse": rollout_metrics.rollout_mse,
                "persistence_mse": rollout_metrics.persistence_mse,
                "duffing_rollout_mse": duffing_metrics.rollout_mse,
                "gates": gates,
            },
            sort_keys=True,
        ),
    )
    for handler in logger.handlers:
        handler.flush()
    if not all(gates.values()):
        failed = ", ".join(name for name, passed in gates.items() if not passed)
        raise RuntimeError(f"V0.3 smoke gate(s) failed: {failed}")


if __name__ == "__main__":
    main()
