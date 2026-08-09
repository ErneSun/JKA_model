#!/usr/bin/env python3
"""Teach the V0.3 continuous-time Koopman core with concrete mathematics."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from jka_model.config import load_config
from jka_model.data import (
    TrajectoryDataset,
    damped_oscillator_generator_matrix,
    generate_damped_oscillator_trajectories,
    generate_duffing_trajectories,
    make_split_manifest,
    select_split,
    trajectory_transition_tensors,
)
from jka_model.evaluation import evaluate_rollout
from jka_model.metrics import dominant_oscillatory_mode, relative_frequency_error
from jka_model.models import ContinuousKoopmanCore
from jka_model.training import initialize_direct_koopman, train_direct_koopman
from jka_model.utils import set_global_seed


def _matrix_text(matrix: torch.Tensor) -> str:
    rows = [[round(float(value), 7) for value in row] for row in matrix.detach()]
    return "\n".join(str(row) for row in rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "v0_3_smoke.yaml")
    assert config.koopman is not None
    assert config.oscillator is not None
    assert config.duffing is not None
    assert config.identification is not None
    assert config.evaluation is not None
    if not config.koopman.trainable:
        raise RuntimeError("V0.3 teaching identification requires koopman.trainable=true")
    set_global_seed(config.training.seed, deterministic=True)
    dtype = torch.float64 if config.koopman.dtype == "float64" else torch.float32
    state_dim = config.koopman.state_dim

    print("STEP 1 — Define a continuous dynamical system")
    print("state z=[x,v]; equations: x_dot=v, v_dot=-omega0^2*x-2*gamma*v")
    print(f"omega0={config.oscillator.omega0}, gamma={config.oscillator.gamma}")

    print("\nSTEP 2 — Show generator A")
    true_generator = damped_oscillator_generator_matrix(
        config.oscillator.omega0, config.oscillator.gamma, dtype=dtype
    )
    print("A =")
    print(_matrix_text(true_generator))
    print("A is the continuous generator in dz/dt=A z; it is not a one-step matrix.")

    print("\nSTEP 3 — Explain eigenvalues of A")
    fixed = ContinuousKoopmanCore(
        state_dim, generator=true_generator, trainable=False, dtype=dtype
    )
    true_spectrum = fixed.spectrum()
    print(f"continuous eigenvalues = {true_spectrum.eigenvalues.tolist()}")
    print("real part = decay/growth rate; |imaginary part| = angular frequency")

    print("\nSTEP 4 — Convert A into K(dt)=exp(A dt)")
    dt = config.oscillator.base_dt
    transition = fixed.transition_matrix(dt)
    print(f"dt = {dt}")
    print("K(dt) = exp(A dt) =")
    print(_matrix_text(transition))
    print("torch.matrix_exp computes K; no I+dt*A Euler approximation is used")

    print("\nSTEP 5 — Propagate one state")
    state = torch.tensor([1.0, 0.0], dtype=dtype)
    next_state = fixed.step(state, dt)
    print(f"z_t = {[round(float(v), 7) for v in state]}")
    print(f"z_(t+dt) = {[round(float(v), 7) for v in next_state]}")
    print("code stores z as a row, but computes the mathematical column operation K @ z")

    print("\nSTEP 6 — Change dt and observe K(dt)")
    small_dt, large_dt = 0.5 * dt, 1.5 * dt
    small_row = [round(float(v), 7) for v in fixed.transition_matrix(small_dt)[0]]
    large_row = [round(float(v), 7) for v in fixed.transition_matrix(large_dt)[0]]
    print(f"dt_small={small_dt}; K_small[0]={small_row}")
    print(f"dt_large={large_dt}; K_large[0]={large_row}")
    print("A is fixed; each interval produces a different propagator K(dt)")

    print("\nSTEP 7 — Perform variable-dt rollout")
    variable_dts = torch.tensor([0.02, 0.05, 0.03], dtype=dtype)
    variable_rollout = fixed.rollout(state, variable_dts)
    print(f"dts = {variable_dts.tolist()}")
    print(f"rollout shape = {list(variable_rollout.shape)} (includes z0)")
    print(f"final state = {[round(float(v), 7) for v in variable_rollout[-1]]}")

    print("\nSTEP 8 — Create trainable A")
    records, _ = generate_damped_oscillator_trajectories(
        config.oscillator, seed=config.training.seed, dtype=dtype
    )
    split_manifest = make_split_manifest(records, config.data.split)
    train_records = TrajectoryDataset(select_split(records, split_manifest, "train"))
    evaluation_split = "test" if split_manifest.test else "validation"
    evaluation_records = select_split(records, split_manifest, evaluation_split)
    states, targets, dts = trajectory_transition_tensors(train_records)
    learned = initialize_direct_koopman(
        state_dim,
        seed=config.training.seed,
        init_scale=config.identification.init_scale,
        dtype=dtype,
    )
    initial_generator = learned.A.detach().clone()
    print(
        "trajectory split train/validation/test = "
        f"{len(split_manifest.train)}/{len(split_manifest.validation)}/{len(split_manifest.test)}"
    )
    print("deterministic random A_init =")
    print(_matrix_text(initial_generator))
    print("A_init is not initialized from A_true")

    print("\nSTEP 9 — Show how loss updates A")
    fit = train_direct_koopman(learned, states, targets, dts, config.identification)
    print(f"one-step MSE: {fit.initial_loss:.6e} -> {fit.final_loss:.6e}")
    print("learned A =")
    print(_matrix_text(learned.A))
    print("gradient path: MSE -> exp(A*dt) -> A; A is the only trainable tensor")

    print("\nSTEP 10 — Compare true and learned spectrum")
    learned_growth, learned_omega = dominant_oscillatory_mode(learned.spectrum())
    true_omega = math.sqrt(config.oscillator.omega0**2 - config.oscillator.gamma**2)
    print(f"true angular frequency = {true_omega:.9f}")
    print(f"learned angular frequency = {learned_omega:.9f}")
    print(
        "relative frequency error = "
        f"{relative_frequency_error(learned_omega, true_omega):.6e}"
    )
    print(f"true/learned damping = {config.oscillator.gamma:.9f} / {-learned_growth:.9f}")

    print("\nSTEP 11 — Compare Koopman and persistence")
    horizon = config.evaluation.rollout_horizon
    evaluation_record = evaluation_records[0]
    prediction = learned.rollout(
        evaluation_record.states_raw[0], evaluation_record.dts[:horizon]
    )
    metrics = evaluate_rollout(
        prediction, evaluation_record.states_raw[: horizon + 1]
    )
    print(f"held-out trajectory = {evaluation_record.trajectory_id}")
    print(f"100-step Koopman MSE = {metrics.rollout_mse:.6e}")
    print(f"100-step persistence MSE = {metrics.persistence_mse:.6e}")

    print("\nSTEP 12 — Show Duffing limitation")
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
    )
    duffing_fit = train_direct_koopman(
        duffing_core,
        duffing_states,
        duffing_targets,
        duffing_dts,
        config.identification,
    )
    duffing_evaluation_record = duffing_evaluation_records[0]
    duffing_prediction = duffing_core.rollout(
        duffing_evaluation_record.states_raw[0],
        duffing_evaluation_record.dts[:horizon],
    )
    duffing_metrics = evaluate_rollout(
        duffing_prediction,
        duffing_evaluation_record.states_raw[: horizon + 1],
    )
    print(f"Duffing one-step MSE = {duffing_fit.final_loss:.6e}")
    print(f"Duffing rollout MSE = {duffing_metrics.rollout_mse:.6e}")
    print("x^3 makes the 2D direct state non-closed under one fixed linear A; this is expected")

    print("\nSTEP 13 — Explain what V0.4 will add")
    print("V0.4 will learn U -> z_k, then feed z_k into this already validated KoopmanCore.")
    print("V0.3 implements no encoder, decoder, JEPA, residual, attention, or action conditioning.")


if __name__ == "__main__":
    main()
