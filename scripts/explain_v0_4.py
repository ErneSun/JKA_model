#!/usr/bin/env python3
"""Teach V0.4 learned Koopman coordinates with one trained CPU example."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from jka_model.config import load_config
from jka_model.evaluation import (
    run_duffing_lifting_diagnostic,
    run_known_latent_experiment,
)
from jka_model.metrics import dominant_oscillatory_mode, relative_frequency_error


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "v0_4_smoke.yaml")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    result = run_known_latent_experiment(config)
    record = result.test_records[0]
    true_hidden = result.dataset.latent(record.trajectory_id)
    raw = record.states_raw
    model_state = result.normalizer.transform(raw)
    with torch.no_grad():
        z_k = result.model.encode(model_state[:2])
        z_next = result.model.core.step(z_k[:1], record.dts[:1])
        decoded_next_model = result.model.decode(z_next)
        decoded_next_raw = result.normalizer.inverse_transform(decoded_next_model)
    true_growth, true_omega = dominant_oscillatory_mode(result.true_spectrum)
    learned_growth, learned_omega = dominant_oscillatory_mode(result.learned_spectrum)

    print("=== V0.4: Why Learn Koopman Coordinates? ===")
    print("\nSTEP 1 — Start from hidden linear dynamics s")
    print("s_dot = [[-alpha,-omega],[omega,-alpha]] s")
    print(f"true hidden state s_t: {true_hidden[0].tolist()}")
    print("\nSTEP 2 — Convert s into nonlinear observations U=g(s)")
    print("g(s) = [q,p,q^2,qp,p^2]")
    print(f"nonlinear observation U_t: {raw[0].tolist()}")
    print("\nSTEP 3 — Explain why U is no longer the simplest dynamical coordinate")
    print("U has five correlated channels although the generating state has dimension two.")
    print("\nSTEP 4 — Send U_model into KoopmanEncoder")
    print(f"normalized U_model,t: {model_state[0].tolist()}")
    print("Encoder sees only U_model; evaluation-only s is not a model input.")
    print("\nSTEP 5 — Inspect z_k")
    print(f"encoded z_k,t: {z_k[0].tolist()}")
    print(f"test latent std: {result.latent_diagnostics.std.tolist()}")
    print("\nSTEP 6 — Propagate z_k using V0.3 KoopmanCore")
    print(f"dt_t: {float(record.dts[0]):.9f}")
    print(f"predicted z_k,t+1: {z_next[0].tolist()}")
    print("The propagation is exactly exp(A*dt) z; V0.3 core semantics are unchanged.")
    print("\nSTEP 7 — Decode z_k back to U")
    print(f"decoded U_model,t+1: {decoded_next_model[0].tolist()}")
    print(f"decoded U_raw,t+1: {decoded_next_raw[0].tolist()}")
    print(f"true U_raw,t+1: {raw[1].tolist()}")
    print("\nSTEP 8 — Explain reconstruction loss")
    print("L_rec = MSE(D(E(U_model)), U_model); its target is model space.")
    print(
        "train/test reconstruction MSE: "
        f"{result.training.final_losses['reconstruction_loss']:.6e} / "
        f"{result.test_reconstruction_model_mse:.6e}"
    )
    print("\nSTEP 9 — Explain Koopman consistency loss")
    print("L_K compares exp(A*dt)E(U_t) with E(U_t+1); both use the online encoder.")
    print(
        "train final / held-out one-step latent MSE: "
        f"{result.training.final_losses['koopman_one_step_loss']:.6e} / "
        f"{result.test_one_step_latent_mse:.6e}"
    )
    print("\nSTEP 10 — Explain multi-step rollout loss")
    print("Predicted latent states feed the next step; true futures are targets only.")
    print(
        "train final / held-out closed-loop multi-step latent MSE: "
        f"{result.training.final_losses['koopman_multi_step_loss']:.6e} / "
        f"{result.test_multi_step_latent_mse:.6e}"
    )
    print("\nSTEP 11 — Explain latent collapse")
    print("E(U)=0 makes Koopman consistency trivially zero but carries no state information.")
    print("\nSTEP 12 — Explain variance loss")
    print("L_var penalizes per-coordinate population std below the configured minimum.")
    print(f"minimum held-out latent std: {result.latent_diagnostics.minimum_std:.6e}")
    print("\nSTEP 13 — Compare learned z_k to true s after linear alignment")
    print("The affine map is fitted on train trajectories and applied unchanged to test.")
    print(
        f"held-out alignment R2/MSE: {result.alignment_metrics.r2:.12f} / "
        f"{result.alignment_metrics.mse:.6e}"
    )
    print("\nSTEP 14 — Explain why exact coordinate equality is unnecessary")
    print("For z=T s, A_z=T A_true T^-1; coordinates differ but eigenvalues do not.")
    print("\nSTEP 15 — Compare true and learned spectrum")
    print(f"true eigenvalues: {result.true_spectrum.eigenvalues.tolist()}")
    print(f"learned eigenvalues: {result.learned_spectrum.eigenvalues.tolist()}")
    print(
        "frequency true/learned/error: "
        f"{true_omega / (2 * math.pi):.9f} / "
        f"{learned_omega / (2 * math.pi):.9f} Hz / "
        f"{relative_frequency_error(learned_omega, true_omega):.6e}"
    )
    print(f"decay true/learned: {-true_growth:.9f} / {-learned_growth:.9f}")
    print("\nSTEP 16 — Show Duffing result")
    duffing = run_duffing_lifting_diagnostic(config)
    print(
        "direct-state / learned-lifting rollout MSE: "
        f"{duffing.direct_state_rollout_mse:.6e} / "
        f"{duffing.learned_lifting_rollout_mse:.6e}"
    )
    print("This finite diagnostic does not prove exact finite-dimensional Duffing closure.")
    print("\nSTEP 17 — Explain what V0.5 adds")
    print("V0.5 can reuse Encoder/Decoder/Core training interfaces for 2D/PDE fields")
    print("and first apply PhysicsConstraint after inverse-transforming decoded raw state.")
    print("V0.4 adds no JEPA, EMA, residual closure, attention, PDE, or action dynamics.")


if __name__ == "__main__":
    main()
