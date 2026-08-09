#!/usr/bin/env python3
"""Teach the V0.2 pipeline with actual IDs, indices, values, and residuals."""

from __future__ import annotations

from pathlib import Path

from jka_model.config import load_config
from jka_model.data import (
    ChannelStandardizer,
    TrajectoryWindowDataset,
    generate_advection_diffusion_trajectories,
    make_split_manifest,
    select_split,
)
from jka_model.physics import (
    DiscretePDEResidualConstraint,
    MassConservationConstraint,
    PeriodicBoundaryConstraint,
    evaluate_constraints,
)


def _indices(prefix: str, start: int, stop: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(start, stop))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "v0_2_smoke.yaml")
    toy = config.data.toy_advection_diffusion
    if toy is None:
        raise RuntimeError("V0.2 explanation requires toy data config")

    print("STEP 1 — Generate trajectory")
    records, spec = generate_advection_diffusion_trajectories(
        toy, seed=config.training.seed
    )
    first_record = records[0]
    print(f"trajectory count: {len(records)}")
    print(f"example trajectory_id: {first_record.trajectory_id}")
    print(f"states [T+1,C,Nx]: {list(first_record.states_raw.shape)}")
    print(f"dts [T]: {list(first_record.dts.shape)}; dtype: {first_record.dts.dtype}")

    print("\nSTEP 2 — Inspect raw states")
    assert first_record.mu_static is not None
    print(
        f"mu_static [c,nu]: [{float(first_record.mu_static[0]):.6f}, "
        f"{float(first_record.mu_static[1]):.6f}]"
    )
    print(f"raw U0 channel mean: {float(first_record.states_raw[0].mean()):.6f}")
    print("raw state remains in physical units and is not normalized in place")

    print("\nSTEP 3 — Split trajectories")
    manifest = make_split_manifest(records, config.data.split)
    print(
        f"train/validation/test: {len(manifest.train)}/"
        f"{len(manifest.validation)}/{len(manifest.test)}"
    )
    print(f"first train trajectory: {manifest.train[0]}")
    print("the three trajectory-ID sets are disjoint before any windows exist")

    print("\nSTEP 4 — Fit train normalizer")
    normalizer = ChannelStandardizer(eps=config.data.normalization.eps).fit(
        records, manifest, spec
    )
    assert normalizer.mean is not None and normalizer.scale is not None
    print(f"fit scope: TRAIN ONLY ({len(normalizer.fitted_trajectory_ids)} IDs)")
    print(
        f"frozen channel statistics: mean={float(normalizer.mean[0]):.6f}, "
        f"scale={float(normalizer.scale[0]):.6f}"
    )

    print("\nSTEP 5 — Build one time window")
    train_records = select_split(records, manifest, "train")
    windows = TrajectoryWindowDataset(
        train_records,
        history=config.data.history,
        horizon=config.data.horizon,
        normalizer=normalizer,
    )
    batch = windows[0]
    history = config.data.history
    horizon = config.data.horizon
    print(f"trajectory_id: {batch.trajectory_id[0]}")
    print(f"history: {_indices('U', 0, history)}")
    print(f"future:  {_indices('U', history, history + horizon)}")

    print("\nSTEP 6 — Show exact Ui/dti alignment")
    print(f"history_dts: {_indices('dt', 0, history - 1)}")
    print(f"future_dts:  {_indices('dt', history - 1, history - 1 + horizon)}")
    print(f"actual future_dts: {[round(float(v), 7) for v in batch.future_dts[0]]}")
    print(f"dt{history - 1} belongs only to U{history - 1} -> U{history}")

    print("\nSTEP 7 — Build ProblemBatch")
    print(f"context_states_raw:   {list(batch.context_states_raw.shape)}")
    print(f"future_states_raw:    {list(batch.future_states_raw.shape)}")
    print(f"context_states_model: {list(batch.context_states_model.shape)}")
    print(f"history/future dts:   {list(batch.history_dts.shape)} / {list(batch.future_dts.shape)}")
    print("history_actions/future_actions: None / None (the toy system is actionless)")

    print("\nSTEP 8 — Compare raw/model state")
    raw_state = batch.context_states_raw[0, -1]
    model_state = batch.context_states_model[0, -1]
    print(f"raw endpoint value:        {float(raw_state[0, 0]):.6f}")
    print(f"normalized endpoint value: {float(model_state[0, 0]):.6f}")
    print(f"raw channel mean:          {float(raw_state.mean()):.6f}")
    print(f"normalized channel mean:   {float(model_state.mean()):.6f}")

    print("\nSTEP 9 — Evaluate physics constraints")
    terms = evaluate_constraints(
        [
            PeriodicBoundaryConstraint(),
            MassConservationConstraint(),
            DiscretePDEResidualConstraint(),
        ],
        batch,
        spec,
    )
    print(f"periodic boundary error: {float(terms['periodic_boundary']):.3e}")
    print(f"mass residual:           {float(terms['mass_conservation']):.3e}")
    print(f"PDE residual:            {float(terms['discrete_pde_residual']):.3e}")
    print("all three read future_states_raw, never future_states_model")

    print("\nSTEP 10 — Show what V0.3 will consume")
    print("V0.3 will connect to canonical model-state/action/dt interfaces.")
    print("It will first validate direct-state KoopmanCore mathematics on oscillator systems.")
    print("No Koopman operator, encoder, JEPA, residual closure, or decoder exists in V0.2.")


if __name__ == "__main__":
    main()
