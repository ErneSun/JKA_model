#!/usr/bin/env python3
"""CPU-only end-to-end smoke test for the complete V0.2 pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jka_model.config import load_config
from jka_model.data import (
    ChannelStandardizer,
    SplitManifest,
    TrajectoryWindowDataset,
    collate_problem_batches,
    data_fingerprint,
    generate_advection_diffusion_trajectories,
    make_split_manifest,
    select_split,
    validate_trajectories_against_spec,
)
from jka_model.physics import (
    ChannelMeanProbe,
    ChannelRMSProbe,
    create_constraint,
    evaluate_batch_probes,
    evaluate_constraints,
)
from jka_model.utils import Checkpoint, load_checkpoint, save_checkpoint, set_global_seed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "v0_2_smoke.yaml")
    set_global_seed(config.training.seed, deterministic=config.training.deterministic)
    toy = config.data.toy_advection_diffusion
    if toy is None:
        raise RuntimeError("V0.2 smoke config requires toy_advection_diffusion")

    records, spec = generate_advection_diffusion_trajectories(
        toy, seed=config.training.seed
    )
    validate_trajectories_against_spec(records, spec)
    fingerprint = data_fingerprint(records, spec)
    manifest = make_split_manifest(records, config.data.split)
    with tempfile.TemporaryDirectory(prefix="jka-v0-2-split-") as split_temporary:
        early_manifest_path = Path(split_temporary) / "split_manifest.json"
        manifest.save(early_manifest_path)
        if SplitManifest.load(early_manifest_path) != manifest:
            raise RuntimeError("split manifest round-trip failed")
    normalizer = ChannelStandardizer(eps=config.data.normalization.eps).fit(
        records, manifest, spec
    )
    split_windows = {
        split_name: TrajectoryWindowDataset(
            select_split(records, manifest, split_name),
            history=config.data.history,
            horizon=config.data.horizon,
            normalizer=normalizer,
        )
        for split_name in ("train", "validation", "test")
    }
    for split_name, split_dataset in split_windows.items():
        first_window = split_dataset[0]
        reconstructed_split = normalizer.inverse_transform(
            first_window.context_states_model
        )
        torch.testing.assert_close(reconstructed_split, first_window.context_states_raw)
        if not set(first_window.trajectory_id).issubset(set(getattr(manifest, split_name))):
            raise RuntimeError(f"{split_name} window contains an ID from another split")
    windows = split_windows["train"]
    loader = DataLoader(
        windows, batch_size=3, shuffle=False, collate_fn=collate_problem_batches
    )
    batch = next(iter(loader))
    reconstructed = normalizer.inverse_transform(batch.context_states_model)
    torch.testing.assert_close(reconstructed, batch.context_states_raw)

    constraint_spec = [
        {"name": "finite_values", "parameters": {}},
        {"name": "state_admissibility", "parameters": {"lower": 0.0, "upper": 2.0}},
        {"name": "periodic_boundary", "parameters": {}},
        {"name": "mass_conservation", "parameters": {}},
        {"name": "discrete_pde_residual", "parameters": {}},
    ]
    constraints = [create_constraint(item) for item in constraint_spec]
    physics_terms = evaluate_constraints(constraints, batch, spec)
    probes = evaluate_batch_probes([ChannelMeanProbe(), ChannelRMSProbe()], batch, spec)
    if evaluate_batch_probes([], batch, spec) != {}:
        raise RuntimeError("disabled probes must produce an empty result")

    with tempfile.TemporaryDirectory(prefix="jka-v0-2-") as temporary:
        temporary_root = Path(temporary)
        checkpoint_path = temporary_root / "checkpoint.pt"
        save_checkpoint(
            Checkpoint(
                train_stage=config.training.stage,
                epoch=0,
                global_step=0,
                normalizer_state=normalizer.state_dict(),
                problem_spec=spec,
                config=config,
                data_fingerprint=fingerprint,
                split_manifest=manifest.to_dict(),
                physics_constraint_spec=constraint_spec,
            ),
            checkpoint_path,
        )
        restored = load_checkpoint(checkpoint_path)
        if restored.data_fingerprint != fingerprint:
            raise RuntimeError("checkpoint fingerprint round-trip failed")
        if restored.normalizer_state is None:
            raise RuntimeError("checkpoint lost normalizer state")
        restored_normalizer = ChannelStandardizer()
        restored_normalizer.load_state_dict(restored.normalizer_state)
        torch.testing.assert_close(
            restored_normalizer.transform(batch.context_states_raw),
            batch.context_states_model,
        )

    summary = {
        "architecture_revision": config.architecture.revision,
        "project_version": config.project_version,
        "trajectories": len(records),
        "split_sizes": {
            "train": len(manifest.train),
            "validation": len(manifest.validation),
            "test": len(manifest.test),
        },
        "split_window_counts": {
            split_name: len(split_dataset)
            for split_name, split_dataset in split_windows.items()
        },
        "context_states_raw_shape": list(batch.context_states_raw.shape),
        "future_states_raw_shape": list(batch.future_states_raw.shape),
        "history_dts_shape": list(batch.history_dts.shape),
        "future_dts_shape": list(batch.future_dts.shape),
        "first_future_transition": "U_t -> U_{t+1}",
        "normalization_scope": "TRAIN ONLY; frozen stats applied to train/validation/test",
        "normalizer_fit_ids": list(normalizer.fitted_trajectory_ids),
        "physics_terms": {name: float(value.detach()) for name, value in physics_terms.items()},
        "probe_shapes": {name: list(value.shape) for name, value in probes.items()},
        "data_fingerprint": fingerprint,
        "checkpoint_roundtrip": True,
        "normalizer_state_roundtrip": True,
        "all_probes_disabled_ok": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
