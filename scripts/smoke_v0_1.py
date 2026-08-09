#!/usr/bin/env python3
"""CPU-only end-to-end smoke test for V0.1 contracts."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from jka_model.config import load_config, save_config
from jka_model.contracts import (
    BoundarySpec,
    ChannelSpec,
    DtMode,
    GeometrySpec,
    GridSpec,
    NormalizationSpec,
    ProblemBatch,
    ProblemSpec,
)
from jka_model.physics import PhysicsConstraint
from jka_model.utils import (
    Checkpoint,
    capture_rng_state,
    create_run_directory,
    get_git_commit,
    load_checkpoint,
    save_checkpoint,
    set_global_seed,
)


class SmokeConstraint:
    """Toy Protocol implementation used only to validate the V0.1 interface."""

    def loss(
        self,
        pred_state_raw: Tensor,
        *,
        prev_state_raw: Tensor | None = None,
        action: Tensor | None = None,
        dt: Tensor | None = None,
        spec: ProblemSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Tensor]:
        del prev_state_raw, action, dt, spec, metadata
        return {"interface_zero": pred_state_raw.sum() * 0.0}


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config = load_config(repository_root / "configs" / "v0_1_smoke.yaml")
    set_global_seed(config.training.seed, deterministic=config.training.deterministic)

    spec = ProblemSpec(
        name="toy_scalar_field",
        channels=(ChannelSpec("temperature", "K"),),
        spatial_dim=1,
        grid=GridSpec(
            shape=(4,),
            spacing=(0.25,),
            layout="channels_first",
            cell_weights_required=True,
        ),
        boundary=BoundarySpec("periodic"),
        action_dim=1,
        parameter_dim=1,
        dt_mode=DtMode.CONSTANT,
        constant_dt=0.1,
        normalization=NormalizationSpec("standardize_from_train_split"),
        geometry=GeometrySpec(mask_required=False),
        observable_requirements=("temperature_mean",),
    )

    batch_size, history, horizon = 2, 3, 2
    states_raw = 273.15 + torch.rand(batch_size, history + horizon, 1, 4)
    states_model = (states_raw - 273.15) / 10.0
    batch = ProblemBatch(
        context_states_raw=states_raw[:, :history],
        future_states_raw=states_raw[:, history:],
        context_states_model=states_model[:, :history],
        future_states_model=states_model[:, history:],
        history_actions=torch.zeros(batch_size, history - 1, 1),
        future_actions=torch.zeros(batch_size, horizon, 1),
        history_dts=torch.full((batch_size, history - 1), 0.1),
        future_dts=torch.full((batch_size, horizon), 0.1),
        mu_static=torch.ones(batch_size, 1),
        cell_weights=torch.full((4,), 0.25),
        trajectory_id=["toy-0", "toy-1"],
    )
    physics_constraint = SmokeConstraint()
    if not isinstance(physics_constraint, PhysicsConstraint):
        raise RuntimeError("PhysicsConstraint Protocol check failed")
    physics_terms = physics_constraint.loss(batch.future_states_raw[:, 0], spec=spec)

    with tempfile.TemporaryDirectory(prefix="jka-v0-1-") as temporary_root:
        run = create_run_directory(
            temporary_root,
            seed=config.training.seed,
            config_hash=config.stable_hash,
            train_stage=config.training.stage,
            git_commit=get_git_commit(repository_root),
            run_id="smoke-v0-1",
        )
        resolved_config_path = run.run_dir / "resolved_config.yaml"
        save_config(config, resolved_config_path)

        checkpoint_path = run.run_dir / "checkpoint.pt"
        checkpoint = Checkpoint(
            train_stage=config.training.stage,
            epoch=0,
            global_step=0,
            online_model_state=None,
            target_model_state=None,
            optimizer_state=None,
            scheduler_state=None,
            rng_state=capture_rng_state(),
            normalizer_state={"mean": [273.15], "scale": [10.0]},
            problem_spec=spec,
            config=config,
            data_fingerprint="toy-v0.1-deterministic",
            split_manifest={"train": ["toy-0"], "validation": ["toy-1"], "test": []},
            git_commit=run.git_commit,
        )
        save_checkpoint(checkpoint, checkpoint_path)
        restored = load_checkpoint(checkpoint_path)
        if restored.problem_spec != spec or restored.config != config:
            raise RuntimeError("checkpoint metadata round-trip failed")

        metadata = {
            "run_id": run.run_id,
            "project_version": run.project_version,
            "architecture_revision": run.architecture_revision,
            "seed": run.seed,
            "config_hash": run.config_hash,
            "git_commit": run.git_commit,
            "train_stage": run.train_stage.value,
            "context_states_raw_shape": list(batch.context_states_raw.shape),
            "future_states_raw_shape": list(batch.future_states_raw.shape),
            "context_states_model_shape": list(batch.context_states_model.shape),
            "history_actions_shape": (
                None if batch.history_actions is None else list(batch.history_actions.shape)
            ),
            "future_dts_shape": list(batch.future_dts.shape),
            "checkpoint_roundtrip": True,
            "physics_constraint_interface": set(physics_terms) == {"interface_zero"},
        }
        print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
