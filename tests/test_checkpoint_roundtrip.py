from __future__ import annotations

import pytest
import torch

from jka_model.config import ProjectConfig, load_config
from jka_model.constants import ARCHITECTURE_REVISION, CHECKPOINT_SCHEMA_VERSION
from jka_model.contracts import ProblemSpec
from jka_model.training import TrainStage
from jka_model.utils import Checkpoint, capture_rng_state, load_checkpoint, save_checkpoint


def make_checkpoint(toy_config: ProjectConfig, toy_problem_spec: ProblemSpec) -> Checkpoint:
    return Checkpoint(
        train_stage=TrainStage.KOOPMAN,
        epoch=3,
        global_step=42,
        online_model_state={"weight": torch.tensor([1.0, 2.0])},
        target_model_state=None,
        optimizer_state={"state": {}, "param_groups": []},
        scheduler_state=None,
        rng_state=capture_rng_state(),
        normalizer_state={"mean": [1.0, 2.0], "scale": [0.5, 0.25]},
        problem_spec=toy_problem_spec,
        config=toy_config,
        data_fingerprint="sha256:test-data",
        split_manifest={"train": ["a"], "validation": ["b"], "test": ["c"]},
        physics_constraint_spec=[{"name": "finite_values", "parameters": {}}],
        git_commit=None,
    )


def test_checkpoint_roundtrip(
    tmp_path, toy_config: ProjectConfig, toy_problem_spec: ProblemSpec
) -> None:
    checkpoint = make_checkpoint(toy_config, toy_problem_spec)
    destination = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, destination)
    restored = load_checkpoint(destination)
    assert restored.train_stage is TrainStage.KOOPMAN
    assert restored.epoch == 3
    assert restored.global_step == 42
    assert restored.problem_spec == toy_problem_spec
    assert restored.config == toy_config
    assert restored.config_hash == toy_config.stable_hash
    assert restored.data_fingerprint == "sha256:test-data"
    assert restored.physics_constraint_spec == [{"name": "finite_values", "parameters": {}}]
    assert restored.online_model_state is not None
    torch.testing.assert_close(restored.online_model_state["weight"], torch.tensor([1.0, 2.0]))


def test_architecture_revision_guard(
    tmp_path, toy_config: ProjectConfig, toy_problem_spec: ProblemSpec
) -> None:
    destination = tmp_path / "checkpoint.pt"
    save_checkpoint(make_checkpoint(toy_config, toy_problem_spec), destination)
    payload = torch.load(destination, weights_only=False)
    payload["architecture_revision"] = "2.1"
    torch.save(payload, destination)
    with pytest.raises(ValueError, match="architecture revision"):
        load_checkpoint(destination)
    assert ARCHITECTURE_REVISION == "2.2"


def test_checkpoint_rejects_config_hash_tampering(
    tmp_path, toy_config: ProjectConfig, toy_problem_spec: ProblemSpec
) -> None:
    destination = tmp_path / "checkpoint.pt"
    save_checkpoint(make_checkpoint(toy_config, toy_problem_spec), destination)
    payload = torch.load(destination, weights_only=False)
    payload["config_hash"] = "bad-hash"
    torch.save(payload, destination)
    with pytest.raises(ValueError, match="config_hash"):
        load_checkpoint(destination)


def test_schema_five_v0_5_checkpoint_migrates_losslessly(tmp_path) -> None:
    config = load_config("configs/v0_5/advection_diffusion_2d_cpu_smoke.yaml")
    current = Checkpoint(
        train_stage=TrainStage.KOOPMAN,
        epoch=2,
        global_step=7,
        online_model_state={"weight": torch.tensor([3.0])},
        config=config,
    )
    payload = current.to_payload()
    payload["schema_version"] = 5
    payload["project_version"] = "0.5.0"
    payload.pop("ema_state")
    payload.pop("optimizer_update_step")
    destination = tmp_path / "legacy-v05.pt"
    torch.save(payload, destination)
    restored = load_checkpoint(destination)
    assert restored.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert restored.config == config
    assert restored.config_hash == config.stable_hash
    assert restored.ema_state is None
    assert restored.optimizer_update_step == 0
    torch.testing.assert_close(restored.online_model_state["weight"], torch.tensor([3.0]))
