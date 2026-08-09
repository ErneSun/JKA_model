from __future__ import annotations

import pytest
import torch

from jka_model.config import ProjectConfig
from jka_model.constants import ARCHITECTURE_REVISION
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
    assert restored.online_model_state is not None
    torch.testing.assert_close(restored.online_model_state["weight"], torch.tensor([1.0, 2.0]))


def test_architecture_revision_guard(
    tmp_path, toy_config: ProjectConfig, toy_problem_spec: ProblemSpec
) -> None:
    destination = tmp_path / "checkpoint.pt"
    save_checkpoint(make_checkpoint(toy_config, toy_problem_spec), destination)
    payload = torch.load(destination, weights_only=False)
    payload["architecture_revision"] = "1.0"
    torch.save(payload, destination)
    with pytest.raises(ValueError, match="architecture revision"):
        load_checkpoint(destination)
    assert ARCHITECTURE_REVISION == "2.1"


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
