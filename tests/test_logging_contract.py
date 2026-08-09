from __future__ import annotations

import json

from jka_model.constants import ARCHITECTURE_REVISION, PROJECT_VERSION
from jka_model.training import TrainStage
from jka_model.utils import create_run_directory


def test_run_directory_records_required_metadata(tmp_path) -> None:
    run = create_run_directory(
        tmp_path,
        seed=9,
        config_hash="abc123",
        train_stage=TrainStage.KOOPMAN,
        git_commit=None,
        run_id="unit-test",
    )
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "unit-test"
    assert metadata["project_version"] == PROJECT_VERSION
    assert metadata["architecture_revision"] == ARCHITECTURE_REVISION
    assert metadata["seed"] == 9
    assert metadata["config_hash"] == "abc123"
    assert metadata["train_stage"] == "koopman"
    assert (run.run_dir / "run.log").exists()

