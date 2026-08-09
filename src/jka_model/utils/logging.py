"""Lightweight structured run directory and Python logging setup."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from jka_model.constants import ARCHITECTURE_REVISION, PROJECT_VERSION
from jka_model.training import TrainStage


@dataclass(frozen=True, slots=True)
class RunContext:
    """Metadata identifying one reproducible run."""

    run_id: str
    run_dir: Path
    project_version: str
    architecture_revision: str
    seed: int
    config_hash: str
    git_commit: str | None
    train_stage: TrainStage

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["run_dir"] = str(self.run_dir)
        data["train_stage"] = self.train_stage.value
        return data


def get_git_commit(repository_root: str | Path) -> str | None:
    """Return ``HEAD`` or ``None`` when the workspace is not a Git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repository_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def create_run_directory(
    root: str | Path,
    *,
    seed: int,
    config_hash: str,
    train_stage: TrainStage,
    git_commit: str | None = None,
    run_id: str | None = None,
) -> RunContext:
    """Create a run directory, metadata JSON, and a scoped file logger."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not config_hash:
        raise ValueError("config_hash must not be empty")
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}-{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id may contain only letters, digits, dots, dashes, and underscores")

    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    context = RunContext(
        run_id=run_id,
        run_dir=run_dir.resolve(),
        project_version=PROJECT_VERSION,
        architecture_revision=ARCHITECTURE_REVISION,
        seed=seed,
        config_hash=config_hash,
        git_commit=git_commit,
        train_stage=train_stage,
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(context.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger = logging.getLogger(f"jka_model.run.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.info("run_started metadata=%s", json.dumps(context.to_dict(), sort_keys=True))
    handler.flush()
    return context

