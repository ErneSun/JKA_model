"""Collision-safe versioned experiment IDs for V0.7 and later workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_VALID_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_REVISION = re.compile(r"^(?P<base>.+)-r(?P<revision>[1-9][0-9]*)$")


@dataclass(frozen=True, slots=True)
class VersionedSession:
    requested_id: str
    resolved_id: str
    path: Path
    revision: int


def create_versioned_session(root: str | Path, requested_id: str) -> VersionedSession:
    """Atomically create ``id`` or the first free ``id-rN`` directory."""
    if not _VALID_ID.fullmatch(requested_id):
        raise ValueError(
            "validation id may contain only letters, digits, dots, dashes, underscores"
        )
    match = _REVISION.fullmatch(requested_id)
    base = match.group("base") if match else requested_id
    start_revision = int(match.group("revision")) if match else 0
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    revision = start_revision
    while True:
        candidate = base if revision == 0 else f"{base}-r{revision}"
        path = root_path / candidate
        try:
            path.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            revision += 1
            continue
        return VersionedSession(requested_id, candidate, path.resolve(), revision)
