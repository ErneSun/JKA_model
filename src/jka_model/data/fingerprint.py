"""Stable fingerprints for static problem metadata and trajectory content."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from jka_model.contracts import ProblemSpec
from jka_model.data.datasets import TrajectoryRecord


def _update_tensor(digest: Any, name: str, tensor: Tensor | None, include_content: bool) -> None:
    digest.update(name.encode("utf-8"))
    if tensor is None:
        digest.update(b"none")
        return
    cpu = tensor.detach().cpu().contiguous()
    descriptor = {"shape": list(cpu.shape), "dtype": str(cpu.dtype)}
    digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if include_content:
        digest.update(cpu.view(torch.uint8).numpy().tobytes(order="C"))


def data_fingerprint(
    records: Sequence[TrajectoryRecord],
    spec: ProblemSpec,
    *,
    include_content: bool = True,
) -> str:
    """Return a deterministic SHA-256 independent of input record ordering."""
    identifiers = [record.trajectory_id for record in records]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("fingerprint records must have non-empty unique IDs")
    digest = hashlib.sha256()
    digest.update(json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for record in sorted(records, key=lambda item: item.trajectory_id):
        digest.update(record.trajectory_id.encode("utf-8"))
        try:
            metadata = json.dumps(
                dict(record.metadata),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"trajectory {record.trajectory_id!r} metadata must be JSON-serializable"
            ) from error
        digest.update(metadata.encode("utf-8"))
        for name in (
            "states_raw",
            "actions",
            "dts",
            "mu_static",
            "coordinates",
            "cell_weights",
            "valid_mask",
        ):
            _update_tensor(digest, name, getattr(record, name), include_content)
    return f"sha256:{digest.hexdigest()}"
