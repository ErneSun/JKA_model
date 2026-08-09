"""Deterministic trajectory-level splits."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jka_model.config import SplitConfig
from jka_model.data.datasets import TrajectoryRecord


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Serializable assignment of complete trajectories to disjoint splits."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed: int
    ratios: tuple[float, float, float]

    def __post_init__(self) -> None:
        groups = self.train + self.validation + self.test
        if not groups:
            raise ValueError("split manifest must contain at least one trajectory")
        if len(groups) != len(set(groups)):
            raise ValueError("split manifest trajectory IDs must be disjoint and unique")
        if abs(sum(self.ratios) - 1.0) > 1e-9:
            raise ValueError("split manifest ratios must sum to 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "seed": self.seed,
            "ratios": list(self.ratios),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SplitManifest:
        allowed = {"train", "validation", "test", "seed", "ratios"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown SplitManifest field(s): {', '.join(sorted(unknown))}")
        ratios = tuple(float(value) for value in data["ratios"])
        if len(ratios) != 3:
            raise ValueError("SplitManifest ratios must contain three entries")
        return cls(
            train=tuple(str(value) for value in data["train"]),
            validation=tuple(str(value) for value in data["validation"]),
            test=tuple(str(value) for value in data["test"]),
            seed=int(data["seed"]),
            ratios=(ratios[0], ratios[1], ratios[2]),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> SplitManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("split manifest payload must be a mapping")
        return cls.from_dict(payload)


def _apportion_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    order = sorted(range(3), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    positive = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if total >= len(positive):
        for empty in (index for index in positive if counts[index] == 0):
            donors = [index for index in positive if counts[index] > 1]
            if donors:
                donor = max(donors, key=lambda index: counts[index])
                counts[donor] -= 1
                counts[empty] += 1
    return counts[0], counts[1], counts[2]


def make_split_manifest(
    records_or_ids: Sequence[TrajectoryRecord] | Sequence[str], config: SplitConfig
) -> SplitManifest:
    """Split sorted identifiers with a local seeded RNG; input order has no effect."""
    identifiers = [
        item.trajectory_id if isinstance(item, TrajectoryRecord) else str(item)
        for item in records_or_ids
    ]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("split input IDs must be non-empty and unique")
    identifiers.sort()
    random.Random(config.seed).shuffle(identifiers)
    n_train, n_validation, _ = _apportion_counts(
        len(identifiers), (config.train, config.validation, config.test)
    )
    train_end = n_train
    validation_end = n_train + n_validation
    return SplitManifest(
        train=tuple(identifiers[:train_end]),
        validation=tuple(identifiers[train_end:validation_end]),
        test=tuple(identifiers[validation_end:]),
        seed=config.seed,
        ratios=(config.train, config.validation, config.test),
    )


def select_split(
    records: Sequence[TrajectoryRecord], manifest: SplitManifest, split: str
) -> tuple[TrajectoryRecord, ...]:
    """Return records in manifest order for one named split."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be 'train', 'validation', or 'test'")
    lookup = {record.trajectory_id: record for record in records}
    wanted = getattr(manifest, split)
    missing = set(wanted) - set(lookup)
    if missing:
        raise ValueError(f"manifest references missing trajectories: {', '.join(sorted(missing))}")
    return tuple(lookup[identifier] for identifier in wanted)
