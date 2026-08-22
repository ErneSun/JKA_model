"""Observable error attribution across data, representation, and operator levels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def observable_error_attribution(
    metrics_by_level: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Convert four-level metrics into auditable incremental contributions."""
    required = ("data", "reconstruction", "nominal", "adaptive")
    if any(level not in metrics_by_level for level in required):
        raise ValueError("error attribution requires data/reconstruction/nominal/adaptive levels")
    names = set(metrics_by_level["data"])
    if any(set(metrics_by_level[level]) != names for level in required):
        raise ValueError("error attribution metric sets must align")
    rows: list[dict[str, Any]] = []
    for name in sorted(names):
        data = float(metrics_by_level["data"][name])
        reconstruction = float(metrics_by_level["reconstruction"][name])
        nominal = float(metrics_by_level["nominal"][name])
        adaptive = float(metrics_by_level["adaptive"][name])
        rows.append(
            {
                "metric": name,
                "data_floor": data,
                "reconstruction": reconstruction,
                "nominal": nominal,
                "adaptive": adaptive,
                "representation_increment": reconstruction - data,
                "nominal_dynamics_increment": nominal - reconstruction,
                "adaptive_dynamics_increment": adaptive - nominal,
            }
        )
    return rows
