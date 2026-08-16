"""Version-scoped deterministic memory problem for closure-mechanism diagnostics."""

from __future__ import annotations

import torch

from jka_model.residual.cache import ResidualCache, ResidualTrajectory


def make_v0_7_synthetic_memory_cache(
    *,
    seed: int = 7,
    trajectories: int = 48,
    steps: int = 96,
    latent_dim: int = 4,
) -> ResidualCache:
    """Create the declared AR(3) latent diagnostic; never a scientific acceptance source."""
    if trajectories < 6 or steps < 8 or latent_dim < 1:
        raise ValueError("synthetic memory problem requires >=6 trajectories, >=8 steps")
    generator = torch.Generator().manual_seed(seed)
    cached: list[ResidualTrajectory] = []
    identifiers = [f"v0-7-synthetic-memory-{index:04d}" for index in range(trajectories)]
    train_end = int(0.7 * trajectories)
    validation_end = int(0.85 * trajectories)
    groups = {
        "train": identifiers[:train_end],
        "validation": identifiers[train_end:validation_end],
        "test": identifiers[validation_end:],
    }
    split_lookup = {identifier: split for split, values in groups.items() for identifier in values}
    for identifier in identifiers:
        latent = torch.empty(steps + 1, latent_dim)
        latent[:3] = torch.randn(3, latent_dim, generator=generator)
        for index in range(2, steps):
            latent[index + 1] = (
                0.92 * latent[index] + 0.18 * latent[index - 1] - 0.10 * latent[index - 2]
            )
        base = 0.92 * latent[:-1]
        residual = latent[1:] - base
        cached.append(
            ResidualTrajectory(
                trajectory_id=identifier,
                split=split_lookup[identifier],
                latents=latent,
                dts=torch.ones(steps),
                parameters=torch.empty(0),
                residuals=residual,
            )
        )
    split_manifest = {
        **groups,
        "seed": seed,
        "ratios": [0.7, 0.15, 0.15],
    }
    return ResidualCache(
        trajectories=tuple(cached),
        backbone_checkpoint_sha256="synthetic-diagnostic-no-checkpoint",
        backbone_config_hash="v0_7_synthetic_latent_memory",
        data_fingerprint="deterministic-ar3",
        split_manifest=split_manifest,
        normalizer_state={"kind": "identity", "problem_owner": "v0.7"},
    )
