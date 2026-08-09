# Changelog

## 0.2.0 — Data Windows & Physics Contracts

- Added strict trajectory records, trajectory-level deterministic splits, stable data
  fingerprinting, and train-only channel normalization.
- Added trajectory-safe window datasets and canonical `ProblemBatch` collation.
- Added deterministic analytic 1D periodic advection-diffusion data with constant/variable dt.
- Added finite-value, admissibility, periodic-boundary, mass-conservation, and discrete PDE
  residual constraints, plus an explicit registry and optional physical probes.
- Added checkpoint schema 2 metadata for constraint specifications and the complete V0.2
  smoke/explanation/test workflow.
- Removed deprecated `ProblemBatch` aliases so v2.2 canonical names are the sole public API.
- Preserved architecture revision 2.2 and added no V0.3 model functionality.

## 0.1.0 — Architecture migration to revision 2.2

- Removed the legacy physical-latent field from `LatentState`; core state is now `z_k` plus
  optional history-conditioned `z_r`.
- Added the minimal non-latent `PhysicsConstraint` Protocol without implementing physics loss.
- Replaced the legacy train-stage readout reservation with `training_decoder`.
- Updated config, checkpoint guards, logging metadata, smoke output, tests, and documentation
  to architecture revision 2.2.
- Kept all V0.1 data alignment, reproducibility, checkpoint, and configuration behavior intact.

## 0.1.0 — V0.1 Project Skeleton & Contracts

- Added installable `jka_model` source package and Python project metadata.
- Added `ProblemSpec`, `ProblemBatch`, latent naming, and transition contracts.
- Added strict architecture/training/data configuration with stable hashing.
- Added stage-aware parameter ownership, reproducibility, checkpoint, and run logging APIs.
- Added CPU unit tests and `scripts/smoke_v0_1.py`.
- Explicitly did not implement V0.2+ data, physics, model, training, or rollout features.
