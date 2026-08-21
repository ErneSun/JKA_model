# Changelog

## 0.9.0 — Context-conditioned low-rank adaptive Koopman dynamics

- Added smooth and abrupt time-varying cylinder-wake conditions with trajectory-level
  stratification, physical/numerical dataset gates, and explicit condition visibility.
- Kept the accepted V0.8 context and backbone frozen while training only
  `A_t = A_0 + U diag(eta_t) V^T`, initialized to recover the nominal operator exactly.
- Added known-condition and history-only latent-condition variants, rank selection, operator
  burden/smoothness/stability regularization, exact-resume schema-9 checkpoints, and teacher-free
  closed-loop rollout.
- Added nominal, static-update, and shuffled-history controls; operator residual decomposition;
  nested 3-backbone by 3-initialization aggregation; and auditable compact reports.
- Added a one-command RTX-5080 workflow with revision-safe run IDs. Formal GPU scientific
  acceptance remains pending execution.

## 0.8.0 validation hardening

- Enforced the configured context effective-rank threshold and the R3 shuffled-history control.
- Added adequacy calibration, per-horizon and longest-horizon rollout decisions, and long-window
  shedding-frequency non-inferiority.
- Made V0.9 readiness require joint evidence within the same backbone seed before nested
  aggregation and all three backbone seeds, preventing unrelated or marginal seed-wise passes
  from being combined.
- Expanded compact results with physical/grid/backbone/route audit evidence, candidate summaries,
  selected-family training curves, numerical summaries, and horizon-resolved reporting.

## 0.8.0 — Residual-supervised dynamic context on transient cylinder wake

- Added a fixed-condition transient 2-D cylinder-wake Problem Adapter and reusable offline
  D2Q9-BGK datasets with physical/numerical acceptance and grid-adequacy gates.
- Reused the V0.6 JEPA–Koopman architecture with problem-owned nonperiodic CNN padding and
  trained new cylinder-specific backbones rather than transferring periodic-flow weights.
- Reused the updated V0.7 R1/R2/R3 assessment on the new problem; low residual magnitude remains
  diagnostic and cannot discard residuals, and R0 is not a legal route.
- Added a unified compact context interface: instantaneous MLP for R2 and validation-selected
  history MLP or causal Attention for R3, with standardized residual and adequacy teacher heads.
- Kept the encoder, decoder, EMA target and nominal generator A0 frozen; no eta, adaptive A,
  persistent residual state, or V0.9 functionality is implemented.
- Added exact-resume schema-8 context checkpoints, leakage-free histories, teacher-free rollout,
  physical non-inferiority checks, nested 3x3 aggregation, ten auditable figures, and one-command
  RTX-5080 validation. Formal GPU scientific acceptance remains pending execution.

## 0.6.0 — JEPA online/EMA-target over the validated V0.5 Koopman model

- Added a frozen EMA target encoder while retaining the complete V0.5 online model,
  objective and raw-space physics constraints.
- Added strict online-Koopman versus EMA-JEPA target separation.
- Added optimizer-step-indexed EMA, schema-6 exact resume state, collapse/tracking
  diagnostics, matched no-JEPA controls, tests and a one-command GPU workflow.
- Local implementation passes; GPU and scientific acceptance remain pending.

## 0.5.0 — 2-D PDE Koopman and CPU/GPU validation workflow

- Added analytic endpoint-free periodic 2-D advection-diffusion data and a problem adapter.
- Added a circular CNN field encoder/decoder around the exact continuous-time core.
- Added differentiable raw-unit mass and trapezoidal PDE-operator constraints with warmup.
- Added canonical train/evaluate APIs, complete run records, exact resume, and CSV history.
- Fixed device-neutral normalizer and RNG restoration for CUDA checkpoint resume.
- Added CPU numerical/integration tests and a thin-wrapper GPU validation package.
- GPU validation is not run locally; scientific status remains `PENDING_GPU`.

## 0.4.0 — Learned Koopman Coordinates

- Added deterministic known-latent rotation-decay data with nonlinear five-channel observations
  and evaluation-only true latent trajectories.
- Added a trainable `KoopmanEncoder`, two-hidden-layer Tanh `TrainingDecoder`, and thin
  `KoopmanAutoencoder` composition while reusing the unchanged V0.3 continuous-time core.
- Added online-encoder one-step consistency, closed-loop multi-step consistency, model-space
  reconstruction, population-variance anti-collapse, and optional stability losses.
- Added train-fit/test-apply affine latent alignment, non-collapse statistics, reconstruction,
  decoded rollout, persistence, and similarity-invariant spectrum diagnostics.
- Added interval-based training loss/latent-statistic snapshots, explicit held-out one-step and
  multi-step latent errors, strict V0.4 cross-section config validation, and checkpoint-type-aware
  latent analysis discovery.
- Added actual reconstruction-off and multi-step-off ablations plus a secondary learned-lifting
  Duffing comparison.
- Added complete learned-model checkpoint round-trip, mandatory unit/integration coverage,
  23-step smoke, 17-step teaching script, latent analyzer, walkthrough, and checklist.
- Updated project version to 0.4.0 and checkpoint schema to 4; architecture remains revision 2.2.
- Added no JEPA, target/EMA encoder, residual closure, attention, PDE training, physics loss, or
  action conditioning.

## 0.3.0 — Direct-State Continuous-Time KoopmanCore

- Added fixed/trainable continuous generator `A` with exact `torch.matrix_exp` propagation.
- Added scalar/batch/variable-dt step and closed-loop rollout APIs with column-state convention.
- Added detached continuous spectrum diagnostics and persistence/rollout metrics.
- Added reusable relative-frequency and spectral-growth metrics.
- Added independent analytical damped-oscillator data and reference-only Duffing RK4 data.
- Reused deterministic trajectory splits and structured run logging for held-out evaluation.
- Added minimal direct-state matrix-exponential identification that trains only `A`.
- Added 100-step oscillator evaluation, checkpoint reload, spectrum analyzer, smoke and teaching
  scripts, mandatory mathematical tests, walkthrough, and completion checklist.
- Updated project version to 0.3.0 and checkpoint schema to 3; architecture remains revision 2.2.
- Added no encoder, learned lifting, JEPA, residual, attention, or action conditioning.

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
