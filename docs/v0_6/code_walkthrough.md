# Code walkthrough

- `models/field_jepa_koopman.py`: online model, frozen target, hard sync, EMA update.
- `losses/field_jepa.py`: complete V0.5 objective plus isolated JEPA terms.
- `training/ema.py`: optimizer-step schedule and serializable EMA state.
- `train/train_v0_6.py`: V0.5 initialization, exact resume, AMP-aware step/EMA,
  records and checkpoints.
- `eval/evaluate_v0_6.py`: online-only rollout plus target/collapse diagnostics.
- `evaluation/v0_6_diagnostics.py`: latent, tracking and near-identity measurements.
- `scripts/smoke_v0_6.py`, `scripts/explain_v0_6.py`: local audit workflows.
- `gpu_validation/v0_6/scripts/gpu_validate_all.py`: one-command remote validation.

Schema 6 adds `ema_state` and `optimizer_update_step`. Schema-5 project-0.5 checkpoints
are accepted only by explicit initialization; V0.6 resume remains schema-, version-,
config-, split- and fingerprint-strict.
