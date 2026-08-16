# Implementation checklist

- [x] Preserve latest validated V0.5 model and complete objective.
- [x] Add same-architecture online/EMA target encoders.
- [x] Keep Koopman matrix exponential as the only predictor.
- [x] Separate online Koopman targets from EMA JEPA targets.
- [x] Freeze target and exclude it from optimizer/inference.
- [x] Update EMA only after a successful optimizer update.
- [x] Save exact target, EMA schedule/count/tau and optimizer-update count.
- [x] Add V0.5 legacy initialization and strict V0.6 resume paths.
- [x] Add collapse, tracking, spectrum and near-identity diagnostics.
- [x] Add matched no-JEPA and three-seed GPU workflow.
- [x] Pass local unit/regression tests and end-to-end smoke.
- [ ] Run remote FP32/AMP and three-seed matched full validation.
- [ ] Review results and declare final scientific acceptance.
