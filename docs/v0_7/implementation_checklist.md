# Implementation checklist

- [x] V0.6 online encoder, Koopman A, decoder, and EMA target frozen.
- [x] `TrainStage.RESIDUAL` owns closure parameters only.
- [x] Residual uses online encoder and exact `exp(A dt)`; target encoder is forbidden.
- [x] Cache records checkpoint/config/data/split/normalizer fingerprints.
- [x] Variable next-dt enters every learned closure.
- [x] Zero, linear, instantaneous, history, shuffled-history variants exist.
- [x] Instantaneous MLP parameter count is matched as closely as integer width permits.
- [x] H=1 exactly matches the current-state Markovian information set.
- [x] Configured H sweep has at least four levels and formal GPU levels `[1,2,4,8,16]`.
- [x] Comparison-only script validates provenance and never retrains.
- [x] Formal rollout feeds predicted history after the initial context.
- [x] Teacher-forced and closed-loop metrics are separate.
- [x] Standalone schema-7 checkpoints contain backbone and closure.
- [x] New V0.7 tests and CPU end-to-end smoke pass.
- [x] GPU workflow is non-silent and writes automatic reports.
- [x] Results include sweep CSV, classification JSON, five diagnostic plots, and two reports.
- [ ] Multi-seed RTX 5080 measurement.
- [ ] Human review of scientific evidence.
