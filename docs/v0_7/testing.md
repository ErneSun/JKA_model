# Testing

The convergent test policy applies: run new V0.7 tests and only directly affected compatibility tests during development; reserve the full historical suite for release/high-risk changes.

Covered contracts include exact residual formula, online-versus-EMA identity, stop-gradient, H=1 Markovian equivalence, window/next-dt alignment, no future leakage, frozen backbone, optimizer ownership, predicted-history rollout, parameter matching, shuffled-history current-state preservation, sweep provenance, classification schema, cache fingerprinting, schema-7 checkpoint, ID `-rN` allocation, and CPU cache/train/evaluate integration.

Local command:

```bash
python -m pytest -q tests/test_v0_7_residual.py tests/test_v0_7_gpu_workflow.py tests/test_v0_7_integration.py
```

GPU validation emits `START/PASS/FAIL` for each major stage. It does not print every formal GPU epoch; epoch data are retained under each variant’s `logs/epoch_metrics.csv`.
