# Testing

The convergent test policy applies: run new V0.7 tests and only directly affected compatibility tests during development; reserve the full historical suite for release/high-risk changes.

Covered contracts include exact residual formula, online-versus-EMA identity, stop-gradient, H=1 Markovian equivalence and paired initialization, zero-output initialization, window/next-dt alignment, no future leakage, frozen backbone, optimizer ownership, predicted-history rollout, parameter matching without RNG consumption, shuffled-history current-state preservation, validation-only selection, strict physics rejection, multi-initialization sweep provenance, classification schema, cache fingerprinting, schema-7 checkpoint, ID `-rN` allocation, and CPU cache/train/evaluate integration.

Local command:

```bash
python -m pytest -q tests/test_v0_7_residual.py tests/test_v0_7_gpu_workflow.py tests/test_v0_7_integration.py
```

GPU validation emits `START/PASS/FAIL` for each major stage. It does not print every formal GPU epoch; epoch data are retained under each variant’s `logs/epoch_metrics.csv`.

The formal three-backbone, three-closure-initialization sweep creates 144 evaluation records. `completion.json` is written only after comparison and report copying succeed; `failure.json` marks an incomplete session.
