# Testing

The convergent test policy applies: run new V0.7 tests and only directly affected compatibility tests during development; reserve the full historical suite for release/high-risk changes.

Covered contracts include exact residual formula, online-versus-EMA identity, stop-gradient, train-only standardization, raw/normalized metric separation, H=1 paired equivalence, zero-output initialization, no future leakage, frozen ownership, predicted-history rollout, RNG-free analytical parameter matching, shuffled controls, nested seed ownership, validation-only route selection, locked test confirmation, all R0–R3 plus inconclusive routes, three-condition physics AND, exact record/provenance checks, completion/failure schemas, checkpoint scale provenance, and CPU cache/train/evaluate integration.

Local command:

```bash
python -m pytest -q tests/test_v0_7_residual.py tests/test_v0_7_gpu_workflow.py tests/test_v0_7_integration.py
```

GPU validation emits `START/PASS/FAIL` for each major stage. It does not print every formal GPU epoch; epoch data are retained under each variant’s `logs/epoch_metrics.csv`.

The formal three-backbone, three-closure-initialization sweep creates 144 evaluation records. `completion.json` is written only after comparison and report copying succeed; `failure.json` marks an incomplete session.
