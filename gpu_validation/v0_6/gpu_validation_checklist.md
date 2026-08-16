# GPU validation checklist

- [x] Correct training commit and recorded git state.
- [x] Seed-47/53/59 validated V0.5 checkpoints discovered.
- [x] Test suite passes.
- [x] CUDA preflight passes.
- [x] FP32 smoke passes.
- [x] AMP smoke passes; skipped optimizer steps imply skipped EMA.
- [x] Matched configs differ only in JEPA objective/tags.
- [x] Each pair uses the same V0.5 checkpoint, fingerprint, split and normalizer.
- [x] Target has no gradients and is absent from optimizer/rollout.
- [x] EMA count equals successful optimizer-update count.
- [x] Collapse, rollout, physics, spectrum and efficiency artifacts exist.
- [x] Three-seed mean/std and per-seed gates exist.
- [x] Human review completed before scientific `PASS` is published.
