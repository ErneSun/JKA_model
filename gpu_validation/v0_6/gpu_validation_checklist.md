# GPU validation checklist

- [ ] Correct commit and clean/recorded git state.
- [ ] Seed-47/53/59 validated V0.5 checkpoints discovered.
- [ ] Test suite passes.
- [ ] CUDA preflight passes.
- [ ] FP32 smoke passes.
- [ ] AMP smoke passes; skipped optimizer steps imply skipped EMA.
- [ ] Matched configs differ only in JEPA objective/tags.
- [ ] Each pair uses the same V0.5 checkpoint, fingerprint, split and normalizer.
- [ ] Target has no gradients and is absent from optimizer/rollout.
- [ ] EMA count equals successful optimizer-update count.
- [ ] Collapse, rollout, physics, spectrum and efficiency artifacts exist.
- [ ] Three-seed mean/std and per-seed gates exist.
- [ ] Human review completed before scientific `PASS` is published.
