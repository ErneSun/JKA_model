# GPU test plan

1. Confirm commit/environment and CUDA availability.
2. Run all CPU/GPU-independent tests.
3. Run FP32 and AMP tiny V0.5→V0.6 smoke, including checkpoint reload.
4. Discover validated V0.5 checkpoints for seeds 47, 53 and 59.
5. For each seed, derive two configs differing only in JEPA weights/tags.
6. Train no-JEPA then JEPA from the exact same V0.5 checkpoint.
7. Evaluate online-only rollout, physics, spectrum, collapse and tracking diagnostics.
8. Gate each seed and retain mean/sample-standard-deviation summaries.
9. Human-review logs, plots and scientific scope before declaring acceptance.

Required retained evidence: resolved configs, init checkpoint paths, fingerprints,
splits, environment/git metadata, epoch/step/EMA logs, checkpoints, evaluation JSON,
per-seed comparisons and aggregate summary.
