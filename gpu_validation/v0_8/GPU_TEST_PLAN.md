# GPU test plan

- G0: only the three converged V0.8 targeted test files;
- G1: three offline physical datasets plus coarse/nominal grid adequacy;
- G2: three new V0.6-compatible cylinder JEPA–Koopman backbones and acceptance;
- G3: fingerprinted frozen-online latent/residual caches;
- G4: full inherited V0.7 3-seed/3-init/multi-H R1/R2/R3 route matrix;
- G5: route-dependent V0.8 candidate training (`R1/INCONCLUSIVE` stops normally);
- G6: validation-selected family opens the locked test exactly once per 3×3 seed pair;
- G7: teacher-free rollout, physical non-inferiority, nested aggregation and compact report.

The workflow emits `completion.json` only after every route-required stage completes. Any exception emits
`failure.json` with stage, run, completed counts, expected work, last checkpoint and git commit.

