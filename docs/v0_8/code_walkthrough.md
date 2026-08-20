# V0.8 code walkthrough

1. `CylinderWake2DProblemAdapter` owns data, geometry, fixed BC constraints and physical metadata.
2. `cylinder_wake_2d.py` generates or fingerprint-loads offline trajectories and applies the pre-ML gate.
3. Existing `train_v0_6` trains a new cylinder backbone; architecture semantics are reused, weights are not.
4. Existing V0.7 cache/train/evaluate/compare functions determine R1/R2/R3 without problem-name routing.
5. `ContextWindowDataset` exposes only causal latent history, aligned `dt`, static parameters and detached targets.
6. `DynamicContextModel` provides instantaneous/history/Attention encoders and unified residual/adequacy heads.
7. `train_v0_8` freezes inherited modules by operating solely on the residual cache, selects checkpoints on
   validation, writes exact-resume state every epoch and never opens test.
8. `evaluate_v0_8` opens locked test, runs ablations/shuffling and teacher-free closed-loop physical decoding.
9. `aggregate_v0_8_results` performs nested-seed aggregation and writes compact CSV/JSON/report/figures.
10. `gpu_validate_all.py` is the only formal orchestration entrypoint and delegates to canonical APIs.

