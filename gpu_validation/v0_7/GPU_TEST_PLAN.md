# GPU test plan

1. Run only the new V0.7 software tests.
2. Verify CUDA and BF16 capability.
3. Discover compatible validated V0.6 JEPA checkpoints for seeds 47/53/59.
4. Regenerate data and build a fingerprinted residual cache per seed.
5. For each seed, sweep `H=[1,2,4,8,16]`; train ordered-history and parameter-matched instantaneous closures at every H, plus shuffled history for H>1. Train zero/linear controls at H=1.
6. Evaluate teacher-forced residual prediction separately from closed-loop field/physics metrics and record error/burden curves.
7. Validate checkpoint/data/split/normalizer/evaluation-trajectory identity before comparison.
8. Report residual learnability, closed-loop utility, and memory class separately; write CSV/JSON/plots/Markdown reports.

The synthetic latent-memory problem is a mechanism diagnostic only and cannot produce scientific acceptance.
