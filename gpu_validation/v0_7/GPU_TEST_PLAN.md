# GPU test plan

1. Run only the new V0.7 software tests.
2. Verify CUDA and BF16 capability.
3. Discover compatible validated V0.6 JEPA checkpoints for seeds 47/53/59.
4. Regenerate data and build a fingerprinted residual cache per seed.
5. For each backbone/data seed, repeat closure initialization seeds `[101,211,307]`, then sweep `H=[1,2,4,8,16]`; train ordered-history and parameter-matched instantaneous closures at every H, plus shuffled history for H>1. Train zero/linear controls at H=1.
6. Compute validation residual significance, Markovian predictability, and conditional history gain; lock preliminary R0/R1/R2/R3/INCONCLUSIVE routing without test selection.
7. Validate checkpoint/data/split/normalizer/evaluation-trajectory identity before comparison.
8. Confirm the locked route on test, keep closed-loop utility and memory class separate, apply all three physics gates, and write primary assessment JSON, CSV, seven plots, Markdown reports, and completion proof.

The synthetic latent-memory problem is a mechanism diagnostic only and cannot produce scientific acceptance.
