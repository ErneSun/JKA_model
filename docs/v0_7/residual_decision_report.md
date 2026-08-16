# V0.7 residual decision report

This is the version-controlled interpretation contract. A completed GPU session writes the populated report to `gpu_validation/v0_7/results/<id>/reports/residual_decision_report.md`.

The report must answer separately:

1. residual learnability: `STRONG / MODERATE / WEAK / NONE`;
2. closed-loop utility: `POSITIVE / NEUTRAL / NEGATIVE`;
3. memory class: `MARKOVIAN / SHORT_MEMORY / LONG_MEMORY_CANDIDATE / INCONCLUSIVE`.

It must state effective H in steps and physical time, parameter-matched and shuffled-control outcomes, multi-seed consistency, physics non-inferiority, and the boundary that finite-history evidence is not an exact Mori–Zwanzig kernel.
