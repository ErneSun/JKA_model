# Mori–Zwanzig semantics

Projection of unresolved dynamics can produce a generalized Langevin structure containing a Markov term, history-dependent memory, and an orthogonal/noise contribution. This is secondary motivation for the operational residual learnability and history-sweep experiment; it is not the primary claim.

V0.7 does **not** derive a projection operator, orthogonal dynamics, convolution kernel, fluctuation–dissipation relation, or exact Mori–Zwanzig equation. The fixed-history MLP is therefore described only as “Mori–Zwanzig-inspired closure.”

Operational memory labels:

- `MARKOVIAN`: H>1 has no material joint gain over H=1 and controls;
- `SHORT_MEMORY`: ordered history helps consistently and plateaus within the tested range;
- `LONG_MEMORY_CANDIDATE`: reliable gain persists at maximum tested H without plateau;
- `INCONCLUSIVE`: gains or controls disagree across seeds/metrics.

Residual learnability and closed-loop utility have separate labels. Closed-loop evidence has priority over ACF. All labels describe this dataset/checkpoint pair, not a universal property of Koopman models.
