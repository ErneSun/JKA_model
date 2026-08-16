# Evaluation

Teacher-forced evaluation predicts cached one-step residuals and reports MSE, normalized RMSE, global/per-dimension R², and cosine similarity. It is not a forecast result.

Closed-loop evaluation receives true latent history only at initialization. Every later history state is its own prediction. For every configured horizon it reports latent/raw-field error curves, relative L2, relative mass drift, spectral-step operator MSE, and the per-step burden `||r_hat||/(||r_hat||+||delta_z_base||+eps)`.

Primary comparisons use the same V0.6 checkpoint, data, split, normalizer, and evaluation trajectories:

1. zero closure;
2. linear closure;
3. parameter-matched instantaneous MLP;
4. ordered fixed-history MLP;
5. shuffled-history MLP.

The sweep reports three independent conclusions:

- residual learnability: `STRONG / MODERATE / WEAK / NONE`;
- closed-loop utility: `POSITIVE / NEUTRAL / NEGATIVE`;
- memory class: `MARKOVIAN / SHORT_MEMORY / LONG_MEMORY_CANDIDATE / INCONCLUSIVE`.

Ordered history counts as memory evidence only when teacher-forced and closed-loop gains are material, consistent across seeds, and survive both controls. ACF remains auxiliary. The effective horizon is the shortest H reaching 95% of the observed joint gain; physical time is reported with the step count.
