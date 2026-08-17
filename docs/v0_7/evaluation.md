# Evaluation

Teacher-forced evaluation predicts cached one-step residuals and reports MSE, normalized RMSE, global/per-dimension R², and cosine similarity separately on validation and test splits. Validation residual NRMSE alone selects the closure family and H; test metrics are read after selection and never used as an oracle. Teacher-forced evaluation is not a forecast result.

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

Ordered history counts as memory evidence only when teacher-forced and closed-loop gains are material, survive both controls, are consistent across closure-initialization repeats within a backbone/data seed, and are then consistent across backbone/data seeds. `H=1` history and instantaneous models must be training-identical under a paired seed. ACF remains auxiliary. The effective horizon is the shortest H reaching 95% of the observed joint gain; physical time is reported with the step count.

Closed-loop utility must improve the selected test rollout while satisfying both absolute V0.5 physical limits and baseline-relative non-inferiority. Mean closure burden must remain at or below `0.25`. Error bars aggregate backbone/data and closure-initialization variability.
