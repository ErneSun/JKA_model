# Evaluation

Teacher-forced evaluation reports raw MSE/RMS, training-RMS-standardized MSE, NRMSE, global/per-dimension R², and cosine similarity. Validation selects closure family, H, history/no-history decision, and preliminary route. Test only confirms the locked validation decision and cannot change which configuration is selected.

Closed-loop evaluation receives true latent history only at initialization. Every later history state is its own prediction. For every configured horizon it reports latent/raw-field error curves, relative L2, relative mass drift, spectral-step operator MSE, and the per-step burden `||r_hat||/(||r_hat||+||delta_z_base||+eps)`.

Primary comparisons use the same V0.6 checkpoint, data, split, normalizer, and evaluation trajectories:

1. zero closure;
2. linear closure;
3. parameter-matched instantaneous MLP;
4. ordered fixed-history MLP;
5. shuffled-history MLP.

The primary assessment reports:

- residual significance: `NEGLIGIBLE / NON_NEGLIGIBLE / INCONCLUSIVE`;
- residual learnability: `STRONG / MODERATE / WEAK / NONE`;
- conditional history gain: `PRESENT / ABSENT / INCONCLUSIVE`;
- residual route: `R0 / R1 / R2 / R3 / INCONCLUSIVE`.

Independent secondary conclusions remain:

- residual learnability: `STRONG / MODERATE / WEAK / NONE`;
- closed-loop utility: `POSITIVE / NEUTRAL / NEGATIVE`;
- memory class: `MARKOVIAN / SHORT_MEMORY / LONG_MEMORY_CANDIDATE / INCONCLUSIVE`.

Ordered history counts as memory evidence only when teacher-forced and closed-loop gains are material, survive both controls, are consistent across closure-initialization repeats within a backbone/data seed, and are then consistent across backbone/data seeds. `H=1` history and instantaneous models must be training-identical under a paired seed. ACF remains auxiliary. The effective horizon is the shortest H reaching 95% of the observed joint gain; physical time is reported with the step count.

Closed-loop utility must improve the selected test rollout while satisfying both absolute V0.5 physical limits and baseline-relative non-inferiority. Mean closure burden must remain at or below `0.25`. Error bars aggregate backbone/data and closure-initialization variability.

The significance threshold is config-owned: `S_R < 0.01` is the current R0 candidate rule, corresponding to less than 1% residual energy relative to the true latent increment. It is recorded in every resolved config and result rather than hidden in classifier code.
