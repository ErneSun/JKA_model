# Latest technology review (reviewed 2026-08-18)

## Adopt now

- Frozen-backbone residual identification: isolates missing latent dynamics from representation retraining.
- Direct post-Koopman Δz prediction with variable dt.
- Explicit `H=[1,2,4,8,16]` fixed-delay sweep plus parameter-matched instantaneous and shuffled-history controls.
- Closed-loop evaluation and cache/checkpoint provenance.
- Effective-horizon/plateau inference with closed-loop evidence prioritized over ACF.
- Per-dimension train-RMS standardized residual regression and zero-output closure initialization.
- Markovian and finite-history probes with shuffled-history and parameter-matched controls.
- Hierarchical closure-init/backbone validation and the S_R/P_R/G_H Residual Structure Assessment.

This is consistent with the motivation of latent Koopman memory closure in Gupta et al. (2025), while deliberately avoiding their joint autoencoder/operator training and recurrent memory models. That published work motivates testing non-Markovian latent corrections; it does not establish that the residual in this repository is history-dependent, so V0.7 still requires the R0-R3 evidence route.

## Reviewed but deferred

- Neural DDEs: mature evidence supports delay closures and partial observation, but continuous-delay solvers and learned delays add a new numerical contract.
- RNN/GRU/LSTM: established memory models, but hidden recurrent state would prevent V0.7 from first identifying whether simple finite history is useful.
- Mamba-assisted closure: promising constant-step inference and sequence training, but the June 2026 work is a new preprint and much larger than the minimal test needed here.
- Attention-free Koopman memory/re-encoding: promising long-rollout results, but its correction-before-Koopman and dynamic re-encoding differ from V0.7’s frozen post-step residual contract.

The 2026 Mamba-assisted and attention-free Koopman works remain recent preprints. They are useful V0.8+ candidates only if V0.7 reaches a physically accepted R3 result; R0, R1, or R2 does not justify adopting them.

## Rejected for V0.7

Joint fine-tuning, a new residual latent state, attention, stochastic forcing, physics-loss closure training, or claims of an exact Mori–Zwanzig kernel. Any later adoption requires V0.7 evidence and a new versioned mathematical review.
