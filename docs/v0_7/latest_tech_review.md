# Latest technology review (reviewed 2026-08-16)

## Adopt now

- Frozen-backbone residual identification: isolates missing latent dynamics from representation retraining.
- Direct post-Koopman Δz prediction with variable dt.
- Explicit `H=[1,2,4,8,16]` fixed-delay sweep plus parameter-matched instantaneous and shuffled-history controls.
- Closed-loop evaluation and cache/checkpoint provenance.
- Effective-horizon/plateau inference with closed-loop evidence prioritized over ACF.

This is consistent with the motivation of latent Koopman memory closure in Gupta et al. (2025), while deliberately avoiding their joint autoencoder/operator training and recurrent memory models.

## Reviewed but deferred

- Neural DDEs: mature evidence supports delay closures and partial observation, but continuous-delay solvers and learned delays add a new numerical contract.
- RNN/GRU/LSTM: established memory models, but hidden recurrent state would prevent V0.7 from first identifying whether simple finite history is useful.
- Mamba-assisted closure: promising constant-step inference and sequence training, but the June 2026 work is a new preprint and much larger than the minimal test needed here.
- Attention-free Koopman memory/re-encoding: promising long-rollout results, but its correction-before-Koopman and dynamic re-encoding differ from V0.7’s frozen post-step residual contract.

## Rejected for V0.7

Joint fine-tuning, a new residual latent state, attention, stochastic forcing, physics-loss closure training, or claims of an exact Mori–Zwanzig kernel. Any later adoption requires V0.7 evidence and a new versioned mathematical review.
