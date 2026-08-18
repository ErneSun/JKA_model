# V0.8 route recommendation contract

V0.7 does not implement V0.8. Its completed result selects only a defensible residual route:

- `R0`: no context learner; retain Koopman-only;
- `R1`: diagnose noise, missing observables/forcing, latent adequacy, stochasticity, and Koopman capacity;
- `R2`: V0.8 may test a small instantaneous dynamic-context MLP;
- `R3`: V0.8 may test small causal Attention at the locked H;
- `INCONCLUSIVE`: do not select a context family.

Memory class remains supporting evidence and no longer directly chooses the architecture family. The generated recommendation is evidence-dependent and never treats software success as scientific support. V0.7 implements none of `c_t`, `eta_t`, adaptive `A_t`, or V0.8 training.
