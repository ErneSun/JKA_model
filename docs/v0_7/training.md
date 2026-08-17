# Training

1. Load the final V0.6 JEPA checkpoint and verify the inherited architecture/data/scientific contract.
2. Regenerate the exact split and normalizer; reject fingerprint mismatch.
3. Freeze all V0.6 modules.
4. Build the residual cache with the online encoder.
5. Analyze residual scale, quantiles, ACF, cross-dimension correlation, z/dt dependence, and closure burden.
6. Sweep `H=[1,2,4,8,16]` and train each neural closure from the same cache with a dimension-standardized residual loss. For every H, construct a parameter-matched instantaneous control; for H>1, construct a shuffled-history control. Repeat the closure experiment with initialization seeds `[101,211,307]`, kept independent of the backbone/data seed:

\[
 s_j=\max\!\left(\sqrt{\mathbb E_{\rm train}[r_j^2]},\epsilon_s\right),\qquad
\mathcal L_{\rm residual}=\frac1{Nd}\sum_{i,j}
\left(\frac{\widehat r_{ij}-r_{ij}}{s_j}\right)^2.
\]

The scale is computed only from the training split and is recorded in provenance. It changes optimization conditioning, not the residual definition or the Koopman equation. Linear and neural output layers are initialized to zero, so every trainable closure begins from the frozen Koopman baseline. Parameter matching uses an analytic count and therefore does not consume random numbers before initialization.

Physics terms do not enter this loss. AMP may be used for the closure, while the inherited Koopman core retains its FP32 exact-matrix-exponential contract. Formal GPU training prints start and compact final results; complete raw and standardized epoch metrics remain in CSV logs.

Checkpoints are saved at epoch boundaries with optimizer, scheduler, AMP scaler, RNG state, backbone, closure, config, split, normalizer, source-checkpoint SHA-256, and cache fingerprint. Resume rejects any mismatch.

`H=1` uses only `(z_t, dt_t, mu)` and therefore exactly defines the current-state Markovian information set. Increasing H adds only ordered past latent states and the intervals between them; it does not change the frozen residual target.
