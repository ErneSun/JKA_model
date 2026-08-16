# Architecture contract

For normalized field \(U_t\),

\[
z_t=E_\theta(U_t),\qquad
\hat z_{t+\Delta t}=\exp(A\Delta t)z_t,
\qquad z^{J}_{t+\Delta t}=E_{\bar\theta}(U_{t+\Delta t}).
\]

`KoopmanCore` and its matrix exponential are the only predictor. `E_bar` is
hard-synchronized only on V0.5 initialization, frozen, excluded from the optimizer, and
kept in evaluation mode. Formal rollout uses online encoder + Koopman core + decoder.

The decoded rollout returns to raw units before the unchanged V0.5 relative-mass and
exact Fourier-step constraints are evaluated.

| Module | Trainable | Optimizer | Inference |
|---|---:|---:|---:|
| Online encoder | yes | yes | yes |
| Continuous generator `A` | yes | yes | yes |
| Training decoder | yes | yes | yes |
| EMA target encoder | no | no | no |
| Physics constraints/normalizer | no | no | normalizer only |
