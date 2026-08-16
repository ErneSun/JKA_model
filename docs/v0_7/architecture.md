# Architecture

The frozen V0.6 inference backbone is

\[
z_t=E_{\theta}^{\rm online}(U_t),\qquad
z^{\rm base}_{t+1}=\exp(A\Delta t_t)z_t,\qquad
\widehat U_{t+1}=D(z_{t+1}).
\]

V0.7 adds only

\[
\Delta z_{t+1}=C_\psi(z_{t-H+1:t},\Delta t_{t-H+1:t-1},\Delta t_t,\mu),
\qquad
z_{t+1}=z^{\rm base}_{t+1}+\Delta z_{t+1}.
\]

The correction is applied after the exact matrix-exponential Koopman step. No gate, residual state `z_R`, attention, RNN, GRU, LSTM, Mamba, re-encoding, or joint backbone fine-tuning exists in V0.7.

For `H=1`, the past-interval sequence is empty and the closure sees exactly `(z_t, Δt_t, μ)`. The formal GPU experiment sweeps `H=[1,2,4,8,16]`; the target definition and frozen backbone never change with H.

Module ownership:

| group | V0.7 state |
|---|---|
| online encoder | frozen/eval |
| Koopman generator A | frozen/eval |
| training decoder | frozen/eval |
| EMA target encoder | frozen/eval; provenance only |
| residual head | trainable |
