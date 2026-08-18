# Residual semantics

Definitions:

\[
z^{\rm true}_{t+1}=E_{\theta}^{\rm online}(U_{t+1}),\quad
z^{\rm base}_{t+1}=\exp(A\Delta t_t)E_{\theta}^{\rm online}(U_t),
\]

\[
r_{t+1}=\operatorname{sg}(z^{\rm true}_{t+1}-z^{\rm base}_{t+1}),\quad
\widehat r_{t+1}=C_\psi(\text{past resolved data},\Delta t_t,\mu).
\]

`z_true` is an offline supervised target. `z_base` is the frozen Koopman prediction. `residual_target` is detached. `predicted_correction` is the closure output used during rollout.

The EMA target encoder is not interchangeable with the frozen online encoder. Changing EMA weights must leave the cache unchanged. Each `dt[t]` belongs only to `z[t] -> z[t+1]`; windows cannot cross trajectory or split boundaries.

`H=1` supplies exactly `(z_t, dt_t, mu)` and is therefore the operational Markovian baseline. `H>1` adds `(z_{t-H+1:t-1}, dt_{t-H+1:t-1})`. No future latent, future interval, EMA representation, or physics residual is included in the closure input or target.

Actual batch tensors are `history_z:[B,H,d_K]`, `history_dts:[B,H-1]`, `next_dt:[B,1]`, optional known static parameters `[B,d_mu]`, and `residual_target:[B,d_K]`. The equivalent two-dimensional dt representation is retained rather than rewritten solely for notation.

Residual standardization uses only training data. Validation and test reuse the stored per-dimension training RMS. Raw MSE/RMS, standardized MSE, NRMSE, R², and per-dimension metrics remain separately reported.
