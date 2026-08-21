# V0.9 mathematical contract

冻结在线编码器、decoder、nominal continuous generator 与 V0.8 context encoder：

\[
z_t=E_\theta(U_t),\qquad c_t=\Phi_{0.8}(z_{t-H:t}),\qquad
\eta_t=G_\omega(c_t).
\]

唯一主预测方程为

\[
A_t=A_0+U\operatorname{diag}(\eta_t)V^\top,\qquad
\hat z_{t+1}=e^{A_t\Delta t_t}z_t.
\]

`eta` 输出层为零初始化，因此初始化严格退化为 `A_t=A0`。主链没有 `+ r_hat`，也没有 persistent
`z_R`。低秩要求 `r < d_K`，默认候选为 `{1,2,4,8}`，只使用 validation 选择。

残差分解固定为

\[
r^0=z_{t+1}^{true}-e^{A_0\Delta t}z_t,
\quad r^{op}=(e^{A_t\Delta t}-e^{A_0\Delta t})z_t,
\quad r^{rem}=z_{t+1}^{true}-e^{A_t\Delta t}z_t,
\]

并逐样本满足 `r0 = rop + rrem`。Operator-explained fraction：

\[
\Gamma_{op}=1-\frac{\mathbb E\|r^{rem}\|^2}{\mathbb E\|r^0\|^2+\epsilon}.
\]
