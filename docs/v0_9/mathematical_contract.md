# V0.9 mathematical contract

冻结在线编码器、decoder、nominal continuous generator 与 V0.8 context encoder：

\[
z_t=E_\theta(U_t),\qquad c_t=\Phi_{0.8}(z_{t-H:t}),\qquad
q_t=G_\omega(c_t),\qquad
\eta_t=\sigma(h_\omega(c_t))\eta_{max}\tanh(q_t).
\]

唯一主预测方程为

\[
A_t=A_0+U\operatorname{diag}(\eta_t)V^\top,\qquad
\hat z_{t+1}=e^{A_t\Delta t_t}z_t.
\]

`eta` 输出层为零初始化，因此初始化严格退化为 `A_t=A0`。gate 初值约为 0.2，但不会破坏该等价性。
主链没有 `+ r_hat`，也没有 persistent `z_R`。低秩要求 `r < d_K`，修订候选为 `{2,4,8,12}`，
只使用 validation 选择。

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

阶段 2 不改变上述主预测方程，只增加问题适配器给出的冻结 decoder 可观测量目标：

\[
\mathcal L_{obs}=\frac{\sum_{h\in H_{obs}}\alpha_h
\sum_k w_k\ell_k(D(\hat z_{t+h}),U_{t+h})}{\sum_h\alpha_h}.
\]

当前 cylinder adapter 的 `k` 为速度、涡量、散度、壁面、升力、阻力；Koopman 核心不包含这些名称或公式。
decoder、encoder、context 与 `A0` 继续冻结，梯度只经 `D` 对输入的 Jacobian 回到低秩 adaptive operator。
