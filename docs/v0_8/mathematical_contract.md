# V0.8 mathematical contract

## Frozen nominal system

在线编码器、EMA target、training decoder 与 nominal continuous generator 均来自经过新问题验证的
V0.6-compatible backbone。V0.8 固定

\[
z_t=E_\theta(U_t),\qquad z_{t+1}^{0}=\exp(A_0\Delta t_t)z_t,
\]

并使用与 V0.7 完全相同的、由 frozen online encoder 定义的目标

\[
r_{t+1}^{0}=\operatorname{stopgrad}\!\left[
E_\theta(U_{t+1})-\exp(A_0\Delta t_t)E_\theta(U_t)
\right].
\]

EMA encoder 不参与残差坐标定义。

## Route-owned context

更新后的 V0.7 只允许 `R1/R2/R3/INCONCLUSIVE`：

- R1：测试信息下不可稳定预测，只诊断，不训练 context；
- R2：可预测但历史无材料增益，使用 current-state instantaneous context；
- R3：历史有可重复材料增益，比较 history MLP 与 causal Attention；
- INCONCLUSIVE：停止架构升级并保留证据。

R2：

\[
c_t=\Phi_M(z_t,\Delta t_t,\mu),
\]

R3：

\[
c_t=\Phi_H(z_{t-H+1:t},\Delta t_{t-H+1:t},\mu),
\]

两条路径共用

\[
\hat r_{t+1}^{0}=R_\phi(c_t,z_t,\Delta t_t,\mu),\qquad
\hat m_t=Q_\chi(c_t),\quad m_t=\sqrt{d_K^{-1}\|r_{t+1}^{0}\|_2^2}.
\]

训练残差逐维使用 train split RMS 标准化，adequacy 使用 train split 标量 RMS；validation/test
只复用该尺度。损失为

\[
\mathcal L_{0.8}=\operatorname{MSE}(\hat r^0/s_r,r^0/s_r)
+\lambda_Q\operatorname{MSE}(\hat m/s_m,m/s_m).
\]

Residual 与 adequacy head 采用零输出初始化，因此初始 additive probe 严格退化为 nominal Koopman。

## Absolute boundary

V0.8 不实现 `eta_t`、`A_t`、低秩 operator update、persistent `z_R`、gate、joint fine-tuning、
MPC/RL 或变化边界。Closed-loop 中的 additive residual 只是 context utility probe，不能解释为最终 closure。

