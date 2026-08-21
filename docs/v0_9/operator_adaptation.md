# Operator adaptation

`LowRankAdaptiveOperator` 只训练 `G_omega` 和低秩 factors。`U/V` 列归一化移除尺度 gauge，并记录
orthogonality、singular values、effective rank 与 operator burden：

\[
B_A(t)=\frac{\|A_t-A_0\|_F}{\|A_0\|_F+\epsilon}.
\]

Known-condition 输入为冻结动态 context 与 train-only 标准化后的当前 `[Re,U]`；latent-inferred 只接收冻结
context。任何向 latent-inferred 传入 condition tensor 的调用都会失败。

坐标先经过 `eta_max*tanh` 上界，再乘 learned sigmoid trust gate。它限制“算子能改多大、何时应该改”，
但不裁剪 latent state 或物理场。

Validation rank sweep 同时考虑 known/latent composite objective。候选必须具有非负 H32 validation gain 且
operator burden 不超过阈值；在满足约束的候选中选择位于最优值 2% 内的最小 rank。若无候选满足，报告明确记录
fallback，不把它伪装为成功。Locked test 不参与 rank、正则或 checkpoint 选择。
