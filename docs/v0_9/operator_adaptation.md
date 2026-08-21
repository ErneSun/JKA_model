# Operator adaptation

`LowRankAdaptiveOperator` 只训练 `G_omega` 和低秩 factors。`U/V` 列归一化移除尺度 gauge，并记录
orthogonality、singular values、effective rank 与 operator burden：

\[
B_A(t)=\frac{\|A_t-A_0\|_F}{\|A_0\|_F+\epsilon}.
\]

Known-condition 输入为冻结动态 context 与 train-only 标准化后的当前 `[Re,U]`；latent-inferred 只接收冻结
context。任何向 latent-inferred 传入 condition tensor 的调用都会失败。

Validation rank sweep 同时考虑 known/latent validation objective，并选择位于最优值 2% 内的最小 rank。
Locked test 不参与 rank、正则或 checkpoint 选择。
