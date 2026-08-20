# Context semantics

`z_t` 表示当前 learned Koopman state；`c_t` 表示从 nominal residual 教师任务中提取的、对当前
Koopman inadequacy 有用的紧凑信息。它不被预先命名为频率、阻尼、涡相位或工况。

R2 使用 current `z_t`、next `dt` 与静态参数的小 MLP。R3 的 causal Attention token 只包含可见
history latent、与转移严格对齐的 `dt` 和静态参数；上三角 mask 禁止未来访问。Shuffled-history control
保留 current state 与 target，仅破坏 older-history 时序关系。

统一输出合同允许 V0.9 在不改 downstream API 的情况下研究 operator adaptation，但 V0.8 本身不会把
`c_t` 接到 `A0`。Attention weights 仅为诊断，不作因果解释。

Context collapse 通过坐标方差、effective rank 与常数性检查。科学支持还要求 context ablation 明显
降低 residual prediction，并且 R3 的真实历史优于 shuffled-history。

