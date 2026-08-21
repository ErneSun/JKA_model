# V0.8 context handoff

V0.9 读取 V0.8 compact decision 与 raw checkpoints。确认性 `strict` 路线要求：

- `v0_9_ready=true`；
- `joint_v0_9_support_fraction=1.0`；
- 正好三个 backbone/data seeds；
- 每个 seed 的 `v0_9_supported=true`；
- route 为 R2/R3；
- selected context family 和 checkpoint 可解析；
- context checkpoint 的 backbone SHA 与实际 backbone 相同。

每个 backbone seed 从 validation 指标中选择该 locked family 的最佳 context-init checkpoint。V0.9 不重新进行
context-family search，也不把 Attention 写死；当前 handoff 若选择 History MLP，就冻结并复用 History MLP。

另设显式 `supported` 路线：要求 V0.8 聚合结论为 `dynamic_context=SUPPORTED`，三套 seed artifact
完整，但允许个别 seed 未通过以及额外的 3/3 V0.9 readiness 为 `NOT_READY`。失败 seed 仍进入
V0.9，用于检验 adaptive operator 能否修复该情形，且其来源状态完整写入 handoff audit。这不是把 strict gate
伪装成通过，而是允许 V0.9 检验 adaptive operator 能否修复已知不足；结果必须标记为
`EXPLORATORY_CONDITIONAL`，最高只能得到 `CONDITIONALLY_SUPPORTED`，并禁止进入 V1.0 readiness。

两条路线都不自动重训 V0.8；旧报告仅使用现有 checkpoint 重评 locked test。
