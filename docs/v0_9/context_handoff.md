# V0.8 context handoff

V0.9 读取 V0.8 compact decision 与 raw checkpoints，并要求：

- `v0_9_ready=true`；
- `joint_v0_9_support_fraction=1.0`；
- 正好三个 backbone/data seeds；
- 每个 seed 的 `v0_9_supported=true`；
- route 为 R2/R3；
- selected context family 和 checkpoint 可解析；
- context checkpoint 的 backbone SHA 与实际 backbone 相同。

每个 backbone seed 从 validation 指标中选择该 locked family 的最佳 context-init checkpoint。V0.9 不重新进行
context-family search，也不把 Attention 写死；当前 handoff 若选择 History MLP，就冻结并复用 History MLP。

任一条件失败时输出 `V0.9 BLOCKED BY V0.8 READINESS`，不降低门槛、不自动重训 V0.8。
