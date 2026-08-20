# V0.8 — Residual-Supervised Dynamic Context Learning

V0.8 在固定工况的瞬态二维圆柱绕流上，先训练新的 V0.6-compatible
JEPA–Koopman backbone，再执行更新后的 V0.7 R1/R2/R3 残差结构判定，并仅在证据允许时训练紧凑 context。

主链路为：

```text
offline cylinder trajectories
  -> V0.6 online/EMA JEPA + nominal continuous Koopman A0
  -> frozen-online residual r0
  -> V0.7 R1/R2/R3 route
  -> R2 instantaneous MLP OR R3 validation-selected history MLP/causal Attention
  -> c_t -> residual and adequacy teacher heads
```

当前 V0.7 合同优先于原始 V0.8 提示词：不存在 R0。残差幅值只作诊断；即使幅值低，也进入
R1/R2/R3 的可学习性判定，不能被忽略。

文档：

- `mathematical_contract.md`：唯一数学合同与禁止范围；
- `cylinder_wake_problem.md`：物理问题、离线求解器和验收门槛；
- `context_semantics.md`：R2/R3 context 的语义；
- `evaluation.md`：validation/test、closed-loop 与物理判据；
- `testing.md`：收敛测试策略和 GPU 一行命令；
- `technology_review.md` / `technology_adoption_declaration.md`：近期方法审查与取舍；
- `status.md`：实现状态，不等同于科学通过；
- `v0_9_problem_extension.md`：只定义后续接口，不实现 V0.9。

