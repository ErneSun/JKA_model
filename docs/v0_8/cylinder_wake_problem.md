# Fixed-condition transient cylinder wake

## Physical problem

研究对象是圆柱外二维不可压缩 Navier–Stokes：

\[
\nabla\cdot u=0,\qquad
\partial_tu+(u\cdot\nabla)u=-\nabla p+Re^{-1}\nabla^2u,
\]

默认 `Re=100`、`D=1`、`U_inf=1`，固定均匀入口、固定上下远场、零梯度出口和圆柱无滑移。
边界不随时间变化；变化工况属于 V0.9。

学习状态为 `[u/U_inf, v/U_inf, pressure_coefficient]`。压力保留是为了可审计的升阻力比较；固体
mask、坐标、cell weights 和 `[Re,U_inf,D]` 条件元数据均显式保存。

## Offline numerical generator

仓库没有现成 CFD 引擎，因此实现 minimum reliable low-Mach D2Q9 BGK 离线生成器。它的用途是构造
可重复的研究基准，不是 production CFD，也不宣称替代高阶有限元/有限体积基准。训练 epoch 内绝不运行
CFD；每个 flow seed 只生成一次 `.pt`，并以完整物理/数值合同指纹加载。

CNN 类结构与 V0.6 一致，但 padding 由 ProblemSpec 决定：周期问题保持 `circular`，固定边界圆柱问题
使用 `zeros`，避免把出口卷回入口。

## Mandatory pre-ML gate

正式训练前检查：有限/有界场、散度、无灾难失稳、非平凡瞬态、升力振荡与可辨识谱峰；同时以
`128x64 -> 256x128` 的同 snapshot-dt 检查脱涡频率变化。只有物理门槛和 grid adequacy 都通过，才进入
backbone 训练。

该门槛是内部基准一致性检查。若后续需要定量流体结论，应再与 Schäfer–Turek benchmark 或独立 CFD
数据校准，而不能把当前 reduced-grid traction diagnostic 当成高精度表面力积分。

