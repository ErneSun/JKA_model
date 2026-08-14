# V0.5 Latest Technology Review

调研日期：2026-08-09。只使用论文原文与官方框架文档；技术选择服从 V0.5 边界。

| Topic | Decision | V0.5 use | Reason / later destination |
|---|---|---|---|
| Continuous-time Koopman autoencoder | ADOPT | CNN encoder + exact matrix exponential latent dynamics | 与 variable-dt、spectrum 和 long rollout 目标直接一致；不引入额外 integrator |
| Structured/stable Koopman generators | DEFER | unconstrained small continuous generator, but spectrum is recorded | Stable/skew/dissipative parameterizations can improve long rollouts, but changing generator geometry would confound the first 2-D physics baseline |
| High-dimensional fluid Koopman learning | ADOPT (diagnostics only) | closed-loop field rollout, spectrum, decay, and persistence comparison | Recent fluid work supports evaluating dynamics and spectral structure together; larger flow backbones and datasets remain outside V0.5 |
| Multi-step closed-loop training | ADOPT | small-horizon latent rollout loss | 直接约束累积传播误差，禁止 teacher forcing |
| Physics-informed operator residual | ADOPT | raw-unit mass + exact Fourier step consistency | 周期光滑场的 Fourier 方法具有谱精度；避免解析数据与低阶有限差分残差之间的训练偏差 |
| Circular CNN padding | ADOPT | every spatial Conv2d preceded by circular padding | 与 endpoint=False periodic topology 一致 |
| PyTorch AMP | OPTIONAL | GPU wrapper supports fp32/amp_fp16/amp_bf16 | official AMP API适合 convolution；matrix exponential、finite differences 和 reductions 保持 fp32 precision islands |
| torch.profiler | OPTIONAL | GPU short profiler wrapper | 只 profile 少量 steps，记录 CNN/matrix_exp/physics operator 时间和 memory |
| FNO / PINO backbone | DEFER | not implemented | 对 PDE operator learning 很相关，但会同时改变 backbone 与 Koopman/physics baseline；留给后续受控版本 |
| PhysicsNeMo model stack | DEFER | documentation reference only | 有成熟 neural-operator/physics/GPU scaling 工具，但本版必须复用现有轻量 PyTorch infrastructure |
| ResNet/UNet/Transformer/Attention | REJECT | none | 超出第一版小 CNN 和 V0.5 architecture boundary |
| Learned parameter-conditioned `A(mu)` | DEFER | fixed `cx,cy,nu` and fixed A | V0.5 多轨迹只改变 phase/amplitude/mean；参数条件动力学需单独版本验证 |

## Engineering conclusions

1. CPU correctness 使用普通 FP32；reference/operator convergence tests 使用 FP64。
2. GPU 先做 FP32 parity，再独立验证 AMP；不把 bitwise equality 当作 parity 标准。
3. `torch.matrix_exp` 显式进入至少 FP32 precision island。
4. physics loss 必须在 differentiable inverse normalization 后的 raw state 上计算。
5. FNO、PINO、PhysicsNeMo、distributed training 均不进入当前核心实现。
6. 当前固定系数周期线性 PDE 使用可微 Fourier 精确传播；这只是 physics constraint，未引入 FNO backbone。

详细来源见 `references.md`。
