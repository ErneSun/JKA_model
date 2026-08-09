# V0.4 Implementation Checklist

本文件是 V0.4 mandatory completion gate。只有所有 Mandatory 项为 `[x]` 且没有阻塞项，
才能报告 `V0.4 COMPLETE: YES`。

## A. Repository audit

- [x] 阅读架构 v2.2 的 V0.4、V0.1/V0.2/V0.3 walkthrough。
- [x] 审计现有 trajectory/split/normalizer/window/ProblemBatch/config/checkpoint/seed/logging。
- [x] 确认原样复用 V0.3 `ContinuousKoopmanCore`，不建立第二套 propagation。

## B. Architecture boundary

- [x] 仅实现 online `KoopmanEncoder`、`TrainingDecoder` 与薄组合层。
- [x] 使用 `TrainStage.KOOPMAN`，optimizer 仅含 encoder、decoder、A。
- [x] 无 JEPA、target/EMA encoder、residual、Attention、PDE/physics training、action conditioning。

## C. Synthetic systems

- [x] 实现 deterministic hidden rotation-decay dynamics 与 nonlinear observation `g(s)`。
- [x] 输出 V0.2-compatible trajectories `[T+1,5]`/`dts[T]` 和 `ProblemSpec`。
- [x] true latent 与模型输入分离，只允许 evaluation 使用。
- [x] trajectory-first split、train-only normalization、split 后 window。
- [x] 保留线性 observation sanity 与小型 Duffing diagnostic。

## D. KoopmanEncoder

- [x] 小型 learned encoder：model-space observation -> `z_k`，primary latent dim 2；正式配置采用线性层，作为推荐 MLP 的已记录偏差。
- [x] 严格 shape/dtype/finite/latent-dimension 验证。
- [x] 无 oracle inverse-map 初始化、无 BatchNorm/CNN/Attention/residual backbone。

## E. TrainingDecoder

- [x] 小型 MLP：`z_k` -> model-space observation reconstruction。
- [x] decoder 不接受 dt、不包含 propagation/A。
- [x] 严格 shape/dtype/finite 验证。

## F. Koopman representation losses

- [x] one-step latent consistency，target 来自同一个 online encoder，两侧有梯度。
- [x] multi-step closed-loop latent loss，不 teacher force。
- [x] model-space reconstruction loss。
- [x] stable `unbiased=False` variance loss。
- [x] optional differentiable stability regularizer，`lambda_spec=0` 默认关闭。
- [x] config 独立配置 `lambda_k/multi/rec/var/spec`。

## G. Multi-step rollout

- [x] `H_K>1` 正确工作并由 config 控制。
- [x] latent rollout 只调用 V0.3 core，包含 initial state。
- [x] decoded rollout 支持 variable dt，finite，且不重新 encode future truth。

## H. Non-collapse

- [x] 记录 latent mean/std/min/max 与 covariance condition。
- [x] 训练期间按 config interval 保存 loss 与 latent mean/std/min/max snapshots。
- [x] collapsed latent 被 variance loss 惩罚，noncollapsed latent loss 很小。
- [x] known-latent smoke minimum std 通过 gate。

## I. Training ownership

- [x] 仅 encoder、decoder、A 是 trainable/optimizer-owned。
- [x] backward 后三组 gradient 均 finite/nonzero。
- [x] normalizer、metrics、true latent 不在 optimizer 且无 gradient。

## J. Metrics

- [x] train/test reconstruction、latent one/multi-step、decoded model/raw rollout MSE。
- [x] held-out one-step 与 closed-loop multi-step latent MSE 单独评估和输出。
- [x] persistence baseline。
- [x] train-fit/test-apply affine latent alignment R2/MSE。
- [x] continuous spectrum/frequency/decay 使用 V0.3 diagnostics。
- [x] similarity-transform spectrum invariance。

## K. Checkpoint

- [x] 保存 encoder、decoder、A、optimizer、epoch/step、config、normalizer、split、fingerprint、RNG。
- [x] reload 后 encode/core/decode/prediction/spectrum 一致。
- [x] schema/project version 更新且旧 smoke 使用当前版本。

## L. Unit tests

- [x] encoder/decoder shape、gradient、latent validation、decoder 无时间接口。
- [x] exact one-step/multi-step/reconstruction losses。
- [x] closed-loop multi-step 防 teacher forcing。
- [x] variance collapse/non-collapse。
- [x] optimizer ownership与 backward gradients。
- [x] generator shape/dynamics/determinism/true-latent isolation。
- [x] affine alignment transform invariance与 spectrum similarity invariance。
- [x] learned rollout shape/history/finite/variable dt。
- [x] V0.4 checkpoint round-trip。

## M. Integration tests

- [x] deterministic learned-lifting integration finite、non-collapse、reconstruction improves。
- [x] V0.3 matrix-exp/semigroup/batch-dt/gradient/spectrum/oscillator 全部回归。

## N. Smoke test

- [x] `scripts/smoke_v0_4.py` 完成规定 23-step CPU pipeline。
- [x] primary acceptance 全部在 held-out test trajectories。
- [x] test reconstruction `<1e-3`、alignment R2 `>0.98`、frequency error `<2%`。
- [x] decoded Koopman rollout finite 且优于 persistence。
- [x] checkpoint reload 与 Duffing diagnostic 完成。
- [x] reconstruction-off/multi-step-off ablation 真实执行并报告。

## O. Teaching script

- [x] `scripts/explain_v0_4.py` 按 17 STEP 运行。
- [x] 打印小型 `s/U/U_model/z_k/z_next/decoded U` 真实例子。
- [x] 解释 alignment、similarity invariance、collapse、loss、Duffing 与 V0.5 接口。

## P. Code walkthrough

- [x] 创建中文 `docs/v0_4_code_walkthrough.md`，20 部分完整。
- [x] 记录真实层宽、参数量、shape、loss 权重、调用图和结果。
- [x] 明确 model-space decoder target、无 JEPA/EMA/physics loss。

## Q. Regression

- [x] 全量 pytest 通过。
- [x] V0.1/V0.2/V0.3/V0.4 smoke 全部通过。
- [x] `explain_v0_4.py` 与已实现的 `analyze_latent_v0_4.py` 通过。
- [x] latent analyzer 自动跳过非 V0.4 checkpoint，不依赖目录中最后一次运行类型。

## R. Final validation

- [x] Ruff、mypy、`git diff --check` 通过。
- [x] README、CHANGELOG、project/config/checkpoint version 同步至 V0.4。
- [x] 回查 checklist，无未完成或阻塞项。
- [x] 最终 diff 无 CUDA hard-code 和 V0.5+ 功能。
- [x] 最终报告列出所有不匹配、调整及真实数值。
- [x] V0.4 cross-section config mismatch 在训练前显式拒绝。
