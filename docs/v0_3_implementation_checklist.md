# V0.3 Implementation Checklist

本文件是 V0.3 mandatory completion gate。只有所有 Mandatory 项为 `[x]`，且没有
任何阻塞项，才能报告 `V0.3 COMPLETE: YES`。

## A. Repository audit

- [x] 完整复核架构 v2.2 中时间语义、V0.3 数学/API/边界。
- [x] 阅读 V0.1/V0.2 walkthrough 和当前 `src/tests/scripts/configs/docs`。
- [x] 确认复用 trajectory、ProblemBatch、config、checkpoint、seed、logging。
- [x] 确认未建立平行基础设施，未实现 V0.4+。

## B. Mathematical core

- [x] 实现唯一 column-vector convention 的 continuous generator `A[d,d]`。
- [x] 支持 fixed-A 与 trainable-A。
- [x] `transition_matrix(dt)` 真正调用 `torch.matrix_exp`，无 Euler/RK 替代。
- [x] `step()` 支持 `[d]`、`[B,d]`、scalar dt、batch `[B]` dt。
- [x] 接受 `dt=0` identity，拒绝 negative/non-finite/wrong-shape dt。
- [x] batched dt 使用 batched `matrix_exp`，不是 Python loop 主实现。
- [x] matrix exponential 位于 CPU-safe precision island。
- [x] float64 reference 与 float32 normal model 均可用。
- [x] gradient 可经 `matrix_exp` 到达 `A`，finite 且非零。
- [x] semigroup property 已验证。

## C. Oscillator data

- [x] 实现 deterministic damped harmonic oscillator generator。
- [x] 实现 independent rotation-decay closed-form reference。
- [x] physical `[x,v]` fixed-A smoke 使用独立 underdamped analytical transition。
- [x] 支持 constant 与 deterministic variable dt。
- [x] 实现 deterministic unforced Duffing RK4 reference generator。
- [x] RK4 只存在于 reference data generator，不进入 KoopmanCore。
- [x] 增加 `scripts/generate_oscillator.py`。

## D. KoopmanCore API

- [x] `transition_matrix(dt)`。
- [x] `generator_matrix()`。
- [x] `step(z, dt)`。
- [x] `rollout(z0, dts, horizon=None)`，包含 initial state。
- [x] rollout 支持 scalar、`[H]`、`[B,H]` schedules。
- [x] rollout 是 closed-loop prediction-to-prediction。
- [x] `spectrum()` 返回 detached diagnostics。

## E. Direct-state identification

- [x] 只训练矩阵 `A`，无其它 trainable model。
- [x] loss 只有 one-step matrix-exponential MSE。
- [x] deterministic non-near-truth initialization。
- [x] 实现小型 zero-grad/forward/backward/step loop。
- [x] constant-dt oscillator identification frequency error `<1%`。
- [x] variable-dt identification 明确使用各自 dt。
- [x] 报告 learned frequency/damping 与相对误差。
- [x] relative-frequency 与 spectral-growth metrics 为可复用 API。
- [x] Duffing identification loss finite。

## F. Spectrum diagnostics

- [x] 实现 continuous eigenvalues、growth rates、angular frequencies、Hz frequencies。
- [x] diagnostics detached，不参与训练 loss。
- [x] 比较逻辑 permutation-invariant，不依赖 eig 顺序。
- [x] 增加 `src/jka_model/metrics/spectral.py`。
- [x] 增加并实际运行 `scripts/analyze_spectrum.py`。

## G. Rollout

- [x] 单轨迹 constant/variable-dt rollout 正确。
- [x] batch shared/per-sample schedules 正确。
- [x] utility 只调用 core rollout，保持单一实现。
- [x] learned oscillator 100-step rollout finite。
- [x] 报告 one-step/rollout MSE、growth rate。
- [x] persistence baseline 已计算，Koopman 明显更优。
- [x] Duffing rollout finite，并报告 limitation diagnostics。

## H. Tests

- [x] closed-form matrix exponential。
- [x] zero-dt identity 与 semigroup。
- [x] single/batch/shared/per-sample step shapes 与数值。
- [x] negative dt rejection。
- [x] matrix-exp gradient。
- [x] constant/variable/batch/include-initial/repeated-step rollout。
- [x] known spectrum、frequency extraction、detached diagnostics。
- [x] damped oscillator independent reference。
- [x] learned frequency `<1%`。
- [x] 100-step finite、beats persistence。
- [x] Koopman checkpoint round-trip step/spectrum consistency。
- [x] Duffing pipeline finite/correct shapes。
- [x] 推荐的 dtype、batch-vs-loop、CPU transfer、config round-trip 测试。
- [x] non-finite/wrong-shape/wrong-length dt 与 invalid rollout schedule 测试。
- [x] batch shared/per-sample rollout 与逐轨迹 reference 数值一致。

## I. Smoke test

- [x] `scripts/smoke_v0_3.py` 完成 17-step real pipeline。
- [x] fixed-A closed-form/semigroup errors 已报告。
- [x] identification frequency/damping 已报告。
- [x] 100-step Koopman/persistence errors 已报告。
- [x] checkpoint reload consistency 已报告。
- [x] 复用 trajectory split；held-out trajectory 用于 rollout evaluation。
- [x] 复用 V0.1 run metadata、resolved config 与 structured logging。
- [x] Duffing one-step/rollout/finite 已报告。
- [x] mandatory PASS/FAIL summary；CPU-only。

## J. Teaching script

- [x] `scripts/explain_v0_3.py` 按 13 STEP 教学顺序运行。
- [x] 打印具体 `A`、eigenvalues、`dt`、`K(dt)`、`z`、next state。
- [x] 解释 variable dt、gradient/learning、spectrum、persistence、Duffing limitation。
- [x] 只说明 V0.4 接口，不实现 V0.4。

## K. Code walkthrough

- [x] 创建中文 `docs/v0_3_code_walkthrough.md`。
- [x] 完成 V0.1/V0.2/V0.3 对照与核心数学。
- [x] 解释 `A != K(dt)`、column convention 与不使用 Euler。
- [x] 完成真实代码结构、数据流、shape/spectrum tables。
- [x] 完成 oscillator fixed/trainable 结果与 variable-dt 示例。
- [x] 解释 Duffing closure limitation。
- [x] 完成 tests table、reading order、top symbols、V0.4 interface。

## L. Regression

- [x] 全量 pytest 包含 V0.1/V0.2/V0.3，全部通过。
- [x] `scripts/smoke_v0_1.py` 成功。
- [x] `scripts/smoke_v0_2.py` 成功。
- [x] 未破坏 V0.2 additional numerical verification。

## M. Final validation

- [x] `scripts/smoke_v0_3.py` 成功。
- [x] `scripts/explain_v0_3.py` 成功。
- [x] `scripts/analyze_spectrum.py` 成功。
- [x] checkpoint 保存 A、optimizer、config、revision/version、epoch/step、RNG。
- [x] checkpoint 测试逐项验证 optimizer/config/version/epoch/step/RNG。
- [x] Ruff、mypy、`git diff --check` 成功。
- [x] README、CHANGELOG、project/config/checkpoint version 已同步至 V0.3。
- [x] 重新打开本 checklist，所有 Mandatory 均为 `[x]`，无 TODO/stub/blocker。
- [x] 最终 diff 确认无 CUDA hard-code、无 encoder/JEPA/residual/Attention/V0.4 功能。
