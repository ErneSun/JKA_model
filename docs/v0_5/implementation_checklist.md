# V0.5 Local CPU Implementation Checklist

只有所有 Local Mandatory 项完成，才能报告 `V0.5 LOCAL CPU IMPLEMENTATION: PASS`。
GPU 与 Scientific 状态必须独立报告；本地未执行 GPU full training 时 Scientific 为
`PENDING_GPU`。

## A. Repository audit

- [x] 阅读架构 v2.2、V0.1–V0.4 walkthrough 与现有源码、测试、配置、脚本。
- [x] 确认复用 ProblemSpec/ProblemBatch/normalizer/split/window/checkpoint/RNG/logging。
- [x] 确认复用 V0.3 ContinuousKoopmanCore 与 V0.2 PhysicsConstraint，不建立平行 infrastructure。

## B. Technology review

- [x] 创建 latest_tech_review.md 与 references.md，只引用 primary sources。
- [x] 对 continuous Koopman、fluid/operator learning、physics-informed、AMP/profiling 作 ADOPT/OPTIONAL/DEFER/REJECT 判断。

## C. Problem abstraction

- [x] 建立 PhysicalProblem contract 与 advection-diffusion 2D adapter。
- [x] trainer/evaluator 不含具体 PDE 分支。
- [x] 文档说明替换新物理问题所需 adapter/dataset/spec/constraints/config。

## D. PDE dataset

- [x] Fourier analytical 2D periodic trajectories，endpoint=False。
- [x] 固定 cx/cy/nu，trajectory 改变 phase/amplitude/mean。
- [x] 复用 trajectory split、train-only normalization、windowing、ProblemBatch。
- [x] 解析 frequency、decay、mass truth 可计算。

## E. Field contracts

- [x] single/batch/context/future shapes 分别为 [C,Nx,Ny]/[B,C,Nx,Ny]/[B,H,C,Nx,Ny]/[B,K,C,Nx,Ny]。
- [x] coordinates、cell_weights、mu_static、variable dt 严格对齐。
- [x] invalid grid/config/transition 明确报错，无 silent correction。

## F. CNN encoder

- [x] 小型 Conv2d/SiLU/stride/pool/Linear encoder。
- [x] 所有 spatial convolution 使用 circular padding，不使用 zero padding。
- [x] shape/dtype/device/finite validation 与 gradient test。

## G. Training decoder

- [x] latent -> model-space 2D field decoder，不包含 dt/propagation。
- [x] 输出严格匹配 [B,C,Nx,Ny]，支持 autograd。

## H. Koopman integration

- [x] 薄 2D autoencoder 仅组合 encoder/V0.3 core/decoder。
- [x] encode U0 一次并 closed-loop rollout，支持 variable dt。
- [x] spectrum 使用 V0.3 implementation，按 nearest frequency pair 评估。

## I. PhysicsConstraint

- [x] differentiable inverse normalization 后在 raw units 计算 physics。
- [x] mass loss 使用 cell_weights。
- [x] operator consistency 使用 periodic finite differences 与 trapezoidal residual。
- [x] physics gradient 到 encoder、decoder、A，且 finite/nonzero。
- [x] 支持 physics_warmup_epochs 与 lambda_physics=0 ablation。

## J. Training function

- [x] 唯一 canonical Python 入口 `src.train.train_v0_5.train_v0_5`。
- [x] TrainStage.KOOPMAN optimizer 只含 encoder/decoder/A。
- [x] NaN guard 检查 loss/grad/parameters。
- [x] CLI `scripts/train_v0_5.py` 是 thin wrapper。

## K. Evaluation

- [x] 唯一 canonical `evaluate_v0_5`，CLI 为 thin wrapper。
- [x] field/reconstruction/short-medium-long rollout/persistence/spectrum/physics metrics。
- [x] run inspector 显示 commit/device/config/best epoch/forecast/spectrum/physics/checkpoint。

## L. Experiment records

- [x] runs/v0_5/<backend>/<run_id> 标准目录完整。
- [x] manifest/environment/git/data/model summary 完整。
- [x] canonical epoch_metrics.csv 与 step_metrics.jsonl。
- [x] last/best_forecast/best_physics checkpoints 与 evaluation JSON/CSV/reports。

## M. CPU tests

- [x] 2D data/split/normalizer/batch/variable-dt tests。
- [x] Dx/Dy/Dxx/Dyy/periodic/grid-convergence/mass truth tests。
- [x] CNN/decoder/invalid-grid/closed-loop tests。
- [x] raw-unit inverse-normalization与 physics backprop tests。
- [x] training/run-record/checkpoint/resume/evaluation tests。
- [x] V0.1–V0.4 与 V0.3 KoopmanCore regression 全部通过。

## N. CPU smoke

- [x] `scripts/smoke_v0_5.py` CPU-only 通过。
- [x] tiny overfit loss/reconstruction/Koopman loss 下降。
- [x] closed-loop finite 且相对 persistence 指标被真实报告。
- [x] `scripts/explain_v0_5.py` 通过。

## O. Resume

- [x] resume 恢复 model/optimizer/epoch/step/RNG/normalizer/split/fingerprint。
- [x] fingerprint 或 split mismatch 默认失败。
- [x] resume 不重新 split，实际执行一次并验证继续训练。

## P. Documentation

- [x] README/architecture/problem_contract/physics/training/evaluation/testing/tech/references/walkthrough/status 完整。
- [x] training.md 第一屏明确 canonical Python function 与 CLI。
- [x] Local -> Git -> GPU handoff 不假定 branch 名。

## Q. GPU validation package

- [x] gpu_validation/v0_5 README/plan/Codex task/checklist/results 完整。
- [x] gpu_smoke/full/full_no_physics configs 复用正式 schema。
- [x] preflight/smoke/train/evaluate/profile 全是 thin wrappers。
- [x] GPU parity/FP32/AMP/physics gradient/resume/performance/result-record 流程完整。

## R. Final local validation

- [x] 项目版本 0.5.0、checkpoint schema 与历史 configs 同步。
- [x] pytest、ruff、mypy、git diff --check 通过。
- [x] V0.1–V0.5 smoke、tiny train、resume、evaluation、inspector、explain 实际通过。
- [x] GPU status 如实为 NOT RUN，Scientific status 如实为 PENDING_GPU。
- [x] 未实现 JEPA/EMA/z_r/residual/GRU/Transformer/Attention/MPC/RL/V0.6。
