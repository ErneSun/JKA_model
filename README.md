# Koopman-Structured Physical JEPA World Model

本仓库当前完成 **V0.4 — Learned Koopman Encoder + Training Decoder**。项目版本为 `0.4.0`，
唯一有效架构修订为 `2.2`。

V0.4 在不改变 V0.3 连续时间 core 的前提下学习动力学坐标：

\[
U^{model}\xrightarrow{E_K}z_K\xrightarrow{e^{A\Delta t}}\hat z_K
\xrightarrow{D_{train}}\hat U^{model}.
\]

`ContinuousKoopmanCore` 使用真正的 `torch.matrix_exp`，支持 scalar/batch/variable `dt`、
closed-loop rollout、continuous spectrum 和通过 matrix exponential 的 gradient。V0.4 新增 learned
`KoopmanEncoder`、training-only decoder、闭环 multi-step loss、非坍缩诊断与仿射 latent 对齐。

## 已完成版本

```text
V0.1  engineering contracts / config / checkpoint / reproducibility
V0.2  trajectory windows / train-only normalization / PhysicsConstraint
V0.3  direct-state continuous-time Koopman generator
V0.4  learned Koopman coordinates + training decoder（当前）
V0.5+ PDE field encoder / physics training / JEPA / closure（未实现）
```

V0.4 同时包含：

- known-latent rotation-decay 与 nonlinear observation `U=[q,p,q²,qp,p²]`；
- trajectory split、train-only normalization 和 V0.2 `ProblemBatch` windows 原样复用；
- one-step/multi-step latent consistency、model-space reconstruction、variance 与 optional stability loss；
- train-fit/test-apply affine alignment、similarity-invariant spectrum 与 persistence baseline；
- held-out one-step/multi-step latent、decoded model/raw prediction diagnostics，以及训练期
  loss/latent-statistic snapshots；
- encoder/decoder/A/optimizer/normalizer/split/RNG 完整 checkpoint round-trip；
- reconstruction-off 与 multi-step-off 实际消融；
- learned-lifting Duffing 二级诊断，并保留 finite-dimensional closure limitation；
- CPU-only tests、23-step smoke、17-step教学脚本与 latent analyzer。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

如本机已有 PyTorch/NumPy：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e . --no-deps
```

## 完整验证

```bash
pytest -q
python scripts/smoke_v0_1.py
python scripts/smoke_v0_2.py
python scripts/generate_oscillator.py
python scripts/smoke_v0_3.py --checkpoint-output /tmp/jka_v0_3.pt
python scripts/explain_v0_3.py
python scripts/analyze_spectrum.py --checkpoint /tmp/jka_v0_3.pt
python scripts/smoke_v0_4.py --checkpoint-output /tmp/jka_v0_4.pt
python scripts/explain_v0_4.py
python scripts/analyze_latent_v0_4.py --checkpoint /tmp/jka_v0_4.pt
ruff check .
MYPYPATH=src mypy
```

## 数学与 API 约定

数学统一使用 column state：`z_next = K @ z`。代码中的 `[B,d]` 每一行存储一个数学 column
state，因此 shared transition 的等价 batch 运算是 `z @ K.T`。

```python
core.transition_matrix(dt)
core.step(z, dt)
core.rollout(z0, dts, horizon=None)
core.spectrum()
```

`rollout()` 包含初始状态，不使用 ground truth teacher forcing。负 `dt` 被拒绝，`dt=0` 用于
identity test。

## V0.4 范围边界

V0.4 没有 JEPA、target/EMA encoder、`z_r`、residual closure、Attention、action-conditioned
dynamics、PDE field encoder、physics loss、MPC 或 RL。当前 primary encoder 是随机初始化并经训练
得到的线性层；decoder 是两层 Tanh MLP。Duffing 只作为有限维 learned-lifting diagnostic，不能
据此宣称存在精确有限维线性 closure。

文档入口：

- [架构规范](./koopman_structured_physical_jepa_world_model_v2_2.md)
- [V0.4 Code Walkthrough](./docs/v0_4_code_walkthrough.md)
- [V0.4 Implementation Checklist](./docs/v0_4_implementation_checklist.md)
- [V0.3 Code Walkthrough](./docs/v0_3_code_walkthrough.md)
- [V0.2 历史导读](./docs/v0_2_code_walkthrough.md)
