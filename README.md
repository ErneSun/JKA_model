# Koopman-Structured Physical JEPA World Model

本仓库当前完成 **V0.3 — Direct-State Continuous-Time KoopmanCore**。项目版本为 `0.3.0`，
唯一有效架构修订为 `2.2`。

V0.3 第一次实现真实动力学模型，但范围严格限定为低维 direct state：

\[
\dot z=Az,\qquad z(t+\Delta t)=e^{A\Delta t}z(t).
\]

`ContinuousKoopmanCore` 使用真正的 `torch.matrix_exp`，支持 scalar/batch/variable `dt`、
closed-loop rollout、continuous spectrum 和通过 matrix exponential 的 gradient。当前没有 learned
encoder，物理/动力学 state 本身就是 `z`。

## 已完成版本

```text
V0.1  engineering contracts / config / checkpoint / reproducibility
V0.2  trajectory windows / train-only normalization / PhysicsConstraint
V0.3  direct-state continuous-time Koopman generator（当前）
V0.4+ learned lifting / JEPA / closure（未实现）
```

V0.3 同时包含：

- fixed-A closed-form、zero-dt、semigroup 和 gradient 验证；
- damped harmonic oscillator 独立解析 reference；
- variable-dt direct-state system identification；
- deterministic trajectory-level train/test split 与 held-out rollout evaluation；
- detached continuous eigenspectrum diagnostics；
- 100-step closed-loop rollout 与 persistence baseline；
- checkpoint/reload；
- V0.1-compatible run metadata、resolved config 与 structured log；
- unforced Duffing RK4 reference，用于展示固定 `2×2 A` 的 closure limitation；
- CPU-only tests、smoke、13 STEP 教学脚本和 spectrum analyzer。

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

## 范围边界

V0.3 没有 `KoopmanEncoder`、learned lifting、`TrainingDecoder`、JEPA、EMA、`z_r`、residual、
Attention、action-conditioned dynamics、PDE encoder、MPC 或 RL。Duffing 结果不完美是 direct
state fixed linear generator 的预期局限，不能通过偷偷增加 lifting/residual 来修饰。

文档入口：

- [架构规范](./koopman_structured_physical_jepa_world_model_v2_2.md)
- [V0.3 Code Walkthrough](./docs/v0_3_code_walkthrough.md)
- [V0.3 Implementation Checklist](./docs/v0_3_implementation_checklist.md)
- [V0.2 历史导读](./docs/v0_2_code_walkthrough.md)
