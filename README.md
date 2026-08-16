# Koopman-Structured Physical JEPA World Model

本仓库当前完成 **V0.7 本地实现 — residual learnability, closed-loop utility, and memory characterization over a frozen Koopman backbone**。项目版本为 `0.7.0`，
唯一有效架构修订为 `2.2`。

V0.5 在不改变连续时间 core 的前提下，将学习扩展到二维周期 PDE 场：

\[
U^{model}\xrightarrow{E_K}z_K\xrightarrow{e^{A\Delta t}}\hat z_K
\xrightarrow{D_{train}}\hat U^{model}.
\]

`ContinuousKoopmanCore` 使用真正的 `torch.matrix_exp`，支持 scalar/batch/variable `dt`、
closed-loop rollout、continuous spectrum 和通过 matrix exponential 的 gradient。V0.5 新增 circular
CNN encoder/decoder、二维质量守恒与梯形 PDE operator constraint，以及独立 GPU 验证包。

## 已完成版本

```text
V0.1  engineering contracts / config / checkpoint / reproducibility
V0.2  trajectory windows / train-only normalization / PhysicsConstraint
V0.3  direct-state continuous-time Koopman generator
V0.4  learned Koopman coordinates + training decoder
V0.5  2-D PDE field encoder + raw-unit physics training（GPU 已验证）
V0.6  online/EMA-target JEPA shell（多种子 GPU 验证通过）
V0.7  residual identification → learnability → closed-loop utility → multi-H memory characterization（本地通过，GPU 待验证）
```

V0.5 保留全部 V0.1–V0.4 回归能力，并新增：

- endpoint-free 二维周期 advection-diffusion Fourier 解析数据与 Problem Adapter；
- `[B,C,Nx,Ny]` circular CNN encoder、continuous-time Koopman core 与 training decoder；
- 可微 inverse normalization 后的 raw-unit mass / trapezoidal PDE operator loss；
- 唯一 `train_v0_5` / `evaluate_v0_5`、CSV/JSON/plot/report 运行记录与精确 resume；
- CPU numerical/integration/smoke/tiny-overfit 验证和独立 thin-wrapper GPU 验证包。

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
python scripts/smoke_v0_5.py
python scripts/explain_v0_5.py
python scripts/smoke_v0_6.py --device cpu
python scripts/explain_v0_6.py
python -m pytest -q tests/test_v0_7_residual.py tests/test_v0_7_gpu_workflow.py tests/test_v0_7_integration.py
python scripts/explain_v0_7.py
python scripts/train_v0_5.py --config configs/v0_5/advection_diffusion_2d_cpu_tiny_train.yaml --device cpu
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

## V0.7 范围边界

V0.7 冻结最新 V0.6 online encoder、Koopman generator 和 decoder，只识别并预测其 latent residual。
它没有 `z_r`、gate、GRU/LSTM/Mamba/Attention、joint fine-tuning、action-conditioned dynamics、
MPC 或 RL。V0.7 本地 CPU 实现已通过；多种子 GPU scientific acceptance 为 `PENDING_GPU`。

文档入口：

- [架构规范](./koopman_structured_physical_jepa_world_model_v2_2.md)
- [V0.5 文档入口](./docs/v0_5/README.md)
- [V0.5 状态](./docs/v0_5/status.md)
- [V0.6 文档入口](./docs/v0_6/README.md)
- [V0.6 状态](./docs/v0_6/status.md)
- [V0.7 文档入口](./docs/v0_7/README.md)
- [V0.7 状态](./docs/v0_7/status.md)
- [V0.4 Code Walkthrough](./docs/v0_4_code_walkthrough.md)
- [V0.4 Implementation Checklist](./docs/v0_4_implementation_checklist.md)
- [V0.3 Code Walkthrough](./docs/v0_3_code_walkthrough.md)
- [V0.2 历史导读](./docs/v0_2_code_walkthrough.md)
