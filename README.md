# Koopman-Structured Physical JEPA World Model

本项目计划构建一个具有物理锚定、预测性 latent、谱结构动力学与有界 residual
closure 的物理世界模型。当前只完成 **V0.1 — Project Skeleton & Contracts**，项目版本
为 `0.1.0`，架构修订为 `2.1`。

## 当前已实现

- 不可变、可序列化的 `ProblemSpec`；
- 严格时间对齐且区分 raw/model state 的 `ProblemBatch`；
- 公共 latent 命名契约：`z_phys`、`z_k`、`z_r`、`z_k_base`、`delta_z_k`、`z_k_next`；
- 分离 architecture/training/data 的严格 YAML 配置及稳定 SHA-256 hash；
- `TrainStage` 与统一 freeze/optimizer ownership 检查；
- Python、NumPy、PyTorch CPU/CUDA 的 seed 与 RNG state capture/restore；
- 带 schema、架构 revision guard 的 checkpoint round-trip；
- 结构化 run directory 与基础 logging；
- CPU 单元测试和 V0.1 smoke test。

## 安装

Python 需要 3.10 或更高版本。仓库已按 Python 虚拟环境使用方式配置：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

如果本机已经统一安装 PyTorch/NumPy 等大型依赖，也可以像当前开发环境一样创建：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e . --no-deps
```

## 验证

```bash
source .venv/bin/activate
pytest -q
python scripts/smoke_v0_1.py
```

可选静态检查：

```bash
ruff check .
mypy
```

## 时间与单位契约

唯一合法的底层 trajectory 对齐是：

```text
U_i --(a_i, dt_i)--> U_{i+1}
states  [T+1, ...]
actions [T, d_a] or None
dts     [T]
```

`states_raw` 保留真实物理单位，供未来 `PhysicalAnchor`、物理损失和最终指标使用；
`states_model` 是 normalization/preprocessing 后的神经网络输入。代码不会在两者之间自动
转换、reshape、truncate 或补齐。

## 版本路线

```text
V0.1  工程骨架与契约（当前）
V0.2  数据窗口与 PhysicalAnchor
V0.3  Direct-state continuous-time KoopmanCore
V0.4  Learned KoopmanEncoder
V0.5  2D/PDE Koopman-only baseline
V0.6  JEPA online/target shell
V0.7  Residual target + tiny closure
V0.8  Attention closure / z_r
V0.9  Controlled joint fine-tuning
V1.0  Stable research baseline
```

## 当前明确未实现

V0.1 **没有**实现 Dataset/window sampler、Normalizer、PhysicalAnchor、KoopmanEncoder、
KoopmanCore、Transformer/Attention、JEPA、Residual Closure、Decoder、训练循环或 rollout。
这些功能不得在对应后续版本之前被声称可用。

架构原文见 [Koopman-Structured Physical JEPA World Model.md](./Koopman-Structured%20Physical%20JEPA%20World%20Model.md)，
实现决策见 [docs/implementation_notes_v0_1.md](./docs/implementation_notes_v0_1.md)。

