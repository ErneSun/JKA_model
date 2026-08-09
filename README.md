# Koopman-Structured Physical JEPA World Model

本仓库当前完成 **V0.2 — Data Windows & Physics Contracts**。项目版本为 `0.2.0`，唯一有效
架构修订为 `2.2`。V0.2 把完整物理 trajectory 安全地变成训练窗口，并提供作用于 raw-unit
state 的具体物理约束；尚未实现任何神经网络或训练逻辑。

## 当前数据主链

```text
analytic trajectories [T+1 states, T transitions]
  -> deterministic split by whole trajectory
  -> fit channel normalizer on train IDs only
  -> window inside each trajectory
  -> ProblemBatch(raw states + model states)
  -> concrete PhysicsConstraint(raw states only)
```

唯一合法时间对齐为：

```text
U_i --(a_i, dt_i)--> U_{i+1}
states  [T+1, ...]
actions [T, d_a] or None
dts     [T]
```

`ProblemBatch` 只使用架构 v2.2 的 canonical 字段，例如 `mu_static` 与 `valid_mask`；不再提供
`parameters`、`mask` 或 aggregate state/action/dt 别名。raw state 保持真实物理单位，model
state 由 train-only normalizer 产生。物理约束与 probe 只能读取 raw state。

## V0.2 已实现

- `TrajectoryRecord` / `TrajectoryDataset` 及静态 `ProblemSpec` 交叉校验；
- trajectory-level deterministic `SplitManifest`，可 JSON round-trip；
- 对输入顺序稳定、对内容敏感的 SHA-256 data fingerprint；
- train-split-only channel standardizer、inverse transform 与 checkpoint state；
- 不跨 trajectory 的 `TrajectoryWindowDataset` 与 `ProblemBatch` collator；
- 解析 Fourier 解的一维周期 advection-diffusion toy 数据，支持 constant/variable `dt`；
- 有限值、状态范围、周期边界、质量守恒及离散 advection-diffusion residual 约束；
- 显式 constraint registry，以及可选 channel mean/RMS probes；
- checkpoint schema 2，保存 split、fingerprint、normalizer 与 constraint specification；
- V0.1 regression smoke、V0.2 end-to-end smoke、教学解释脚本及单元测试。

## 安装

Python 需要 3.10 或更高版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

若本机已有 PyTorch/NumPy，也可以复用系统包：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e . --no-deps
```

## 验证与入口

```bash
source .venv/bin/activate
pytest -q
python scripts/smoke_v0_1.py
python scripts/generate_toy_advection_diffusion.py
python scripts/smoke_v0_2.py
python scripts/explain_v0_2.py
ruff check .
mypy
```

生成可复用的 toy artifact：

```bash
python scripts/generate_toy_advection_diffusion.py --output /tmp/advection_diffusion.pt
```

## 状态设计与范围边界

架构的未来核心 latent 仍只有 `z_k` 与可选 closure/memory `z_r`。物理约束不是 latent，项目
也不存在 `z_phys`。V0.2 **没有**实现 KoopmanEncoder、KoopmanCore、JEPA online/target、
residual closure、attention/Transformer、decoder、trainer、rollout 或模型 loss 聚合。这些属于
V0.3 及之后的版本。

```text
V0.1  工程骨架与契约（完成）
V0.2  数据窗口与具体 PhysicsConstraint（当前完成）
V0.3  Direct-state continuous-time KoopmanCore（未开始）
V0.4+ encoder / PDE baseline / JEPA / closure / fine-tuning（未开始）
```

唯一有效架构规范见
[koopman_structured_physical_jepa_world_model_v2_2.md](./koopman_structured_physical_jepa_world_model_v2_2.md)。
V0.2 代码主导读见 [docs/v0_2_code_walkthrough.md](./docs/v0_2_code_walkthrough.md)，V0.1 历史
导读见 [docs/v0_1_code_walkthrough.md](./docs/v0_1_code_walkthrough.md)。
