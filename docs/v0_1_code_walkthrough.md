# V0.1 Code Walkthrough

> 历史导读说明：仓库运行时现为 V0.2/`0.2.0`。V0.1 smoke 仍作为回归入口，但
> `ProblemBatch` 旧聚合别名已在 V0.2 移除，checkpoint schema 已升级为 2。当前实现请以
> `docs/v0_2_code_walkthrough.md` 为准。

本文是当前 V0.1 代码的主要阅读入口。内容来自仓库中的实际 Python、YAML、测试和
`scripts/smoke_v0_1.py`，不描述尚未实现的模型。

## 1. 一句话说明 V0.1 在做什么

V0.1 是后续物理世界模型的“数据与运行契约层”：它先固定数据含义、版本兼容、随机状态、
训练阶段所有权和保存格式，再让未来模型在同一套规则上接入。

当前没有实现：

```text
Koopman
JEPA
Residual Closure
Attention
Physics Solver
Neural Encoder
Decoder
```

当前只完成：

```text
data contracts
problem metadata
configuration
reproducibility
train-stage contract
checkpoint
logging
tests
smoke test
```

在训练模型之前先建立这些基础设施，是为了尽早消除最难察觉的工程错误：state/action/dt
错位、raw/model state 混用、配置漂移、随机状态丢失、错误 checkpoint 静默加载，以及冻结
参数被 optimizer 意外更新。这样以后模型预测效果变化时，才能把原因归到模型或数据本身，
而不是隐藏的基础设施差异。V0.1 也让后续版本能够独立替换模型组件而不改变数据语义。

## 2. 项目目录结构

以下目录树忽略 `.venv`、缓存和构建产物：

```text
JKA_model/
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── koopman_structured_physical_jepa_world_model_v2_2.md
├── configs/
│   └── v0_1_smoke.yaml
├── docs/
│   ├── implementation_notes_v0_1.md
│   └── v0_1_code_walkthrough.md
├── scripts/
│   └── smoke_v0_1.py
├── src/jka_model/
│   ├── __init__.py
│   ├── constants.py
│   ├── py.typed
│   ├── config/
│   │   ├── __init__.py
│   │   └── schema.py
│   ├── contracts/
│   │   ├── __init__.py
│   │   ├── batch.py
│   │   └── spec.py
│   ├── physics/
│   │   ├── __init__.py
│   │   └── constraints.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── stages.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── checkpoint.py
│   │   ├── logging.py
│   │   └── seed.py
│   ├── data/__init__.py
│   ├── evaluation/__init__.py
│   └── models/__init__.py
└── tests/
    ├── conftest.py
    ├── test_checkpoint_roundtrip.py
    ├── test_config_roundtrip.py
    ├── test_latent_contracts.py
    ├── test_logging_contract.py
    ├── test_no_z_phys_core_contract.py
    ├── test_physics_constraint_interface.py
    ├── test_problem_batch_contract.py
    ├── test_problem_spec_serialization.py
    ├── test_seed_reproducibility.py
    └── test_train_stage_contract.py
```

重要文件职责：

| 文件 | 当前实际职责 |
|---|---|
| `src/jka_model/constants.py` | 集中定义 project version、architecture revision、checkpoint schema version。 |
| `src/jka_model/contracts/spec.py` | 定义一个物理问题的静态、不可变、可序列化元数据。 |
| `src/jka_model/contracts/batch.py` | 定义 trajectory/window tensor、时间对齐和未来 latent 输出的 shape 契约。 |
| `src/jka_model/config/schema.py` | 把 YAML 严格解析成配置 dataclass，并计算稳定 hash。 |
| `src/jka_model/training/stages.py` | 定义未来训练阶段以及参数 freeze/optimizer ownership 规则。 |
| `src/jka_model/physics/constraints.py` | 仅定义 `PhysicsConstraint` Protocol；没有具体物理方程。 |
| `src/jka_model/utils/seed.py` | 设置、捕获和恢复 Python/NumPy/PyTorch RNG。 |
| `src/jka_model/utils/checkpoint.py` | 定义 checkpoint envelope，并负责原子保存、加载和兼容性检查。 |
| `src/jka_model/utils/logging.py` | 创建 run directory、metadata JSON 和 `run.log`。 |
| `scripts/smoke_v0_1.py` | 当前唯一端到端入口，串起全部 V0.1 基础设施。 |
| `configs/v0_1_smoke.yaml` | smoke 使用的真实配置文件。 |
| `tests/` | 对契约、错误分支和 round-trip 行为做回归保护。 |
| `data/`、`models/`、`evaluation/` | 目前只有说明性 `__init__.py`，没有实现。 |

`src/jka_model/__init__.py` 和各子包 `__init__.py` 只负责重新导出公共 symbol，不包含业务逻辑。
`py.typed` 表示该包提供类型信息。

## 3. 模块调用关系图

当前唯一完整程序入口是 `scripts/smoke_v0_1.py -> main()`：

```text
scripts/smoke_v0_1.py::main()                         [入口]
│
├─ config.load_config(path)                           [utility]
│  └─ YAML mapping
│     └─ ProjectConfig.from_dict()                    [frozen dataclass]
│        ├─ ArchitectureConfig                        [frozen dataclass]
│        ├─ TrainingConfig                            [frozen dataclass]
│        │  └─ TrainStage.KOOPMAN                     [enum]
│        └─ DataConfig                                [frozen dataclass]
│
├─ seed.set_global_seed(seed, deterministic=True)     [utility]
│  ├─ Python random
│  ├─ NumPy RNG
│  ├─ PyTorch CPU RNG
│  └─ PyTorch CUDA RNG（仅 CUDA 可用时）
│
├─ construct ProblemSpec                              [frozen dataclass]
│  ├─ ChannelSpec
│  ├─ GridSpec
│  ├─ BoundarySpec
│  ├─ NormalizationSpec
│  └─ GeometrySpec
│
├─ torch.rand / zeros / full
│  └─ construct ProblemBatch                          [dataclass]
│     └─ __post_init__() -> validate()                [自动 shape/alignment 校验]
│
├─ construct SmokeConstraint                          [toy implementation]
│  ├─ isinstance(..., PhysicsConstraint)              [runtime-checkable Protocol]
│  └─ loss(raw state) -> {"interface_zero": scalar}
│
└─ tempfile.TemporaryDirectory
   ├─ logging.get_git_commit(repository_root)          [utility]
   ├─ logging.create_run_directory(...)               [utility]
   │  ├─ RunContext                                   [frozen dataclass]
   │  ├─ run_metadata.json
   │  └─ run.log
   ├─ config.save_config(...)                         [utility]
   │  └─ resolved_config.yaml
   ├─ seed.capture_rng_state()                        [utility]
   │  └─ RNGState                                     [dataclass]
   ├─ construct Checkpoint                            [dataclass]
   │  ├─ references ProblemSpec
   │  ├─ references ProjectConfig
   │  └─ references RNGState
   ├─ checkpoint.save_checkpoint(...)                 [utility]
   │  └─ checkpoint.pt（临时文件 + atomic replace）
   ├─ checkpoint.load_checkpoint(...)                 [utility]
   │  ├─ schema/revision/version guard
   │  └─ Checkpoint.from_payload()
   ├─ compare restored.problem_spec / restored.config
   └─ print summary JSON
```

`configure_train_stage()` 没有被 smoke 调用。它当前只由
`tests/test_train_stage_contract.py` 使用 toy `nn.Module` 验证未来行为。

## 4. 一次 smoke test 到底发生了什么

运行：

```bash
python scripts/smoke_v0_1.py
```

### Step 1：进入 `main()` 并定位仓库

入口是 `scripts/smoke_v0_1.py::main()`。代码用脚本路径的父目录得到 `repository_root`，
没有依赖当前 shell 的相对路径。

### Step 2：加载 YAML config

`load_config(repository_root / "configs" / "v0_1_smoke.yaml")` 读取 YAML，依次构造：

- `ArchitectureConfig(revision="2.2", package="jka_model")`
- `TrainingConfig(seed=7, stage=KOOPMAN, deterministic=True, run_root="runs")`
- `DataConfig(problem_name="toy_scalar_field", action_dim=1, parameter_dim=1, ...)`
- 外层 `ProjectConfig(project_version="0.2.0", tags=("v0.1", "smoke"))`

未知字段会失败；revision 或 project version 不兼容也会失败。

### Step 3：设置全局随机种子

`set_global_seed(7, deterministic=True)` 同步设置 Python、NumPy、PyTorch CPU，以及可用时
的 CUDA RNG，并打开 PyTorch deterministic algorithms。

### Step 4：构造 `ProblemSpec`

smoke 在 Python 中创建静态问题定义：单一 `temperature [K]` channel、一维 4 点周期网格、
间距 `0.25`、固定 `dt=0.1`、action/parameter dimension 均为 1。这个对象没有 batch tensor。

注意：当前 YAML 没有包含 channels/grid/boundary；smoke 的 `ProblemSpec` 是独立构造的。

### Step 5：构造 raw/model tensors 与 `ProblemBatch`

局部常量为 `B=2, H=3, K=2`：

```text
states_raw   = 273.15 + rand([2,5,1,4])
states_model = (states_raw - 273.15) / 10
```

随后分成 context `[:, :3]` 和 future `[:, 3:]`，并加入：

- history actions `[2,2,1]`
- future actions `[2,2,1]`
- history dts `[2,2]`
- future dts `[2,2]`
- static parameters `[2,1]`
- cell weights `[4]`
- trajectory IDs `['toy-0', 'toy-1']`

`ProblemBatch.__post_init__()` 立即调用 `validate()`。任何 raw/model shape 不一致、action/dt
错位、非正 dt 或 batch size 不一致都会在这里失败，不会被自动修复。

### Step 6：验证 `PhysicsConstraint` 接口

`SmokeConstraint` 是脚本中的 toy class，不是 PDE 实现。它满足 Protocol 的 `loss()` 签名，
对一个 future raw state 返回标量零：

```python
{"interface_zero": pred_state_raw.sum() * 0.0}
```

这一步只证明接口可导入、可由 structural typing 实现；没有检查 Navier–Stokes 或任何守恒律。

### Step 7：创建临时 run directory

`TemporaryDirectory(prefix="jka-v0-1-")` 先创建系统临时根目录，随后：

```python
create_run_directory(..., run_id="smoke-v0-1")
```

得到：

```text
<temporary_root>/smoke-v0-1/
├── run_metadata.json
└── run.log
```

smoke 没有使用 `config.training.run_root == "runs"`；它显式把 temporary root 传给函数。
因此正常退出 `with` block 后，整个 smoke run directory 会被删除。

### Step 8：保存 resolved config

`save_config(config, run.run_dir / "resolved_config.yaml")` 把已经解析和补齐默认值的配置写入
run directory。保存使用排序后的 YAML key。

### Step 9：捕获 RNG 并创建 `Checkpoint`

`capture_rng_state()` 生成 `RNGState`。随后 `Checkpoint(...)` 保存：

- `epoch=0`, `global_step=0`
- 当前 `TrainStage.KOOPMAN`
- `ProblemSpec` 与 `ProjectConfig`
- 自动计算的 `config_hash`
- toy normalizer state、data fingerprint、split manifest
- Git commit（仓库可用时）
- model/optimizer/scheduler state 均为 `None`

### Step 10：原子保存 checkpoint

`save_checkpoint()` 先在目标目录写隐藏临时文件，再用 `Path.replace()` 变成
`checkpoint.pt`。异常时残留临时文件会被清理。

### Step 11：重新加载 checkpoint

`load_checkpoint()` 默认加载到 CPU，随后 `Checkpoint.from_payload()` 检查 schema、architecture
revision 和 project version，并重建 `RNGState`、`ProblemSpec`、`ProjectConfig`。

当前加载使用 `weights_only=False`，所以只应读取可信的本地 checkpoint。

### Step 12：判定成功

脚本要求：

```python
restored.problem_spec == spec
restored.config == config
```

否则显式抛出 `RuntimeError`。最后打印 metadata JSON，其中
`checkpoint_roundtrip=true` 且 `physics_constraint_interface=true`。正常退出即说明完整 V0.1
路径可运行。临时目录随后自动删除，所以 smoke 不留下持久 run artifact。

## 5. V0.1 Data Dictionary

### 5.1 对象总览

| 对象 | Python 类型 | 保存什么 | 不保存什么 |
|---|---|---|---|
| `ProblemSpec` | frozen dataclass | 问题的静态物理/网格/单位定义 | batch tensor、拟合 normalizer、模型参数 |
| `ProblemBatch` | dataclass | 一次 window 的 raw/model tensors、action、dt、参数和几何数据 | `ProblemSpec`、latent、loss、模型输出 |
| `ProjectConfig` | frozen dataclass | architecture/training/data 配置和 tags | `ProblemSpec`、batch tensor、模型实例 |
| `LatentState` | dataclass | 未来接口的 `z_k`、optional `z_r` | 物理状态、第三个 physical latent |
| `TransitionOutput` | dataclass | 未来 transition 的 base/correction/gate/final 输出 | 当前 V0.1 的实际预测结果 |
| `Checkpoint` | dataclass | resume/兼容性所需的统一 envelope | 当前没有真实 trainer/model state |
| `TrainStage` | string enum | 四个允许的阶段名称 | 训练循环本身 |
| `PhysicsConstraint` | runtime-checkable Protocol | 一个 `loss()` 方法签名 | 具体 PDE/BC/守恒实现 |
| `RunContext` | frozen dataclass | run identity 与追踪 metadata | config/checkpoint 对象本身 |
| `RNGState` | dataclass | 四类 RNG 的瞬时状态 | seed 值的语义说明或 DataLoader sampler 状态 |

### 5.2 `ProblemSpec`：这个物理问题是什么

最关键的区分：**`ProblemSpec` 描述“这个物理问题是什么”，`ProblemBatch` 描述“这一次给
未来模型的数据是什么”。**

#### `ProblemSpec` 字段

| 字段 | 类型 | 必需 | 当前含义 |
|---|---|---:|---|
| `name` | `str` | 是 | 问题名称，不能为空。 |
| `channels` | `tuple[ChannelSpec, ...]` | 是 | 状态 channel 的名称和单位；至少一个且名称唯一。 |
| `spatial_dim` | `int` | 是 | 空间维数，可为 0，不能为负。 |
| `grid` | `GridSpec` | 是 | grid layout、shape、spacing 和数据要求。 |
| `boundary` | `BoundarySpec` | 是 | 边界类型及只读 metadata；不实现边界算子。 |
| `action_dim` | `int` | 否，默认 0 | action vector 的维数。 |
| `parameter_dim` | `int` | 否，默认 0 | static parameter `mu` 的维数。 |
| `dt_mode` | `DtMode` | 否 | `constant` 或 `variable`。 |
| `constant_dt` | `float | None` | 条件必需 | constant mode 必须为正；variable mode 必须为 `None`。 |
| `normalization` | `NormalizationSpec` | 否 | 声明 preprocessing 方法；不是拟合统计值。 |
| `geometry` | `GeometrySpec` | 否 | 声明是否要求 mask 及 geometry metadata。 |
| `observable_requirements` | `tuple[str, ...]` | 否 | 未来可选 observable/probe 的名称要求；当前不计算。 |
| `metadata` | `Mapping[str, Any]` | 否 | JSON-compatible 扩展 metadata，构造时递归冻结。 |

#### 子规格对象

| 对象 | 字段 | 验证/用途 |
|---|---|---|
| `ChannelSpec` | `name`, `unit` | 两者必须为非空字符串。 |
| `GridSpec` | `layout` | 只能是 `channels_first` 或 `channels_last`。 |
| `GridSpec` | `shape` | optional tuple；每项为正，长度必须等于 `spatial_dim`。 |
| `GridSpec` | `spacing` | optional tuple；每项为正，长度必须等于 `spatial_dim`。 |
| `GridSpec` | `coordinates_required` | 声明未来 batch 是否必须提供 coordinates；V0.1 不交叉校验。 |
| `GridSpec` | `cell_weights_required` | 声明未来是否需要积分权重；V0.1 不交叉校验。 |
| `BoundarySpec` | `kind`, `metadata` | 只描述边界，不执行边界条件。 |
| `NormalizationSpec` | `method`, `metadata` | 只声明方法；拟合状态在 checkpoint 的 `normalizer_state`。 |
| `GeometrySpec` | `mask_required`, `metadata` | 只声明要求；mask tensor 位于 `ProblemBatch.valid_mask`。 |

`ProblemSpec.to_dict()` 将 enum/tuple/只读 metadata 转成 YAML/JSON-compatible 数据；
`from_dict()` 拒绝未知字段并重建对象。它是 frozen dataclass，metadata 也通过
`MappingProxyType` 和 tuple 递归冻结。

### 5.3 `ProblemBatch`：这一次的数据是什么

记 `B` 为 batch size，`H` 为 context state 数，`K` 为 future state 数，`*state` 为一个
状态的所有剩余轴。

| 字段 | 类型 | Shape | 单位 | 必需 | 来源/用途 |
|---|---|---|---|---:|---|
| `context_states_raw` | `Tensor` | `[B,H,*state]` | 真实物理单位 | 是 | context，末尾是当前状态 `U_t`；未来 physics/probe/metric 使用。 |
| `future_states_raw` | `Tensor` | `[B,K,*state]` | 真实物理单位 | 是 | `U_{t+1}...U_{t+K}`；未来作为 target/metric。 |
| `context_states_model` | `Tensor` | 与 context raw 完全相同 | preprocessing 后尺度 | 是 | 未来 neural encoder 输入。 |
| `future_states_model` | `Tensor` | 与 future raw 完全相同 | preprocessing 后尺度 | 是 | 未来 representation/训练 target。 |
| `history_dts` | `Tensor` | `[B,H-1]` | 数据定义的时间单位 | 是 | context 内相邻状态的 transition interval。 |
| `future_dts` | `Tensor` | `[B,K]` | 数据定义的时间单位 | 是 | 从 `U_t` 开始的 K 个 transition interval。 |
| `history_actions` | `Tensor | None` | `[B,H-1,d_a]` | 当前代码未声明 | 否 | context 内 transition 的 action。 |
| `future_actions` | `Tensor | None` | `[B,K,d_a]` | 当前代码未声明 | 否 | `future_actions[:,0]` 驱动 `U_t -> U_{t+1}`。 |
| `mu_static` | `Tensor | None` | `[B,d_mu]` | 当前代码未声明 | 否 | 每个 sample 的静态参数。 |
| `coordinates` | `Tensor | None` | 当前未强制 | 问题相关空间单位 | 否 | grid/mesh coordinates；V0.1 只检查它是 Tensor。 |
| `cell_weights` | `Tensor | None` | 当前未强制 | 问题相关体积/面积单位 | 否 | quadrature weights/cell volumes；V0.1 只检查类型。 |
| `valid_mask` | `Tensor | None` | 当前未强制 | 无单位 | 否 | geometry/domain validity mask；V0.1 只检查类型。 |
| `trajectory_id` | `object | None` | sequence 时长度必须为 B | 不适用 | 否 | 追踪 sample 来源；list/tuple 会检查长度。 |

#### Canonical-only 命名

V0.2 兼容清理后只保留上表中的 dataclass 字段。需要完整窗口时，调用方应显式拼接
`context_*`/`future_*`；`parameters` 与 `mask` 等旧别名不再存在。这样 `mu_static`、
`valid_mask` 等 v2.2 名称只有一个公开入口。

#### `states_raw`

它保留真实物理变量及单位，例如 `temperature [K]`、`rho [kg/m^3]`。当前 smoke 人工生成
`273.15 + random`。未来 `PhysicsConstraint`、可选 `PhysicalProbe` 和 raw-unit metric 应读取
它。V0.1 不会自动做单位转换。

#### `states_model`

它与 raw state shape 完全一致，但数值已经经过 normalization/preprocessing。smoke 中只是
`(states_raw - 273.15) / 10`；仓库尚无正式 Normalizer。它是未来 neural encoder 的输入。
normalized 值不具有 raw 物理单位，不能直接拿来计算质量、能量或 PDE residual。

#### `actions` 与 `dts`

唯一时间语义是：

\[
U_i \xrightarrow{a_i,\Delta t_i} U_{i+1}.
\]

因此 `actions[i]` 与 `dts[i]` 都属于从第 i 个 state 到第 i+1 个 state 的同一 transition。
window 中：

```text
context states : U_{t-H+1}, ..., U_t                  [H states]
history actions: a_{t-H+1}, ..., a_{t-1}              [H-1 transitions]
history dts    : dt_{t-H+1}, ..., dt_{t-1}            [H-1 transitions]
future actions : a_t, ..., a_{t+K-1}                  [K transitions]
future dts     : dt_t, ..., dt_{t+K-1}                [K transitions]
future states  : U_{t+1}, ..., U_{t+K}                [K states]
```

actions 要么 history/future 都存在，要么都为 `None`。所有 dt 必须 finite 且大于 0。

#### `ProblemBatch.to()` 的当前行为

`to(*args, **kwargs)` 对每个 Tensor 原样调用 `Tensor.to()`，再构造并验证一个新 batch；
`trajectory_id` 等非 Tensor 原样保留。它不会 normalize、reshape 或转换单位。因为调用被统一
应用到全部 Tensor，如果显式传 `dtype=...`，mask、action、dt 等 Tensor 也会收到同一 dtype
请求；当前代码没有按字段区分 dtype。

### 5.4 Latent/transition 预留对象

这些 dataclass 固定未来公共 shape，但 V0.1 不产生它们：

| 对象/字段 | Shape | 当前验证 |
|---|---|---|
| `LatentState.z_k` | `[B,d_k]` | 必须是二维。 |
| `LatentState.z_r` | `[B,d_r]` 或 `None` | 若存在，必须二维且 batch size 与 `z_k` 一致。 |
| `TransitionOutput.z_k_base` | `[B,d_k]` | 必须二维。 |
| `TransitionOutput.z_r` | `[B,d_r]` 或 `None` | batch size 必须匹配。 |
| `TransitionOutput.delta_z_k` | `[B,d_k]` | 必须与 `z_k_base` 完全同 shape。 |
| `TransitionOutput.gate` | `[B,1]` | 当前固定为 scalar gate。 |
| `TransitionOutput.z_k_next` | `[B,d_k]` | 必须与 `z_k_base` 完全同 shape。 |

## 6. Tensor Shape Convention

### 6.1 轴符号

| 符号 | 含义 |
|---|---|
| `T` | 一条未 batch trajectory 的 transition 数。 |
| `B` | batch size。 |
| `H` | context 中的 state 数。 |
| `K` | future horizon 中的 state/transition 数。 |
| `C` | 约定中的 channel 轴；`ProblemBatch` 本身不定位该轴。 |
| `d_a` | action dimension。 |
| `d_mu` | static parameter dimension。 |
| `d_k`, `d_r` | 未来 Koopman / closure latent dimension。 |
| `*state` | 一个 state 的全部轴，可为 vector、channel+grid 或其它结构。 |
| `*spatial` | `ProblemSpec.spatial_dim` 描述的空间轴；batch 不单独解析。 |

### 6.2 底层 trajectory

`validate_trajectory_alignment()` 只处理未 batch 的底层 trajectory：

```text
states  : [T+1, ...]
actions : [T,d_a] or None
dts     : [T]
```

代码允许 `states` 在时间轴后具有任意 shape，但要求至少两个 states。`actions` 若存在必须
恰为二维；`dts` 必须一维、finite、positive。

### 6.3 batch/window

```text
context_states_* : [B,H,*state]
future_states_*  : [B,K,*state]
history_actions  : [B,H-1,d_a] or None
future_actions   : [B,K,d_a] or None
history_dts      : [B,H-1]
future_dts       : [B,K]
mu_static        : [B,d_mu] or None
```

state tensor 必须至少三维，即 `[B,time,至少一个 state 轴]`。对于规则网格，常见的
`*state` 可以是 `[C,Nx,Ny]`，但 V0.1 没有硬编码二维网格，也没有强制 channel-first；
实际 layout 由 `ProblemSpec.grid.layout` 描述。

## 7. Configuration 调用关系

```text
configs/v0_1_smoke.yaml
          │ yaml.safe_load
          ▼
plain Mapping[str, Any]
          │ strict from_dict + validation
          ▼
ProjectConfig
├─ ArchitectureConfig
├─ TrainingConfig ── references TrainStage
└─ DataConfig ────── references DtMode
          │
          ├─ to_dict() ──> yaml.safe_dump ──> resolved_config.yaml
          │
          └─ canonical JSON(sort_keys, compact separators)
                         │ SHA-256
                         ▼
                    config_hash
                    ├─ RunContext
                    └─ Checkpoint
```

### 7.1 配置对象查阅表

| 对象 | 字段 | 当前用途 |
|---|---|---|
| `ArchitectureConfig` | `revision`, `package` | revision 必须等于运行时 `2.2`；记录 package identity。 |
| `TrainingConfig` | `seed` | smoke 用于 `set_global_seed()`。 |
| `TrainingConfig` | `stage` | smoke 写入 run/checkpoint metadata；没有启动 trainer。 |
| `TrainingConfig` | `deterministic` | smoke 传给 `set_global_seed()`。 |
| `TrainingConfig` | `run_root` | 当前 YAML 为 `runs`，但 smoke 没有使用，改用临时目录。 |
| `DataConfig` | `problem_name` | 配置身份；smoke 另行构造同名 `ProblemSpec`。 |
| `DataConfig` | `action_dim`, `parameter_dim` | 严格校验非负；当前不自动和 `ProblemSpec`/batch 交叉检查。 |
| `DataConfig` | `dt_mode`, `constant_dt` | 校验 constant/variable dt 配置组合。 |
| `DataConfig` | `normalization` | V0.1 字符串由迁移解析器转成 V0.2 `NormalizationConfig`。 |
| `ProjectConfig` | `architecture`, `training`, `data` | 聚合三类配置。 |
| `ProjectConfig` | `project_version` | 当前回归运行必须等于运行时 `0.2.0`。 |
| `ProjectConfig` | `tags` | 追踪/分类 metadata。 |

`stable_config_hash()` 的目的，是让 run 和 checkpoint 能证明“使用的是完全相同的 resolved
config”。它先生成 key 排序、无多余空白的 canonical JSON，再计算 SHA-256，因此 mapping 的
key 插入顺序不会改变结果。`Checkpoint` 若同时收到 `config` 和错误的 `config_hash` 会拒绝构造。

当前 YAML 只包含基础字段，不包含 optimizer、loss、model architecture 或 dataset path；这些
不是隐藏默认值，而是尚未实现。

## 8. TrainStage 查阅表

| Stage | V0.1 是否真正训练 | `configure_train_stage()` 标记为 trainable 的 canonical groups | 未来语义 |
|---|---:|---|---|
| `KOOPMAN` | 否，仅契约/test | `koopman_encoder`, `online_encoder`, `koopman_core`, `training_decoder` | Koopman representation/core 阶段。 |
| `JEPA` | 否，仅契约/test | 与 `KOOPMAN` 相同 | online/target representation 阶段；target 仍冻结。 |
| `RESIDUAL` | 否，仅契约/test | `residual_memory`, `residual_head`, `gate` | 冻结结构分支后训练 closure。 |
| `JOINT` | 否，仅契约/test | Koopman groups + residual groups + `training_decoder` | 未来联合微调。 |

`target_encoder` 是已知 group，但任何 stage 都不会把它设为 trainable。

`configure_train_stage()` 的真实行为：

1. 从 model 的 canonical child names 自动发现 groups，或调用 model 提供的
   `train_stage_modules()` mapping；
2. 拒绝未知 group；
3. 确认每个 parameter 恰好由一个 group 拥有；
4. 拒绝 unowned 或重复 owned parameter；
5. 统一调用 `module.requires_grad_(...)`；
6. 返回 `{group_name: bool}`，但不创建 optimizer。

`assert_optimizer_matches_trainable_params()` 再验证 optimizer 中的 parameter 与
`requires_grad=True` parameter 精确相等且没有重复。

**V0.1 中 `TrainStage` 是未来训练状态机的接口，不是当前已经存在四套训练流程。**

## 9. PhysicsConstraint 查阅表

`PhysicsConstraint` 是 `@runtime_checkable Protocol`，当前只有一个方法：

| 参数/返回值 | 类型 | 当前语义 |
|---|---|---|
| `pred_state_raw` | `Tensor` | 必需；应是 raw-unit predicted physical state。 |
| `prev_state_raw` | `Tensor | None` | optional previous raw state。 |
| `action` | `Tensor | None` | optional aligned action。 |
| `dt` | `Tensor | None` | optional aligned transition interval。 |
| `spec` | `ProblemSpec | None` | optional static problem metadata；只是引用，不被 Protocol 拥有。 |
| `metadata` | `Mapping[str, Any] | None` | optional call-specific metadata。 |
| 返回值 | `Mapping[str, Tensor]` | named physical loss tensors。 |

接口与实现必须区分：

```text
PhysicsConstraint Protocol  = 已实现
Navier–Stokes constraint    = Not implemented in V0.1
PDE residual                = Not implemented in V0.1
boundary loss               = Not implemented in V0.1
conservation loss           = Not implemented in V0.1
```

`runtime_checkable` 的 `isinstance()` 主要检查结构上是否存在方法，不能在运行时证明输入真的
使用 raw units，也不能证明返回值具有正确物理意义。smoke/test 中的零 loss 只是接口 fixture。

## 10. Checkpoint 完整数据表

`Checkpoint` 是 epoch-boundary resume envelope。当前 schema version 为 2。

| 字段 | 保存什么 | 为什么保存 | smoke 中的真实值 |
|---|---|---|---|
| `schema_version` | checkpoint 格式版本 | 防止字段布局不兼容 | `2` |
| `architecture_revision` | 模型/契约架构 revision | 拒绝旧架构静默加载 | `"2.2"` |
| `project_version` | 软件项目版本 | 保证 runtime 兼容 | `"0.2.0"` |
| `train_stage` | `TrainStage` | 知道保存时处于哪个训练阶段 | `KOOPMAN` |
| `online_model_state` | mapping 或 `None` | 未来 online model 参数 | `None` |
| `target_model_state` | mapping 或 `None` | 未来 target/EMA model 参数 | `None` |
| `optimizer_state` | mapping 或 `None` | 未来恢复 optimizer | `None` |
| `scheduler_state` | mapping 或 `None` | 未来恢复 scheduler | `None` |
| `amp_scaler_state` | mapping 或 `None` | 未来 mixed precision scaler | 默认 `None` |
| `epoch` | 非负 int | epoch-boundary resume 位置 | `0` |
| `global_step` | 非负 int | 全局 step 计数 | `0` |
| `rng_state` | `RNGState | None` | 恢复随机数流 | `capture_rng_state()` 实值 |
| `normalizer_state` | mapping 或 `None` | 恢复拟合 normalization 统计 | toy mean/scale dict |
| `problem_spec` | `ProblemSpec | None` | 恢复问题静态语义 | toy temperature spec |
| `config` | `ProjectConfig | None` | 恢复完整 resolved config | smoke config |
| `config_hash` | `str | None` | 检测 config 漂移/篡改 | 由 config 自动计算 |
| `data_fingerprint` | `str | None` | 未来确认数据身份 | `toy-v0.1-deterministic` |
| `split_manifest` | 任意可保存对象 | 未来恢复 train/val/test 划分 | toy ID dict |
| `physics_constraint_spec` | 可序列化 constraint 声明 | 恢复物理约束配置 | V0.1 smoke 为 `None` |
| `git_commit` | `str | None` | 追踪代码版本 | 当前 Git HEAD 或 `None` |

`to_payload()` 会写出上述所有字段，包括 `amp_scaler_state`。加载时 required-field set 没有要求
旧 payload 必须含 `amp_scaler_state`，缺失时使用 `None`；其余主要 schema 字段缺失会失败。

即使 V0.1 没有模型，也要先定义 checkpoint contract，是因为后续版本若各自发明保存格式，
就无法可靠比较实验或恢复训练。现在 round-trip 已经保护 version、config、problem metadata、
RNG 和 future state slots；未来只需向已有 slot 填入真实 state。

## 11. Reproducibility 调用链

```text
set_global_seed(seed, deterministic)
├─ random.seed(seed)
├─ numpy.random.seed(seed)
├─ torch.manual_seed(seed)
├─ torch.cuda.manual_seed_all(seed)       [CUDA available only]
├─ torch.use_deterministic_algorithms(...)
└─ cudnn.deterministic / cudnn.benchmark

capture_rng_state()
├─ random.getstate()
├─ numpy.random.get_state()
├─ torch.get_rng_state()
└─ torch.cuda.get_rng_state_all()         [CUDA available only]
             │
             ▼
          RNGState
             │ checkpoint serialization
             ▼
restore_rng_state(state)
├─ random.setstate(...)
├─ numpy.random.set_state(...)
├─ torch.set_rng_state(...)
└─ torch.cuda.set_rng_state_all(...)
```

`RNGState` 字段：

| 字段 | 类型 | 内容 |
|---|---|---|
| `python` | `tuple[Any, ...]` | Python `random` 内部状态。 |
| `numpy` | `tuple[Any, ...]` | legacy NumPy global RNG 状态。 |
| `torch_cpu` | `Tensor` | PyTorch CPU RNG bytes。 |
| `torch_cuda` | `tuple[Tensor, ...] | None` | 每个 CUDA device 的 RNG state。 |

恢复 CUDA RNG 时，当前机器必须有 CUDA，且 device count 与捕获时相同，否则显式失败。
V0.1 不保存 DataLoader sampler/mid-batch state，因此只承诺 epoch-boundary resume 基础。

## 12. Logging / Run Directory

`create_run_directory(root, ...)` 的命名逻辑：

- 如果调用者传 `run_id`，必须匹配 `[A-Za-z0-9][A-Za-z0-9_.-]*`；
- 如果不传，自动生成 UTC timestamp 加 8 位 UUID，例如
  `20260809T120000Z-a1b2c3d4`；
- 最终目录是 `Path(root) / run_id`，必须尚不存在。

`RunContext` / `run_metadata.json` 字段：

| 字段 | 来源 |
|---|---|
| `run_id` | 显式传入或自动生成。 |
| `run_dir` | 新目录的 absolute resolved path。 |
| `project_version` | `constants.PROJECT_VERSION`。 |
| `architecture_revision` | `constants.ARCHITECTURE_REVISION`。 |
| `seed` | 调用参数。 |
| `config_hash` | 调用参数，通常来自 `ProjectConfig.stable_hash`。 |
| `git_commit` | 调用参数，通常来自 `get_git_commit()`。 |
| `train_stage` | 调用参数，序列化为 enum value。 |

一次持久 run 的预期文件位置由调用者决定；utility 本身只自动创建：

```text
<root>/<run_id>/
├── run_metadata.json    # create_run_directory 自动写
└── run.log              # create_run_directory 自动写
```

`resolved_config.yaml` 和 `checkpoint.pt` 不是 logging utility 自动生成的；smoke 随后分别调用
`save_config()`、`save_checkpoint()` 把它们写到同一目录。metrics CSV、TensorBoard、W&B、
artifact registry：**Not implemented in V0.1**。

smoke 使用 temporary root，所以这些文件只在脚本运行期间存在。

## 13. Tests 查阅表

当前执行 `pytest` 共收集 30 个 test case，其中 `TrainStage` 参数化测试贡献多个 case。

| Test file | 测试对象 | 防止的 bug |
|---|---|---|
| `conftest.py` | `toy_problem_spec`, `toy_config` fixtures | 统一测试输入，避免每个测试使用不一致的 spec/config。 |
| `test_problem_batch_contract.py` | batch aliases、shape、dt/action alignment、`.to()` | 防止 state/action/dt 差一位、只提供一半 action、raw/model 混淆以及 device/dtype transfer 丢字段。 |
| `test_problem_spec_serialization.py` | spec round-trip、冻结 metadata、strict fields、dt mode | 防止单位/网格 metadata 丢失、外部修改或非法 constant/variable dt 组合。 |
| `test_config_roundtrip.py` | YAML round-trip、stable hash、unknown fields | 防止配置 key 顺序改变 hash、未知字段被静默接受或保存后配置漂移。 |
| `test_seed_reproducibility.py` | seed 与 RNG capture/restore | 防止同 seed 产生不同 Python/NumPy/Torch 序列，或 resume 后随机流错位。 |
| `test_checkpoint_roundtrip.py` | payload round-trip、revision guard、hash guard | 防止 metadata/state 丢失、v2.1 checkpoint 静默加载或 config 被篡改。 |
| `test_train_stage_contract.py` | enum、freeze policy、optimizer ownership | 防止 target encoder 被训练、frozen 参数进入 optimizer、参数无人负责/重复管理，或 frozen 参数发生更新。 |
| `test_latent_contracts.py` | `LatentState`/`TransitionOutput` shape | 防止 latent batch 不一致、correction shape 不一致或 gate 误变成 vector gate。 |
| `test_no_z_phys_core_contract.py` | v2.2 latent fields | 防止废弃的第三个 physical latent 被重新加入公共状态。 |
| `test_physics_constraint_interface.py` | Protocol structural typing 与 named scalar mapping | 防止物理接口无法被独立 toy implementation 实现；不验证真实物理。 |
| `test_logging_contract.py` | run directory/metadata/log file | 防止 run identity、revision、seed、stage 或 log artifact 漏写。 |

## 14. 核心对象关系图

```text
ProblemSpec [owns]
├─ tuple[ChannelSpec]
├─ GridSpec
├─ BoundarySpec
├─ NormalizationSpec
└─ GeometrySpec

ProblemBatch [independent data container]
├─ raw/model state tensors
├─ action/dt tensors
├─ optional parameter/geometry tensors
└─ trajectory IDs
    (does NOT own/reference ProblemSpec or Config)

ProjectConfig [owns]
├─ ArchitectureConfig
├─ TrainingConfig ──references──> TrainStage
└─ DataConfig ───────references──> DtMode

Checkpoint [optionally references/serializes]
├─ ProblemSpec
├─ ProjectConfig
├─ RNGState
├─ TrainStage
└─ model/optimizer/scheduler mappings or None

RunContext
├─ copies version/seed/hash/stage identity
└─ does NOT own ProjectConfig or Checkpoint

PhysicsConstraint.loss(...)
└─ may receive ProblemSpec as a call argument
   (Protocol does NOT own ProblemSpec or ProblemBatch)
```

模块解耦事实：

- `contracts/spec.py` 不导入 PyTorch，也不知道 config/checkpoint。
- `ProblemBatch` 不导入 `ProblemSpec`，所以不会自动比较 channel/action/parameter dimensions。
- config 知道 `DtMode` 和 `TrainStage`，但不知道 batch、checkpoint 或 logging。
- logging 只接收 seed/hash/stage 等值，不加载 config。
- checkpoint 是聚合边界，因此知道 config、spec、stage 和 RNG。
- training stage 逻辑不知道 `ProblemBatch`、physics 或 checkpoint。
- physics Protocol 只知道 `Tensor` 与 optional `ProblemSpec`。

## 15. Recommended Reading Order

建议在 30–60 分钟内按以下顺序阅读：

1. **`scripts/smoke_v0_1.py`**：先看到所有模块怎样在一次真实运行中串起来。
2. **`contracts/spec.py`**：理解静态问题、channel、单位、grid 与 dt semantics。
3. **`contracts/batch.py`**：重点理解 raw/model 分离和 state/action/dt 对齐。
4. **`configs/v0_1_smoke.yaml` + `config/schema.py`**：从文件追到 Python 对象和 hash。
5. **`utils/seed.py`**：理解 deterministic seed 与 checkpoint RNG state 的关系。
6. **`utils/checkpoint.py`**：查看所有对象最终怎样序列化、校验和恢复。
7. **`utils/logging.py`**：理解 run directory 由谁创建、哪些文件自动产生。
8. **`training/stages.py`**：区分“阶段契约”和“真实训练流程”。
9. **`physics/constraints.py`**：确认当前只有 Protocol，没有方程实现。
10. **`tests/`**：以失败案例反向理解每条契约为什么存在。

最后再读 `README.md` 和架构 v2.2，前者提供项目级摘要，后者描述未来数学路线；不要用未来
架构内容替代对当前代码的判断。

## 16. Top 10 Symbols to Understand

| Symbol | File | 一句话作用 |
|---|---|---|
| `main` | `scripts/smoke_v0_1.py` | 当前唯一端到端入口。 |
| `ProblemSpec` | `contracts/spec.py` | 定义物理问题的静态身份、单位、网格和时间语义。 |
| `ProblemBatch` | `contracts/batch.py` | 保存并验证一次 trajectory window 的所有 tensor。 |
| `validate_trajectory_alignment` | `contracts/batch.py` | 保护底层 `[T+1] states / [T] transitions` 契约。 |
| `ProjectConfig` / `load_config` / `stable_config_hash` | `config/schema.py` | 将 YAML 严格转换为三部分配置，并生成稳定 SHA-256。 |
| `TrainStage` / `configure_train_stage` | `training/stages.py` | 统一未来 freeze/unfreeze 的阶段策略。 |
| `PhysicsConstraint` | `physics/constraints.py` | 定义 raw-unit physical loss 的最小 Protocol。 |
| `Checkpoint` / `save_checkpoint` / `load_checkpoint` | `utils/checkpoint.py` | 定义并执行带兼容性 guard 的持久化 round-trip。 |
| `set_global_seed` / `capture_rng_state` / `restore_rng_state` | `utils/seed.py` | 控制可复现随机流及 resume 状态。 |
| `RunContext` / `create_run_directory` | `utils/logging.py` | 建立 run identity、metadata JSON 和 log 文件。 |

## 17. What V0.1 Does NOT Do

当前明确没有发生：

- 没有读取 CFD、FEM 或实验 dataset；
- 没有 dataset loader、trajectory split 或 window sampler；
- 没有正式 Normalizer，smoke 只手工做一次线性缩放；
- 没有训练任何 neural network；
- 没有 Neural Encoder，也没有从 state 构建 `z_k`；
- 没有 ResidualMemory，也没有从历史构建 `z_r`；
- 没有 Koopman matrix/generator/core；
- 没有 JEPA online/target encoder 或 EMA update；
- 没有 Attention、GRU 或 residual closure；
- 没有 TrainingDecoder 或 field decoder；
- 没有 PDE residual、boundary loss、conservation loss 或 physics projection；
- 没有正式 optimizer/scheduler/trainer；
- 没有 teacher-forced 或 closed-loop rollout；
- 没有训练 metric、科学验收结果或 baseline comparison；
- 没有 persistent smoke artifacts，因为 temporary directory 会被清除。

当前的 `LatentState`、`TransitionOutput`、`TrainStage`、model-state checkpoint slots 和
`PhysicsConstraint` 都是受测试保护的接口，不代表对应计算已经存在。

## 18. V0.1 → V0.2 接口连接图

以下只是接入位置说明，不是已经存在的代码：

```text
V0.2 dataset loader                         [future]
│
├─ read trajectory states/actions/dts
│  └─ validate_trajectory_alignment()       [V0.1 existing]
│
├─ trajectory-level split                   [future]
│  ├─ fit normalizer on train only          [future]
│  └─ create split_manifest/data_fingerprint
│
├─ window sampler                           [future]
│  ├─ context/future raw states
│  ├─ transform raw -> model states
│  ├─ aligned history/future actions + dts
│  └─ construct ProblemBatch                [V0.1 existing]
│
├─ concrete PhysicsConstraint               [future implementation]
│  └─ conforms to PhysicsConstraint Protocol [V0.1 existing]
│
└─ checkpoint/run integration
   ├─ ProblemSpec                           [V0.1 existing]
   ├─ normalizer_state                      [V0.1 slot]
   ├─ data_fingerprint                      [V0.1 slot]
   ├─ split_manifest                        [V0.1 slot]
   ├─ ProjectConfig/config_hash             [V0.1 existing]
   └─ save_checkpoint/create_run_directory  [V0.1 existing]
```

V0.2 的 dataset/window code 应输出现有 `ProblemBatch`，而不是发明第二种 batch。它应使用
现有 `ProblemSpec` 描述 channel/unit/grid/boundary，并把拟合 normalizer、fingerprint 和 split
结果填入已有 checkpoint slots。seed、config hash、run logging、revision guard 和 TrainStage
无需因数据层接入而重写。

## 19. 当前实现边界与阅读注意事项

以下是代码事实，不在本次 walkthrough 中重构：

1. `ProblemSpec` 与 `ProblemBatch` 完全分离，因此 batch 不会自动核对 `action_dim`、
   `parameter_dim`、channel count、grid requirement 或 mask requirement。
2. smoke YAML 没有 channels/grid/boundary；这些信息在脚本构造的 `ProblemSpec` 中。
3. `TrainingConfig.run_root` 当前不控制 smoke 输出位置；smoke 显式使用临时目录。
4. `coordinates`、`cell_weights`、`valid_mask` 当前只做 Tensor 类型检查，没有 shape 检查。
5. `PhysicsConstraint` 的 raw-unit 语义由接口文档约定，Protocol 的 runtime check 无法验证单位。
6. `Checkpoint` 的 `amp_scaler_state` 会被新 payload 写出，但 loader 为兼容旧 payload 允许缺失。
7. `get_git_commit()` 在非 Git workspace 或命令失败时返回 `None`，不会中断 run。

这些边界都被本文明确标出；它们不应被误读为已经存在的 V0.2 数据或物理实现。
