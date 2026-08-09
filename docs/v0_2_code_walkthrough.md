# V0.2 Code Walkthrough — Data Windows & Physics Contracts

> 历史导读说明：仓库运行时现为 V0.4/`0.4.0`、checkpoint schema 4；本文保留 V0.2 数据与
> 物理契约的实现语义。当前整体入口见 `docs/v0_3_code_walkthrough.md`。

本文是 V0.2 的代码结构阅读手册。唯一有效架构修订为 `2.2`。V0.2 的目标不是获得预测
能力，而是确保以后模型接收的数据在 trajectory、时间、单位、split 和物理意义上都正确。

## 1. V0.2 相比 V0.1 增加了什么

| 模块 | V0.1 | V0.2 |
|---|---|---|
| 工程定位 | engineering contracts | real data pipeline + executable physics contracts |
| trajectory | 只有 `[T+1]/[T]` 对齐校验函数 | `TrajectoryRecord`、`TrajectoryDataset` 与 spec 交叉校验 |
| split | checkpoint 预留 slot | deterministic trajectory-level `SplitManifest` |
| fingerprint | checkpoint 预留 slot | 对 spec、ID、metadata、tensor shape/dtype/content 敏感的 SHA-256 |
| normalization | 只有语义声明和 checkpoint slot | train-only `ChannelStandardizer` 与 state round-trip |
| windows | 只有 `ProblemBatch` contract | 不跨 trajectory 的 `TrajectoryWindowDataset` |
| physics | `PhysicsConstraint` Protocol | 五个 concrete constraints、operators 与 registry |
| probe | optional 概念 | raw-state channel mean/RMS，可完全关闭 |
| toy system | 无 | 解析 1D periodic advection-diffusion，多 trajectory、variable `dt` |
| checkpoint | schema 1 | schema 2，保存 constraint specification 等 V0.2 metadata |
| 模型 | 无 | 仍然无 |

V0.2 没有 encoder、Koopman operator、JEPA、closure 或 trainer。`model state` 只是准备好的
normalized data，不表示已经存在机器学习模型。

## 2. 最终文件树与职责

```text
configs/
└── v0_2_smoke.yaml
scripts/
├── generate_toy_advection_diffusion.py
├── smoke_v0_2.py
└── explain_v0_2.py
src/jka_model/
├── config/schema.py
├── contracts/
│   ├── spec.py
│   └── batch.py
├── data/
│   ├── datasets.py
│   ├── fingerprint.py
│   ├── normalization.py
│   ├── splits.py
│   ├── toy_advection_diffusion.py
│   └── windows.py
├── physics/
│   ├── constraints.py
│   ├── operators.py
│   ├── probes.py
│   └── registry.py
└── utils/checkpoint.py
tests/
├── test_trajectory_data_contract.py
├── test_splits_and_fingerprint.py
├── test_normalization_pipeline.py
├── test_window_dataset.py
├── test_toy_physics.py
└── test_concrete_physics_constraints.py
```

| 文件 | 负责什么 | 谁调用它 | 它调用谁 |
|---|---|---|---|
| `config/schema.py` | strict V0.2 config | 三个 scripts、tests | `ProblemSpec.DtMode`、`TrainStage` |
| `contracts/spec.py` | 静态 channel/unit/grid/boundary 语义 | generator、validator、physics | 只做 dataclass validation |
| `contracts/batch.py` | canonical batch 与时间 shape | window dataset、physics、future model | PyTorch tensor validation |
| `data/datasets.py` | 完整 trajectory 数据结构 | generator、split、normalizer、windows | `validate_trajectory_alignment()` |
| `data/splits.py` | trajectory-level split/serialization | smoke、explain、tests | `SplitConfig`、local seeded RNG |
| `data/fingerprint.py` | 数据身份 SHA-256 | generator script、smoke | `ProblemSpec.to_dict()`、tensor bytes |
| `data/normalization.py` | train-only channel statistics | windows、smoke、checkpoint | `SplitManifest`、`ProblemSpec` |
| `data/windows.py` | trajectory 内切窗并构造 batch | DataLoader、smoke、explain | `ProblemBatch`、normalizer |
| `data/toy_advection_diffusion.py` | 解析 Fourier trajectories | scripts、physics tests | `TrajectoryRecord`、`ProblemSpec` |
| `physics/operators.py` | 积分与周期有限差分 | concrete constraints、tests | PyTorch tensor operations |
| `physics/constraints.py` | concrete physical diagnostics | smoke、explain、tests | operators、canonical raw batch fields |
| `physics/registry.py` | name → constraint factory | smoke/checkpoint configuration | concrete constraint classes |
| `physics/probes.py` | optional raw-state diagnostics | smoke、tests | `ProblemBatch.future_states_raw` |
| `utils/checkpoint.py` | schema-guarded metadata round-trip | V0.1/V0.2 smoke、tests | config/spec/normalizer state payload |

`fingerprint.py` 与 `registry.py` 是对建议目录的单一职责扩展，没有建立第二套 data 或 physics
体系。

## 3. 完整数据流图

```text
Analytic toy trajectories [T+1 states, T transitions]
                         │
                         ▼
             TrajectoryDataset / TrajectoryRecord
                         │
                         ▼
             make_split_manifest(records, config)
                         │
             ┌───────────┼────────────┐
             ▼           ▼            ▼
           train     validation       test
             │           │            │
             └─────┐     │     ┌──────┘
                   ▼     │     ▼
     ChannelStandardizer.fit(TRAIN IDs ONLY)
                   │
                   ▼ frozen mean/scale
        transform train / validation / test
                   │
                   ▼
         TrajectoryWindowDataset per split
                   │
                   ▼
              ProblemBatch
             ┌─────┴──────────────────┐
             ▼                        ▼
    *_states_raw                *_states_model
             │                        │
             ▼                        ▼
 PhysicsConstraint/Probe       future neural model
                                  (not in V0.2)
```

真实代码顺序是 split → fit → window。不存在“先生成所有 windows 再随机划分”的入口。

## 4. 一条真实数据从头跟到尾

以下是 `configs/v0_2_smoke.yaml`、seed `7` 的实际教学运行结果；可用
`python scripts/explain_v0_2.py` 重现。

### 4.1 原始 trajectory

```text
generated trajectories: 12
example full shape:      [17, 1, 65] = [T+1,C,Nx]
dts shape:               [16] = [T]
default dtype:            torch.float32
```

每条 record 有唯一 ID。deterministic split 后，第一个 train ID 是 `trajectory-0007`，它不会
再出现在 validation/test。

### 4.2 normalization stats 从哪里来

```text
train/validation/test IDs: 8 / 2 / 2
normalizer fit IDs:         仅上述 8 个 train IDs
channel mean:               0.955284
channel scale:              0.199858
```

validation/test 只调用 `transform()`，不会调用 `fit()`。

### 4.3 选择第一个 window

配置 `H=4, K=2`，第一条 train window 结束于 `U3`：

```text
trajectory_id: trajectory-0007
context:       U0 U1 U2 U3
future:        U4 U5
history dts:   dt0 dt1 dt2
future dts:    dt3 dt4
actual dt:     [0.0376075, 0.0327479]
```

这里 `dt3` 必须且只允许描述 `U3 -> U4`。

### 4.4 `ProblemBatch` 与 physics 输入

```text
context_states_raw:   [1,4,1,65]
future_states_raw:    [1,2,1,65]
context_states_model: [1,4,1,65]
history_dts:          [1,3]
future_dts:           [1,2]
actions:              None / None
```

同一位置的 representative 数值：raw `0.819026`，normalized `-0.681772`。physics 收到 raw
`U3`、raw `U4`、`dt3`、`mu_static=[c,nu]`、grid spacing、weights 和 mask；它不会收到
normalized value。该 window 的周期边界误差为 `0`，质量 penalty 约 `2.27e-13`，低阶 PDE
residual 约 `1.95e-4`。

## 5. Data Dictionary

### 5.1 Trajectory 数据

| Field | Shape | Unit | Meaning |
|---|---:|---|---|
| `trajectory_id` | string | — | 完整 trajectory 的 provenance |
| `states_raw` | `[T+1,C,*spatial]` | `ChannelSpec.unit` | `U0...UT` |
| `actions` | `[T,d_a]` 或 `None` | problem-specific | `a_i` 驱动 `U_i -> U_{i+1}` |
| `dts` | `[T]` | time | `dt_i` 属于第 `i` 个 transition |
| `mu_static` | `[d_mu]` 或 `None` | parameter-specific | 整条 trajectory 不变参数 |
| `coordinates` | grid-specific | spatial unit | 网格/mesh coordinates |
| `cell_weights` | grid-specific | length/area/volume | quadrature weights |
| `valid_mask` | grid-specific | dimensionless | 有效物理域 |
| `metadata` | mapping | — | 可序列化数据 provenance |

解析 toy record 的 metadata 还保存 `offset_b`，使测试可直接验证质量真值 `M=bL`，而不只是
验证一个可能错误的积分值随时间保持不变。

### 5.2 `ProblemBatch`

| Field | Shape | Source | Used by |
|---|---:|---|---|
| `context_states_raw` | `[B,H,C,*spatial]` | raw trajectory slice | physics/probe/raw metric |
| `future_states_raw` | `[B,K,C,*spatial]` | raw trajectory slice | physics target/raw metric |
| `context_states_model` | 同 context raw | normalizer transform | future model input |
| `future_states_model` | 同 future raw | normalizer transform | future model target |
| `history_actions` | `[B,H-1,d_a]` 或 `None` | transition slice | future controlled model |
| `future_actions` | `[B,K,d_a]` 或 `None` | transition slice | future rollout |
| `history_dts` | `[B,H-1]` | transition slice | context evolution |
| `future_dts` | `[B,K]` | transition slice | future evolution |
| `mu_static` | `[B,d_mu]` 或 `None` | record | model/physics parameters |
| `coordinates` | batched grid shape | record | spatial semantics |
| `cell_weights` | batched grid shape | record | physical integrals |
| `valid_mask` | batched grid shape | record | domain selection |
| `trajectory_id` | sequence length `B` | record | provenance/leakage audit |

只有 canonical v2.2 names。旧 `parameters`、`mask` 和 aggregate state/action/dt aliases 已移除。

### 5.3 Normalizer state

| State | Meaning |
|---|---|
| `kind` | 当前为 `channel_standardizer` |
| `eps` | 防止 channel scale 为零 |
| `mean` | 每个 channel 的 train-only mean |
| `scale` | 每个 channel 的 train-only standard deviation 加 `eps` |
| `spatial_dim` | 用于恢复 channel axis |
| `layout` | `channels_first` 或 `channels_last` |
| `fitted_trajectory_ids` | 审计 fit 数据来源 |

### 5.4 `SplitManifest`

| Field | Meaning |
|---|---|
| `train` | 完整 train trajectory IDs |
| `validation` | 完整 validation trajectory IDs |
| `test` | 完整 test trajectory IDs |
| `seed` | local deterministic shuffle seed |
| `ratios` | train/validation/test requested ratios |

### 5.5 `ConstraintResult`

| Field | Meaning |
|---|---|
| `name` | 稳定 diagnostic 名称 |
| `penalty` | scalar differentiable tensor；V0.2 只报告，不启动训练 |
| `diagnostics` | detached scalar details，如 max mismatch 或 residual RMS |

Concrete `.evaluate()` 统一返回该类型；兼容 V0.1 Protocol 的 `.loss()` 返回 named scalar mapping。

### 5.6 V0.2 相关 `ProblemSpec`

| Field | Meaning |
|---|---|
| `channels` | channel name 与真实 unit |
| `spatial_dim` | spatial axes 数目 |
| `grid` | layout、shape、spacing 与 geometry requirements |
| `boundary` | toy 为 periodic |
| `action_dim` | action dimension；toy 为 0 |
| `parameter_dim` | static parameter dimension；toy 为 2 |
| `dt_mode` / `constant_dt` | constant 或 variable transition time |
| `normalization` | preprocessing 语义声明，不保存 fitted stats |
| `geometry` | mask requirement |
| `observable_requirements` | 可选 probe/metric names |
| `metadata` | equation 与 parameter names |

## 6. 时间对齐图

最基本的 transition：

```text
U0 ----a0,dt0----> U1 ----a1,dt1----> U2 ----a2,dt2----> U3
```

选 `H=3, K=1, t=2`：

```text
context states  = U0,U1,U2
history actions = a0,a1
history dts     = dt0,dt1
future action   = a2
future dt       = dt2
future state    = U3
```

因此 `a2,dt2` 的唯一含义是 `U2 -> U3`，绝不是 `U3` 之后的 action。

测试还使用附件要求的 `H=3,K=2,t=2` fixture：

```text
states          = U0 U1 U2 U3 U4 U5 ...
context         = U0 U1 U2
future          = U3 U4
history dts     = dt0 dt1
future dts      = dt2 dt3
history actions = a0 a1
future actions  = a2 a3
```

`TrajectoryWindowDataset` 为每条 record 单独建立 `(record_index,t)` reference，所以跨
trajectory window 在数据结构上不可能生成。

## 7. 为什么必须先 split 再 window

错误做法：

```text
Trajectory A
 ├── A[U0..U4]  ──> train
 ├── A[U1..U5]  ──> train
 └── A[U2..U6]  ──> test
```

test window 几乎完整见过 train 的相邻状态，会严重高估泛化能力。

正确做法：

```text
Trajectory A ──> train ──> windows of A only
Trajectory B ──> train ──> windows of B only
Trajectory C ──> test  ──> windows of C only
```

实现先对排序后的 IDs 进行 local seeded shuffle，再用 largest-remainder 分配数量。同一 ID
不能出现在两个 manifest groups；`select_split()` 也会拒绝 manifest 中不存在的 ID。

## 8. Normalizer 生命周期

```text
train raw trajectories
          │
          ▼
 fit channel mean/scale
          │
          ▼
    freeze statistics
     ┌────┼────────────┐
     ▼    ▼            ▼
  train  validation   test
 transform transform transform
```

不能 `fit(all data)`：validation/test distribution 本身就是未知信息，纳入 mean/scale 会泄漏。
测试专门给 validation 放入 `1e6` 极端值，确认 train mean 仍为 `2.0`。

Normalizer 计算：

\[
\mu_c=\frac{1}{N_c}\sum_jU_{j,c},\qquad
\sigma_c=\sqrt{\frac{1}{N_c}\sum_j(U_{j,c}-\mu_c)^2}+\epsilon,
\]

\[
U^{model}_c=\frac{U^{raw}_c-\mu_c}{\sigma_c},\qquad
U^{raw}_c=U^{model}_c\sigma_c+\mu_c.
\]

统计用 CPU float64 累加，transform 输出保持输入 device/dtype；正常 pipeline 默认 float32。
操作不是 inplace，所以 `states_raw` 始终保留，供物理约束和最终指标使用。

## 9. Toy PDE 数学说明

方程为

\[
u_t+c u_x=\nu u_{xx},\qquad x\in[0,L],\qquad u(0,t)=u(L,t).
\]

- `u_t`：局部时间变化；
- `c u_x`：速度 `c` 造成的平移/输运；
- `nu u_xx`：diffusivity `nu >= 0` 造成的平滑与衰减；
- `c` 越大，波形传播越快；
- `nu` 越大，高频 mode 衰减越快。

generator 使用

\[
u(x,t)=b+\sum_{m=1}^{M}A_m e^{-\nu k_m^2t}
\sin(k_m(x-ct)+\phi_m),\qquad k_m=\frac{2\pi m}{L}.
\]

对单个 mode 求导：时间导数产生 `-c*k*cos - nu*k^2*sin`，空间一阶导数产生
`k*cos`，二阶导数产生 `-k^2*sin`；代入后 `u_t+c*u_x-nu*u_xx=0`。整数 Fourier mode
也天然周期，且正弦在完整周期上的积分为零，因此质量

\[
M(t)=\int_0^L u(x,t)dx=bL
\]

保持不变。

网格保存 `x=0` 与 `x=L` 两个 endpoint，并显式令末点等于首点；trapezoidal weights 给两端
各半权重。variable `dt` 时先累计 `t_{i+1}=t_i+dt_i`，再直接在这些真实时间求解析解，
不做 numerical time integration。

这个系统同时具有时间、空间、PDE、boundary、conservation 和解析 reference，又无需复杂
solver，适合作为 V0.2 contract test。

## 10. `PhysicsConstraint` 调用结构

```text
ProblemBatch.context_states_raw[:,-1] ──┐
ProblemBatch.future_states_raw[:,0]  ───┼──> evaluate_constraints()
ProblemBatch.future_dts[:,0]         ───┤
mu_static / weights / mask / spec    ───┘
                                           │
                 ┌─────────────────────────┼──────────────────────┐
                 ▼                         ▼                      ▼
          finite/admissible          periodic BC          conservation/PDE
                 │                         │                      │
                 └──────────── named scalar penalties ───────────┘
```

| Constraint | Inputs | Output meaning |
|---|---|---|
| `FiniteValueConstraint` | predicted raw state | non-finite fraction，NaN/Inf 会被发现 |
| `StateAdmissibilityConstraint` | raw state、generic bounds | squared hinge range penalty |
| `PeriodicBoundaryConstraint` | raw state 首/末 endpoint | boundary mismatch MSE |
| `MassConservationConstraint` | previous/predicted raw state、weights、mask | integral change MSE |
| `DiscretePDEResidualConstraint` | previous/predicted raw、`dt`、`c,nu`、spacing | FTCS residual MSE |

离散 residual 为

\[
r=\frac{u^{n+1}-u^n}{\Delta t}+cD_xu^n-\nu D_{xx}u^n.
\]

空间算子在去除重复 endpoint 的 unique periodic grid 上用 centered difference，再复制第一点
结果到末端。若 `u` 的单位为 `[U]`、`x` 为 length、`t` 为 time，则 `c` 为 length/time、
`nu` 为 length²/time，residual 三项单位均为 `[U]/time`。它是 float64 convergence test 支持的
低阶 diagnostic，不是 solver 或训练 loss。

`create_constraint()` 通过显式 registry 构造实例；未知名称和 duplicate registration 都报错。
`ConstraintResult` 为 `.evaluate()` 提供统一可读结果。

Physical probes 通过 `evaluate_batch_probes()` 只路由 `future_states_raw`。当前只有 channel
mean/RMS，无参数、不属于 latent；传空 list 返回 `{}`，完整 smoke 仍成功。

## 11. Raw state 与 model state

```text
raw state                         model state
真实物理单位                      normalized dimensionless value
PhysicsConstraint / Probe         future neural model
永久保留                          可由 frozen normalizer 重建 raw
```

例如速度 `u_raw=100 m/s`，若 train mean 为 `85 m/s`、scale 为 `20 m/s`：

\[
u_{model}=(100-85)/20=0.75.
\]

`0.75` 没有 `m/s` 单位，不能直接代入真实动能、质量或 PDE。代码中的防火墙是
`evaluate_constraints()` 与 `evaluate_batch_probes()`：两者从 canonical raw fields 取值。
测试把 model state 改成 `±1e9` 或再加 `1e6`，physics/probe 结果保持不变。

## 12. `smoke_v0_2.py` 调用链

从 IDE 可按以下顺序打开：

```text
scripts/smoke_v0_2.py::main
  ↓ load_config()
ProjectConfig / DataConfig
  ↓ generate_advection_diffusion_trajectories()
TrajectoryDataset + ProblemSpec
  ↓ validate_trajectories_against_spec()
validated records
  ↓ data_fingerprint()
sha256 identity
  ↓ make_split_manifest()
SplitManifest(train/validation/test IDs)
  ↓ ChannelStandardizer.fit(records, manifest, spec)
frozen train-only mean/scale
  ↓ select_split() × 3
three disjoint record sequences
  ↓ TrajectoryWindowDataset() × 3
train/validation/test windows, all transformed by the same stats
  ↓ DataLoader(..., collate_problem_batches)
ProblemBatch
  ↓ evaluate_constraints() / evaluate_batch_probes()
named physics terms / optional diagnostics
  ↓ SplitManifest.save() / save_checkpoint()
serialized split + spec + fingerprint + normalizer + constraint metadata
  ↓ SplitManifest.load() / load_checkpoint() / normalizer.load_state_dict()
verified metadata and normalization round-trip
```

当前输出明确报告：12 trajectories、8/2/2 split、96/24/24 windows、第一 future transition
为 `U_t -> U_{t+1}`、normalization scope 为 train-only、probe-disabled path 成功。

另外：

- `generate_toy_advection_diffusion.py` 可只打印 identity，或用 `--output` 保存 plain tensors；
- `explain_v0_2.py` 是独立的 10 STEP 教学流程，不是 smoke 改名。

## 13. Tests 查阅表

| Test/行为 | 防止的 bug | 为什么危险 |
|---|---|---|
| trajectory `[T+1]/[T]` validation | state/dt/action 长度错位 | 未来 transition 学错目标 |
| window exact fixture | off-by-one context/future slices | `dt_t` 错配到别的 state |
| action fixture | action index 漂移 | controlled dynamics 因果关系错误 |
| variable `dt` fixture | 把 `dt` 当常数或 state metadata | V0.3 `exp(A*dt)` 数学错误 |
| no-cross-trajectory | A 的末尾拼 B 的开头 | 制造不存在的物理 transition |
| split before windows | random window leakage | test 几乎见过 train 邻域 |
| same seed/order independence | split 不可复现 | checkpoint 无法重建实验 |
| train-only extreme validation | normalization 偷看 validation | 指标乐观偏差 |
| raw clone unchanged | inplace normalization | 真实单位永久丢失 |
| normalizer/state round-trip | reload 数值漂移 | resume 输入改变 |
| canonical batch shapes | batch axis/time axis 混淆 | 下游 silent broadcasting |
| raw-state physics/probe firewall | normalized field 进入物理公式 | 单位错误但数值可能看似合理 |
| NaN/Inf detection | 非有限状态未报警 | residual/metrics 污染 |
| broken periodic endpoint | boundary check 永远返回零 | BC contract 失效 |
| mass conservation | 用 mean 代替 quadrature | 非均匀网格积分错误 |
| PDE resolution improvement | residual 实现符号/差分错误 | 解析解反而不收敛 |
| probes disabled | optional 模块变成硬依赖 | 不需要 probe 的任务无法运行 |
| fingerprint order/content/metadata | identity 不稳定或不敏感 | 使用错误数据却复用 checkpoint |
| strict V0.2 config | invalid H/K/ratio/nu/Nx 被接受 | 错误推迟到深层 pipeline |
| V0.1 regression suite/smoke | V0.2 破坏基础契约 | 后续版本失去稳定基线 |

附加数值验证使用 float64 reference：

- `test_first_derivative_against_analytic` 验证 `D1 sin(kx) ≈ k cos(kx)`；
- `test_second_derivative_against_analytic` 验证 `D2 sin(kx) ≈ -k² sin(kx)`；
- `test_second_order_grid_convergence` 使用 `Nx=32,64,128`。当前实际 RMS errors 为
  `3.8414e-2 > 9.3592e-3 > 2.3066e-3`，observed orders 为 `2.0372` 与 `2.0206`，符合二阶
  centered difference；测试会打印这些 diagnostics；
- `test_mass_matches_analytic_b_times_length` 同时验证每个时刻 `M(t)=M(0)=bL`；
- negative/zero/wrong-length transition tests 确认无 truncate、padding 或 silent correction；
- `test_zero_variance_channel_normalization` 确认 `sigma+epsilon` 产生有限零值并可逆。

运行全部测试：

```bash
pytest -q
```

## 14. Recommended Reading Order

1. `scripts/smoke_v0_2.py`：先看完整调用链；
2. `data/toy_advection_diffusion.py`：理解数据的数学来源；
3. `data/datasets.py`：固定 `[T+1]/[T]` 语义；
4. `data/splits.py`：确认 split 单位是 trajectory；
5. `data/normalization.py`：确认 fit 只读取 train IDs；
6. `data/windows.py`：逐个检查 slice；
7. `contracts/batch.py`：核对 canonical batch fields；
8. `physics/constraints.py`：检查 raw routing；
9. `physics/operators.py`：理解 quadrature 与 periodic finite differences；
10. `tests/`：看每条 contract 如何被故意破坏并检测。

若第一次接触时间窗口，先运行 `python scripts/explain_v0_2.py`，再读第 6 节与
`test_window_dataset.py`。

## 15. Top Symbols

| Symbol | 文件 | 输入 | 输出 | 一句话作用 |
|---|---|---|---|---|
| `TrajectoryRecord` | `data/datasets.py` | full raw tensors/metadata | validated record | 固定唯一 trajectory 时间语义 |
| `TrajectoryDataset` | `data/datasets.py` | records | immutable sequence | 保证 ID 唯一 |
| `validate_trajectories_against_spec` | `data/datasets.py` | records、spec | `None`/exception | 动态 shape 与静态物理语义交叉校验 |
| `SplitManifest` | `data/splits.py` | ID groups/seed/ratios | serializable dataclass | 保存可重建 split |
| `make_split_manifest` | `data/splits.py` | records/IDs、config | manifest | deterministic trajectory split |
| `data_fingerprint` | `data/fingerprint.py` | records、spec | `sha256:...` | 标识实际数据内容与 metadata |
| `ChannelStandardizer` | `data/normalization.py` | train records、manifest、spec | frozen stats/transforms | 防泄漏 channel normalization |
| `TrajectoryWindowDataset` | `data/windows.py` | one split records、H/K、normalizer | B=1 `ProblemBatch` items | trajectory 内安全切窗 |
| `collate_problem_batches` | `data/windows.py` | B=1 items | batched `ProblemBatch` | 只拼 canonical fields |
| `generate_advection_diffusion_trajectories` | `data/toy_advection_diffusion.py` | toy config、seed、dtype | dataset、spec | 直接求解析 Fourier trajectories |
| `ProblemBatch` | `contracts/batch.py` | aligned window fields | validated batch | raw/model/action/dt 公共接口 |
| `ConstraintResult` | `physics/constraints.py` | name/penalty/diagnostics | result dataclass | 统一 concrete evaluation 输出 |
| `evaluate_constraints` | `physics/constraints.py` | constraints、batch、spec | named scalar mapping | raw physics routing 防火墙 |
| `weighted_integral` | `physics/operators.py` | state、weights、mask | integrated tensor | 正确计算物理积分 |
| `create_constraint` | `physics/registry.py` | name/specification | constraint instance | 显式、无 fallback 的 registry lookup |
| `evaluate_batch_probes` | `physics/probes.py` | probes、batch、spec | diagnostics mapping | 只把 raw batch state 给 probe |

## 16. V0.2 仍然没有什么

```text
No z_k yet
No z_r yet
No Koopman matrix/operator
No KoopmanEncoder
No JEPA or EMA target encoder
No residual target/dataset/learning
No Attention / Transformer / GRU closure
No TrainingDecoder
No neural operator
No trainer or model optimization
No learned rollout, MPC, or RL
```

`src/jka_model/models/__init__.py` 仍是保留边界，没有新增模型文件。V0.2 的 PDE residual 只是
解析数据 diagnostic，不属于模型 loss 或训练循环。

## 17. V0.3 将从哪里接入

V0.2 只提供接口，不实现 V0.3：

```text
ProblemBatch.context_states_model ──┐
ProblemBatch.history/future_dts  ───┼──> V0.3 direct-state KoopmanCore
ProblemBatch actions / mu_static ───┘       (future implementation)
                                               │
                                               ▼
                                      predicted model state
                                               │
                          ChannelStandardizer.inverse_transform
                                               │
                                               ▼
                                    raw PhysicsConstraint/metrics
```

按照架构 v2.2，V0.3 首先在 **没有 learned encoder** 的情况下，用 damped harmonic
oscillator 与 nonlinear oscillator 验证 continuous-time `matrix_exp(dt*A)`、spectrum 和 rollout
数学；不会马上把 PDE field 喂给 encoder。V0.2 到此停止，没有 `step()`、`rollout()`、
`spectrum()` 或任何 Koopman 实现。
