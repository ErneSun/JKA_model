# V0.9 新路线 Stage 1：同样本瓶颈审计

日期：2026-09-09。状态：实现完成，本地针对性软件测试通过；真实 checkpoint GPU 审计待运行。

此处 **Stage 1** 是 Phase 3.7 之后的新路线第一阶段，不是重新执行 2026-08 的 Added Phase 1。
既有训练、Phase 2 负结果、matched frozen/from-scratch 对照以及科学门槛均保持不变。

## 1. 为什么先做审计

来源会话：`v09-added-p37-aligned-20260903T015511Z`，18 个 joint 单元完成，但没有达到原定
matched decoded-field 改善门槛。潜空间改善没有稳定转化为物理场改善，单靠 compact results
不能断言究竟是表示/解码、传播、坐标变化还是输入权限导致。

本阶段不再叠加 loss，不训练新模型、不调整门槛，也不生成新的物理数据。既有受控圆柱尾流
包含平滑、突变、上升、下降和循环工况，足以做**当前模型的误差拆分及相关性诊断**。
它不自动构成普适动力学或输入可辨识性的证明。新加的确定性小算例仅用于检验数学/软件契约。

下一阶段才依据证据决定：优先隔离表示/decoder 修复，还是比较冻结表示下的固定算子重拟合
与 input-conditioned 算子。动态历史机制是否值得保留由后续 matched 物理收益决定。

## 2. 精确数学契约

对于相同轨迹、起点 t、预测步长 h，记真值为 x，模型预测为 p=D(z_hat)，参考解码为 r=D(z_star)：

\[
e=p-x=(p-r)+(r-x)=e_{prop}+e_{decode},
\]

\[
\operatorname{MSE}(e)=\operatorname{MSE}(e_{prop})+
\operatorname{MSE}(e_{decode})+2\langle e_{prop},e_{decode}\rangle/n.
\]

- 三个张量必须在同一单位、同一 gauge、同一 mask、同一批样本上计算。
- 包括**有符号交叉项**；不把 RMSE 相加，不把 reconstruction error 称为不可突破的误差下界。
- 分别采用 `z_star=E_online(x)` 与 `z_star=E_EMA(x)`；两者使用同一当前 decoder。
- online 项是当前 autoencoder 的重构误差；teacher 项是冻结 EMA 坐标的 cross-decoding 误差。
  EMA encoder 并未必与当前 decoder 构成最优自编码对，不能混淆两者。
- 同时计算同一起始 online latent 下 nominal A0 的误差拆分；它**不是**旧科学比较里的
  matched frozen-adaptive checkpoint，不据此重算原有通过结论。
- 冻结继承的 `D_ref(E_ref(x))` 作为重构对照，识别 joint 表示是否改善了 decoder 输出。
- float32 推理、float64 诊断累积；归一化恒等式闭合误差须小于 `1e-10`。

默认评估步长为 `1 ∪ training.rollout_horizons ∪ training.active_observable_horizons`。
当前来源通常为 H1/4/8/16/32/80。所有步长使用同样、足够覆盖最大步长的起点，
不把短步长更多起点的平均值与长步长更少起点混比。

## 3. 物理通道与统计

| 诊断 | 作用与边界 |
|---|---|
| field_all | 原始无量纲 [u,v,p]、包含全网格；保留原 field 指标口径 |
| field_fluid / velocity_fluid / u_fluid / v_fluid | 流体掩码内的场/速度/分量；防止总体指标掩盖通道退化 |
| pressure_fluid | 原始压力通道，不调整压力零点 |
| pressure_demeaned_fluid | 对每个场分别减去流体区域平均压力，**单独列出**的 gauge 敏感性诊断 |
| vorticity_interior | 中心差分；仅使用中心及四邻点都在流体内且不接触外边界的 stencil |

所有 raw 通道保持数据原有无量纲定义，不额外用目标统计量重标度。非均匀网格尚不支持；
当前矩形均匀网格的常数单元面积在 mean 中抵消。没有新增外力、噪声、经验物理校正或 PDE。

对每个 seed / mode / initialization / horizon / channel / reference 分别报告：
先在轨迹内平均窗口，再对轨迹等权平均；不把嵌套的 18 个单元伪装成独立物理实验。
JSONL 保留每个窗口的原始分量，允许改变描述性汇总而无需重放。零目标能量时 relative L2
写 null，绝不靠随意分母把它解释成正常百分比；绝对 MSE 仍可用。

原报告复现检查单独使用原来的**等窗口** decoded field relative L2。
所有已声明 observable horizon 的来源指标必须存在且有限；偏差超过
`2e-5 + 2e-4*abs(source)` 则中止，记录 `FAILED_INCOMPLETE`，不得在错误重放上作判断。
该容差只用于不同 batch/kernel 下 float32 重放，不改变任何科学门。

## 4. 表示谱与动态 gauge

每条轨迹的每个原始状态仅编码一次，计算 online、frozen teacher、inherited online 的奇异谱、
协方差熵 effective rank、99% 能量秩及数值条件数。常量表示的 effective rank 为 0，不伪报 1。

仅用 **TRAIN** 拟合标准化、平移及 row-vector Procrustes 映射 Q；validation/test 不参与拟合：

\[
Q=\arg\min_{Q^TQ=I}\|X_{train,n}Q-Y_{train,n}\|_F^2.
\]

采用 column-state generator A，而 row-state 动力学为 `Xdot=X A^T`，因此旋转部分的动态缺陷为

\[
C=A^TQ-QA^T.
\]

同时报告全空间 `||C||/||A||`、实际数据上的 `||XC||/||XA^T||`，以及 TRAIN 99% 能量子空间上的
action 指标。高方差方向由训练集选定；不能根据 test 选择秩。还报告完整仿射拟合
`Y≈αXQ+b` 对应的动态缺陷 `αXC-bA^T`，避免旋转看似无害却漏掉均值平移对冻结 A0 的影响。

这些是**新诊断**，不是替换旧 commutator 或 drift gate；低数据作用误差不等于全空间同构，
低有效秩也不能单独证明应减少潜维数。拟合矩阵、均值、尺度、子空间和 split 清单保存在 runs。

## 5. 输入时序和 observer 权限

原始数据约定为 `u[t]` 推进 `x[t] -> x[t+1]`。起点 `x[t]` 在该步新输入生效之前。
记录起点输入跳变、后续窗口内输入变化次数、真实 input/target index，并按 changing/unchanged
子集描述误差。变化检测只用相对量级 `1e-6` 的数值容差，不是训练出的识别门。

- known：重放中每步获得原先允许的 future transition schedule（及原 condition target 转换）。
- latent_inferred：只有已观测历史/静态参数和已知 dt；未来真值编码仅用于诊断，不进入 rollout。
- 使用 checkpoint 中**初始 observer admission** 决定分支，不使用后来 locked-test 的 observer 分数。
  原来未获准使用 observer 的单元继续 dynamic-only / q_used=0 回退，不能恢复 static 分支。
- 变化子集误差更高只是相关性证据，不能证明输入不可辨识。若后续需严格检验这一点，应构造
  **完全相同历史、不同未来输入** 的受控配对算例，再明确预测信息权限；本阶段不伪造此证据。

## 6. 如何形成下一步建议

仅在 validation 上给出预先固定的**调查优先级**，不是科学门槛：

1. 若 `|cross| > 0.5*(propagation+decode)`：`COUPLED_CROSS_TERM`，先研究耦合，不能宣称单独下界。
2. 否则 decode > 2×propagation：`REPRESENTATION_DECODE_PRIORITY`。
3. 否则 propagation > 2×decode：`PROPAGATION_PRIORITY`。
4. 否则 `MIXED`；接近机器零量级时 `UNRESOLVED_NEAR_ZERO`，这不是 R0 分类。

online/teacher、物理通道、种子之间若不一致，保留不一致，不通过多数投票改成“已证明”。
test 只作描述。旧 test 已在历次报告中被查看；不能声称它是未使用的独立验证集。
后续若基于这些结果调路线，最终确认性实验需要新锁定测试集。

## 7. 模块与数据安全

- `src/jka_model/manifold/bottleneck.py`：可独立复用的精确分解、field views、谱/gauge、时序及汇总。
- `src/eval/audit_v0_9_bottleneck.py`：严格 checkpoint/provenance 读取、真实推理重放与报告生成。
- `gpu_validation/v0_9/scripts/gpu_audit_bottleneck.py`：单命令工作流、完整来源矩阵、日志/失败报告。
- `tests/test_v0_9_bottleneck_audit.py`：本阶段收敛测试；不加入所有历史测试。

只读取已保存训练 checkpoint 和原始物理数据，不创建 optimizer，不调用训练函数。
复用 Phase-3 模型构造器后 strict-load 所有模型状态，并冻结全部参数。raw config 先按保存内容
校验 hash，再添加运行时默认值；同时保留原始和展开后的 config，不改 checkpoint。
校验 backbone/context SHA、adaptive cache fingerprint、raw dataset fingerprint、冻结 reference
及 EMA state、normalizer、nominal generator。旧绝对路径只按明确 `runs/` 后缀映射到当前仓库，
不搜索同名文件、不换 checkpoint、不绕过 hash/readiness。

已完成的 18 单元自动从来源 summary 读取，不重新跑 G1 handoff/readiness、rank sweep 或训练。
但会核验**既有 handoff 记录及其依赖文件**的来源；文件缺失时报告需恢复的路径，不自动重训。
审计允许 dirty tree，但保存工作区状态、代码 hash 和 commit，因为这不是 formal scientific pass。

## 8. 服务器一行命令

在仓库根目录、已激活包含 torch/pytest 的原 `.venv` 中，先同步新代码，再执行：

```bash
python gpu_validation/v0_9/scripts/gpu_audit_bottleneck.py --source-id v09-added-p37-aligned-20260903T015511Z
```

默认 CUDA，逐窗口推理、编码 chunk=8，针对 RTX 5080 控制显存；无反向图、无训练。
需要保留来源 Phase-3.7 的 checkpoint、其 Phase-2 controlled dataset/cache/handoff JSON，
以及该 handoff 引用的 V0.8 context 和 backbone checkpoint。本地没有这些大文件并不影响提交代码。

自动生成 UTC id `v09-stage1-audit-YYYYMMDDTHHMMSSZ`；显式 `--validation-id NAME` 也支持。
若 raw 或 compact 同名 id 被占用，自动生成 `-r1/-r2/...`，不覆盖、不删除既有结果。

```text
runs/v0_9/<resolved-id>/
  configs/                  # 请求、commit、代码hash、工作区状态
  logs/                     # pytest.log / audit.log
  cells/seed_*/MODE/init_*/  # 逐窗口 JSONL、样本清单、gauge拟合、config、单元报告
  evaluation/               # 进度和汇总
gpu_validation/v0_9/results/<resolved-id>/
  report.md                 # 自包含审阅说明、validation优先级、长时域数值拆分
  audit.json                # 全部单元/时域/通道的汇总、谱/gauge和重放检查
  completion.json           # 工作流状态，不是科学验收
  failure.json              # 仅失败时，含stage/cell/traceback
  partial_audit.json        # 仅失败时，保留此前完成单元
```

主要阶段 START/PASS/FAIL 可见，日志保留；失败也在 results 生成 Markdown 报告。
`PASS + scientific_acceptance=NOT_EVALUATED` 只表示本阶段审计完成。

## 9. 本地验证与后续交接

新增测试覆盖：分解恒等式/负交叉项、尺度与零目标、压力 gauge/流体 stencil、低秩/常量表示、
TRAIN-only gauge 与平移缺陷、输入 off-by-one、轨迹等权汇总、初始 observer 回退、历史 config hash、
安全路径迁移、真实 FactorizedAdaptiveOperator 的已知/latent 两模式无训练重放、来源指标失配拒绝、
成功/失败报告、revision id、paired 窗口失配拒绝。

另只选择原 Phase-3 中与当前调用有关的窗口、gauge、observer 测试；不重跑训练和历史全集。
本地小算例的已知解析 nominal 解只用于验证审计软件，不能代替 GPU 科学证据。

本次实际结果：新增测试 **14 passed**；相关历史测试 **5 passed, 20 deselected**；
新增 Python 文件 Ruff 检查通过，独立脚本 `--help` 导入通过。未在本地生成假 GPU 结果，
没有训练、删除历史 runs 或修改原始 source results。真实 checkpoint 数值审计仍待服务器运行。

下一位开发者首先审阅新 `results/<id>/report.md` 和 `audit.json`，确认重放及数据配对成立，
再讨论 Stage 2 的最小模型。不要把这里的 Stage 1 当成已实现 Stage 2/3，不要继续叠加 Phase 3.7 loss。
