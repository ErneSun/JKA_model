# V0.9 Phase 3.7 结果审阅与路线重评

日期：2026-09-09。范围：读取已有结果、核对源代码和查阅相关原始文献；未运行测试、训练或 checkpoint 重评，未修改训练代码、配置或已有科学结论。

## 1. 结论与证据边界

**本轮工作流完成，科学验证未通过；建议停止沿 Phase 3.6/3.7 叠加损失与准入门的方式迭代，转为先验证受控 Koopman 的最小模型，再按证据添加历史机制。**

这不等于证明 Koopman 或双算子在数学上错误。结果反映的是当前表示、训练预算、数据、控制信息和门控组合尚不成立。Koopman、JEPA 表示、物理约束与残差诊断仍有保留价值。

本地只有 compact results，没有 `runs/v0_9/` 和 checkpoint。可以确认完成状态、汇总指标、逐 run gates 和 observer controls；不能据此确认完整梯度历程、实际解码误差分解、Jacobian 谱或早停前后曲线。相关源文件与运行记录的 commit `893f6538a80be63516cf06aae2daf870134f69b0` 无差异。

原始证据：

- 本轮：[report.md](report.md)、[completion.json](completion.json)、[joint_summary.json](evaluation/joint_summary.json)。
- 上轮：[Phase 3.6 summary](../v09-added-p3-physical-joint-20260830T085348Z/evaluation/joint_summary.json)。
- 固定比较基准：[matched frozen summary](../v09-added-p3-routes-20260829T025754Z/evaluation/matched_route_summary.json)。
- 原始表示审计：[entry audit](../v09-added-p3-audit-20260826T043840Z/evaluation/phase3_route_decision.json)。其中旧 RMS/MSE 判定已有修订，本报告仅引用其重构数值。

## 2. 本轮结果

18 个唯一 seed/mode/init 组合齐全，locked-test 标量全部有限；训练完成 65–70 epoch，最佳 checkpoint 为 49–54 epoch。没有失败中断记录被当作科学负结果。

| 指标 | Phase 3.6 | Phase 3.7 |
|---|---:|---:|
| 正式训练/评估完成 | 18/18 | 18/18 |
| 已定义物理门 | 18/18 | 18/18 |
| 潜空间多时域预测门 | 18/18 | 17/18 |
| round-trip 门 | 12/18 | 12/18 |
| 原始 representation drift 门 | 0/18 | 0/18 |
| dynamical gauge 综合门 | 未启用 | 0/18 |
| latent observer 门 | 1/9 | 0/9 |
| 多时域 decoded-field 2% 门 | 0/18 | 0/18 |
| matched route 最终门 | 0/18 | 0/18 |

物理门指当前 divergence/no-slip/outer-boundary 阈值；不是所有 Navier–Stokes 方程、压力、升阻力或长期稳定性的全面证明。

以下为 18 个 matched runs 各自相对同一 frozen counterpart 的 field error 改善，再取算术平均；正值表示误差下降：

| horizon | Phase 3.6 | Phase 3.7 |
|---|---:|---:|
| H8 | 0.901% | 0.762% |
| H16 | 0.836% | 0.719% |
| H32 | 0.696% | 0.626% |
| H80 | 0.747% | 0.530% |

不仅没有 run 同时通过四个 horizon，18×4=72 个单独 field-gain 检查也全部低于 2%。去掉新 observer/gauge 门仍无法使本轮达到原定 decoded-field 目标。

本轮并非毫无收益，但平均 field 收益略低于上轮；这属于描述性比较，不能用 18 个嵌套 run 冒充 18 个独立物理样本来宣称统计显著退化。

按信息模式区分：

- known 的 H80 潜预测 gain 均值：8.972% → 8.899%，基本持平。
- latent-inferred 的 H80 潜预测 gain 均值：8.350% → 3.425%。9/9 初始 observer 拒绝，全部运行 history-only fallback。
- 这些 latent gain 相对于各自当前初态的 nominal rollout，不能解释成相对 frozen 模型的同等物理收益。
- seed 47 的 6 个 runs 均未通过速度/涡量全时域非劣门；H80 平均速度 gain 为 -2.091%，涡量为 -2.468%。
- seed 59 的 6 个 runs 均未通过 round-trip，均值约 0.276；seed 47/53 均值约 0.173/0.231。

## 3. 当前瓶颈与设计问题

### 3.1 外部输入预测和系统动力学识别被绑定

源代码 `src/jka_model/data/cylinder_wake_2d.py` 明确以预设 inlet schedule 推进求解器；`src/jka_model/adaptive/objectives.py` 的 known 模式在 rollout 每一步获得对应输入，latent 模式获得 None。

物理任务实际为：

\[
U_{t+1}=F(U_t,u_t),
\]

其中外部输入改变入口边界。给定同一历史，外部操作者可以选择不同的未来输入，产生不同的后续流场。因此一般不存在仅靠过去流场唯一恢复任意未来输入的函数。有限、固定 schedule 上可能学到规律，但不等于一般外部输入可预测。

`future_condition_targets[:,0]` 对应即将作用于下一步的 transition input；在突变点，当前状态尚未经历这个新输入。应区分过去已作用的输入估计、当前输入测量、未来控制计划三个时序合同。

建议将已知输入条件预测作为主任务；未知输入作为独立的观测/估计任务，明确允许的测量、输入演化假设和预测范围。两种信息条件不应被默认要求共同通过，才能承认已知输入模型的局部科学价值。改变这一点必须形成新路线验收合同，旧结果仍保留 NOT_SUPPORTED。

### 3.2 observer 可辨识性与历史必要性被错误合并

观察器需要历史，和观察器能准确恢复工况，是不同命题。如果当前状态已经足以辨识工况，instantaneous 与 history 一样好是合理的 R2 结果，不应因此禁止条件输入进入算子。

当前 `classify_observer_admission()` 要求绝对精度、history 优于 instantaneous、history 优于 shuffled 同时成立。建议拆成三个证据：

1. 是否能恢复目标量；
2. 在相同输入条件和容量下，历史是否改善未来状态/残差预测；
3. 使用该信息是否实际改善 rollout。

不要把“精确恢复 Re、U、dRe/dt”作为所有自适应模型必须完成的辅助任务。当前生成器固定黏度和几何，且 `U/U0 = Re/Re0`，Re/U 是同一控制自由度的两个表达，并非两个独立可辨识参数。通用接口应使用独立控制变量和可观测目标，由 problem adapter 给出映射。

### 3.3 固定倒序 control 不是信息消除实验

`observer_history_variant()` 将 pre-current history 固定倒序，并另行训练 control。固定排列是可逆的；在输入可访问时，它不会从信息论意义上移除历史信息。经冻结 context 压缩后可能造成信息变化，但这个变化混入了编码器对输入排列的敏感性，不能单独证明记忆必要性。

建议采用保留当前状态的随机时间破坏诊断，和按当前状态/已知输入匹配后替换历史的条件 control；训练型等容量无历史对照与推理时破坏实验分开解释，并量化替换后的分布偏移。仅随机打乱也不是自动成为无偏因果实验。

这属于上一轮设计的局限，不能归咎于数据。与此同时，本轮初始 history observer NRMSE 为 0.583–1.055，最差分量 R² 为 -0.046–0.088，9/9 连绝对精度门也未通过。修复 shuffle 解释不会自动使 observer 通过。

### 3.4 gauge 门可能受到弱激发坐标影响

本轮 Procrustes 对齐误差仅 0.032–0.049，低于 0.10；CKA 约 0.999。但交换子为 0.341–0.468，高于 0.10，导致所有 gauge 综合门失败。

潜维度配置为 32，表示的 entropy effective rank 仅约 3.3–3.8。有效秩不是严格代数秩，不能据此断言其余方向全为零；但它提示大部分数据能量集中在很少方向。在弱激发方向，Procrustes 的正交延拓可能不稳，而全空间 `||A0T-TA0||_F` 仍对它计分。

建议先检查训练表示协方差谱、奇异值间隙、对齐唯一性以及 held-out 数据加权的动力学对齐残差。列向量对齐 T 的一个候选诊断为：

\[
\frac{\mathbb E\|(TA_0-A_0T)z\|^2}
{\mathbb E\|A_0z\|^2+\epsilon}.
\]

T、可信子空间与阈值只由 train/validation 定义，再在 held-out 数据检验；还要检查未激发方向的可达性与增长。该诊断不能单独保证全空间或 OOD 稳定。

当前 gauge 指标还参与 checkpoint 选择，但训练目标没有直接优化该交换子；其数值意义必须先校准。原始 drift 0.110–0.183 也失败，不能只删除交换子门就宣称表示可行。

### 3.5 表示和 decoder 的误差尚未拆分

现有 joint 只解冻 encoder projection、decoder 最后一个卷积，同时训练 context/operator；decoder 的 latent-to-field lift 主体保留冻结，A0 和 JEPA target 也固定。这个保护性设计限制了表示/解码重建能力，可能已不适合承担更大幅度的 Markov 化。

历史 entry audit 的 reconstruction relative L2 约 0.289；本轮 forecast field relative L2 均值在 H8/H16/H32/H80 为 0.325/0.340/0.367/0.281。这些量来自不同采样集合，不能直接相减或当成不可突破的误差下界，但足以说明需要先拆分重构与传播误差。

精确的向量分解为：

\[
\hat U-U = [D(\hat z)-D(z^*)]+[D(z^*)-U].
\]

平方误差包含两项能量和交叉项，不能将两个 RMSE 直接相加。`z*` 应分别检查 frozen online 和 EMA teacher 编码，因为 D 原本主要对应 online 表示，二者不能默认为完全相同。

pullback 仅是第一项的局部一阶近似，不能直接消除第二项；本轮未做单因素消融，不能断言 pullback 本身导致退化。下一轮应把它变为可选消融项，避免继续同时改变训练目标和多个门。

压力是第三个通道，现有 compact 结果缺少独立 pressure error 和能量占比。不能仅从 field 与 velocity 的差别断言“压力就是主因”。需要同一批样本、同一 mask、同一无量纲约定下的逐通道误差与压力规范检查。

### 3.6 当前负结果不能证明所有双分支都无效

history-only fallback 停用静态 rank 后，未把它的容量转给动态支路；它与 full operator 的有效 rank/参数量不同。另外 observer 预训练预算、表征联合训练路径和输入权限也不同。因此 current fallback 的退化不能视为公平的“分解算子 vs 单算子”消融结论。

可以得出的判断是：当前复杂分解没有提供足以抵消其识别和验收成本的新增物理收益，应退出默认主路线，保留为历史实验与可选对照。

## 4. 推荐增减

| 模块/规则 | 建议 | 理由 |
|---|---|---|
| Koopman 连续时间传播、JEPA 初始化 | 保留 | 与本轮负结果不矛盾 |
| 物理边界、单位、非有限值和增长审计 | 保留 | 是所有新路线的基本合同 |
| R1–R3 与残差诊断 | 保留 | 历史是按证据选择的模型能力，R0 不恢复 |
| 旧 frozen/reference 与 from-scratch 结果 | 原样存档 | 旧结论、来源与可复现性保留 |
| 强制三工况 observer + history 优势准入 | 移出主任务 | 混合了输入估计、记忆必要性和动力学预测 |
| 双 static/dynamic 分解及其 centering/cross-basis 损失 | 暂退出默认 | 首先需要证明简单受控模型尚不够 |
| 全空间交换子硬门 | 暂作诊断，重新设计后预注册 | 需排除弱激发坐标不唯一和门控冲突 |
| pullback 等辅助目标 | 保留模块、按单因素消融启用 | 当前没有显示稳定新增物理收益 |
| 数据权限/transition 时间对齐 | 增加显式合同 | 定义模型实际可获取的过去、当前与未来输入 |
| 同样本解码误差拆分、pressure/force 审计 | 优先增加 | 定位优化上限与物理收益去向 |
| DMDc / 简单受控双线性 Koopman | 增加低成本基线 | 将已知输入的作用与内部动力学分开 |

## 5. 新路线的最小数学结构

先比较无历史的受控模型：

\[
\dot z=\bar A z+Bv(t)+\sum_j v_j(t)N_jz.
\]

这里 v 是中心化、无量纲化的独立实际控制输入；Bv 表达输入引起的偏移，N_j 表达状态与输入耦合。先用 DMDc 类型的 `N_j=0` 基线，再验证双线性项是否必要。可通过增广常数坐标 `[z;1]` 将仿射项写入统一生成元，以矩阵指数处理分段恒定输入；不必求 A 的逆。

这是一条需要验证的新模型族，不是当前离散边界控制系统的精确推导。对时变边界，应先分析 lifting `U=V+L(v)`：

\[
\dot V=F(V+L(v),v)-D_vL(v)\dot v.
\]

因此输入特征、输入导数或非线性输入项的必要性由边界约定和推导决定；突变输入需要分段/jump 处理，不能把它假设为光滑导数。输入项必须来自真实施加的边界或控制，不能凭拟合需求向物理系统添加虚构 forcing。

先在固定编码坐标上识别小模型。新路线可把旧 A0 作为初始化/参照，允许另行识别名义生成元 \(\bar A\)，但必须明确标记为新受控路线。它不修改旧 frozen 合同；若解冻表示，则 E/D/名义生成元应按统一坐标合同协调训练，而不是无限期同时要求 E 改变、A0 数值不变和多个表示锚定。

只有无历史模型在独立 validation 上仍有可学习残差，才添加：

\[
c_t=C(z_{t-H:t},v_{t-H:t}),\qquad
\dot z=\bar A z+Bv+\sum_jv_jN_jz+g_t\Delta A(c_t)z.
\]

历史模块以未来状态/物理残差预测为目标，不必先把 c 解释成 Re/U/dRe。先用等容量 instantaneous/history MLP 对照，再决定 Attention 是否值得加入。未知未来输入只在额外声明的输入模型、测量或不确定性预测合同下研究。

## 6. 执行顺序与停止条件

1. **已有 checkpoint 的诊断审计。** 在相同窗口拆分 propagation/reconstruction/交叉项，检查 online/EMA 差异、pressure energy、表示奇异谱和工况 transition 权限。输出一份独立 audit，不重新训练 18 个 joint runs。
2. **低成本受控识别。** 在固定表示上训练/拟合 DMDc、必要时双线性模型；所有候选使用同样输入权限、split、归一化和预注册预算。若解码误差支配，先对 warm-start decoder/表示做一个独立验证，避免继续堆算子。
3. **按证据加历史。** 无历史模型有可靠 field 收益后，比较同容量有/无历史模型；优势不成立就接受 R2，不强迫构造 R3。只有通过开发验证的候选才扩展到正式 nested matrix。

停止条件：

- 若无历史受控模型已有足够物理收益，保留它；历史/Attention 不是版本必需装饰。
- 若表示/decoder 重构误差支配，停止 operator 调参，转到受控数据上的表示设计。
- 若相同观测历史对应不可区分的不同外部未来输入，增加真实可用输入或定义概率/有限时域任务，不继续要求确定性 observer 解决信息缺失。
- 若受控识别和一次针对性表示改进仍无 validation field 优势，归档当前 benchmark 路线的负结果，再讨论新的状态表示或可解释闭合；不进行无限小步权重搜索。

旧 2% 和 nested 门不因本轮失败而回改。新路线如拆分 known/unknown 或修改诊断门，应预先声明新科学问题及验收，旧报告始终保持 NOT_SUPPORTED。当前 test 已多轮被观察，新路线最终确认应使用预先冻结的新轨迹/输入 schedule；原 test 仍保留为回归与历史比较，不能再被称为未见确认集。

## 7. 成熟方法依据及适用边界

- [Proctor, Brunton, Kutz — Dynamic Mode Decomposition with Control](https://arxiv.org/abs/1409.6358)：把输入作用与内部动力学分开，是本项目加入低成本受控基线的依据；不保证固定 learned latent 的有限维精确闭合。
- [Peitz, Otto, Rowley — Interpolated Koopman Generators](https://arxiv.org/abs/2003.07094)：control-affine 系统的生成元具有相应的输入依赖结构，支持双线性近似路线。本文的精确性条件不能直接套用到本项目的离散 LBM 边界实现。
- [Otto, Peitz, Rowley — Bilinear Models from Partially Observed Trajectories](https://arxiv.org/abs/2209.09977)：部分观测时，observable 构造需要考虑供给的输入序列；这支持未来历史表示同时纳入状态和已知输入。其方法不能让模型在无信息时预知任意外部干预。
- [Brunton et al. — Koopman Invariant Subspaces](https://arxiv.org/abs/1510.03007)：有限维线性表示具有具体适用条件。不能把“增加潜维度就必然全局线性”当作本项目保证，也不能把有关包含原始状态的有限不变子空间的限制扩展成对所有 nonlinear decoder 模型的否定。

上述是路线借鉴；本轮没有实现或验证新的受控模型。
