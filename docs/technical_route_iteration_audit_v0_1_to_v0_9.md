# JKA Model 技术路线逐次迭代审计（V0.1–V0.9 Added Phase 3.6）

> 状态日期：2026-09-03  
> 当前分支：`codex/v0.9-added-route`  
> 用途：在继续修改前，统一回答“每一轮为什么改、数学上改了什么、结果说明什么、哪些内容仍保留”，并给出截至目前唯一的整合技术路线。  
> 证据原则：本文严格区分**程序执行成功**、**局部机制得到支持**和**完整科学路线通过**。GPU 工作流 `PASS` 不自动等于科学结论 `SUPPORTED`。

---

## 1. 先给结论

当前路线不是一次完成的单一模型，而是按以下逻辑逐层搭建并审查：

1. V0.1–V0.2 建立数据、物理量纲、约束和复现实验契约；
2. V0.3–V0.5 建立连续时间 Koopman 主干，并使其真正学到传播相位、频率和衰减；
3. V0.6 用 JEPA 稳定和改善 Koopman 表征，而不替换 Koopman 动力学；
4. V0.7 判断名义 Koopman 残差是否可学习、是否需要历史；
5. V0.8 用残差监督学习动态上下文和充分性指标，但暂不改写算子；
6. V0.9 将上下文转成低秩时变生成元修正，并依次处理稳定性、可观测性、条件可辨识性和表示失配；
7. V0.9 Added Phase 3.6 已证明联合表示训练可以明显改善潜空间预测和部分物理量，但尚未把这些收益充分转移到解码物理场，因此**完整自适应 Koopman 机制仍未得到支持，V1.0 仍未就绪**。

截至目前，最可靠的技术判断是：

- **已成立**：连续生成元 Koopman、精确指数传播、物理单位约束、JEPA 表征增强、残差路由方法、圆柱绕流上的历史上下文价值。
- **部分成立**：低秩自适应算子可以改善潜空间多步预测；联合表示训练能降低表示漂移并改善涡量。
- **尚未成立**：由潜变量可靠推断工况；自适应算子在锁定测试上带来至少 2% 的全时域解码场收益；完整 V0.9 机制跨种子得到支持。
- **明确不应做**：为了宣称成功而降低既定阈值、重新启用“残差很小所以忽略”的 R0、把单次或训练集改进称为科学通过、在自适应算子尚未成立前增加最终残差闭合并混淆归因。

---

## 2. 不随版本改变的基础契约

这些原则在后续版本中应继续保持。

### 2.1 原始物理空间与模型空间分离

原始物理状态记为

\[
U_t\in\mathcal M_{\rm phys},
\]

模型输入可以使用仅由训练集拟合的标准化映射，但守恒、边界、PDE 残差和物理可观测量必须在原始单位下计算。流程固定为：

\[
\text{trajectory split}
\rightarrow \text{train-only normalization}
\rightarrow \text{window construction}.
\]

这样避免测试信息泄漏，也避免把“标准化空间的小误差”误当成“物理空间满足约束”。

### 2.2 连续时间 Koopman 契约

编码器给出潜状态

\[
z_t=E_\theta(U_t),
\]

名义动力学使用连续生成元 $A_0$：

\[
\dot z=A_0z,\qquad
z_{t+\Delta t}^{0}=\exp(A_0\Delta t)z_t.
\]

其中 $A_0$ 不是离散一步矩阵；离散传播算子是 $K_{\Delta t}=\exp(A_0\Delta t)$。因此不规则时间步仍保持一致的连续时间含义。

### 2.3 闭环与证据契约

- 多步预测必须将模型输出继续送回模型，不能只做 teacher forcing。
- 训练、验证、锁定测试以及种子必须分离。
- 路由、秩、历史长度和 checkpoint 只能由训练/验证确定，锁定测试只用于确认。
- `runs/` 保存训练过程和 checkpoint；`gpu_validation/v0_X/results/` 保存紧凑审阅证据。
- 同一验证 ID 再运行时解析为 `-r1`、`-r2` 等，不覆盖旧证据。

---

## 3. 逐版本、逐轮迭代记录

## 3.1 V0.1：软件与数学接口骨架

### 修改内容

- 建立可安装的 `src/jka_model` 包；
- 定义 `ProblemBatch`、`ProblemSpec`、`PhysicsConstraint` 和 checkpoint schema；
- 明确模型态与原始物理态的接口；
- 固定随机性、配置、Git 提交和数据来源的复现字段。

### 数学意义

V0.1 没有引入可训练模型，它解决的是“后续公式到底作用在哪个空间”的问题。特别是把

\[
\mathcal L_{\rm model}(U_{\rm normalized})
\quad\text{与}\quad
\mathcal L_{\rm physics}(U_{\rm raw})
\]

从接口层分开，防止守恒量和边界条件因标准化而失真。

### 保留结论

完整保留。它是后续全部版本的接口基础，而不是一次科学实验。

---

## 3.2 V0.2：轨迹数据、物理约束与可复现窗口

### 修改内容

- 实现轨迹级确定性拆分和数据指纹；
- 标准化器只在训练轨迹上拟合；
- 时间窗不能跨越不同轨迹；
- 实现有限值、状态可接受域、周期边界、质量守恒和离散 PDE 残差；
- 建立周期平流扩散测试问题和物理约束注册机制。

### 数学意义

定义质量守恒类损失时，以 rollout 初始状态为参考：

\[
\mathcal L_{\rm mass}
=\frac{1}{H}\sum_{h=1}^{H}
\left(
\frac{M(\hat U_{t+h})-M(U_t)}{|M(U_t)|+\epsilon}
\right)^2.
\]

它比“相邻两步分别比较”更直接度量长时间累计漂移。物理约束始终读取 raw state，从而保持量纲正确。

### 工程修正

早期单元测试曾要求数值残差小于 $10^{-20}$，而浮点计算得到约 $9.1\times10^{-13}$。这是错误的测试容差，不是守恒定律失效；后续使用与数据类型和离散误差一致的容差。

### 保留结论

完整保留。轨迹拆分、train-only normalization 和 raw-state physics 是后续实验可信度的底座。

---

## 3.3 V0.3：直接状态上的连续 Koopman

### 修改内容

- 在状态本身上学习连续生成元 (A)；
- 使用矩阵指数处理可变时间步；
- 增加闭环 rollout 和谱分析；
- 用阻尼振子做可解析检查，并记录 Duffing 非线性系统的能力边界。

### 数学修改

从“一步拟合”改为连续动力系统：

\[
\hat z_{t+h}=\left(\prod_{j=0}^{h-1}e^{A\Delta t_{t+j}}\right)z_t.
\]

当 $\Delta t$ 恒定时等价于 $e^{Ah\Delta t}z_t$，当 $\Delta t$ 不恒定时仍保持一致的生成元解释。

### 保留结论

保留连续生成元和精确指数传播。V0.3 尚无编码器，因此不能解决强非线性系统的提升问题。

---

## 3.4 V0.4：非线性提升与可解码 Koopman 表征

### 修改内容

- 引入编码器 $E_\theta$ 和解码器 $D_\psi$；
- 在学习到的潜空间中保留 V0.3 的连续 Koopman 核心；
- 增加重构、多步动力学、方差防塌缩和谱正则；
- 使用仿射潜空间对齐进行坐标不变的比较。

### 数学修改

基础目标变为

\[
\mathcal L_{\rm V04}
=\lambda_K\mathcal L_K
+\lambda_{K,m}\mathcal L_{K,\rm multi}
+\lambda_{\rm rec}\mathcal L_{\rm rec}
+\lambda_{\rm var}\mathcal L_{\rm var}
+\lambda_{\rm spec}\mathcal L_{\rm spec},
\]

其中

\[
\mathcal L_{\rm rec}=\|D_\psi(E_\theta(U_t))-U_t\|^2,
\qquad
\mathcal L_K=\|E_\theta(U_{t+1})-e^{A\Delta t}E_\theta(U_t)\|^2.
\]

方差项防止所有样本被压成同一潜向量。潜表示比较不直接逐坐标相减，而先进行允许旋转、缩放和平移的对齐，避免把 Koopman 坐标规范自由度误判为失败。

### 保留结论

保留编码器—生成元—解码器骨架和防塌缩设计。V0.4 仍不是物理场上的最终模型。

---

## 3.5 V0.5：物理约束场 Koopman 的三轮修正

### V0.5 初始正式验证：`v05-final-20260812`

**问题。** 训练损失下降，但传播频率错误约 96.31%，短/中/长 RMSE 分别约 0.7514/0.7218/0.6295。说明网络能重构场，却没有真正识别行波相位。

**原因。** 全局平均池化抹去了空间相位；只靠潜空间一步损失不足以强迫 (A) 学到正确频率。

### V0.5 动力学修正：`v05-final-20260813`

**结构修改。** 去除会消除相位的全局平均池化；让编码器保留行波位置和方向信息。

**损失修改。** 新增生成元一致性和解码预测：

\[
\mathcal L_{\rm gen}
=\left\|
\frac{z_{n+1}-z_n}{\Delta t}-A z_n
\right\|_2^2,
\]

\[
\mathcal L_{\rm forecast}
=\left\|
D_\psi\!\left(e^{A\Delta t}z_n\right)-U_{n+1}
\right\|_2^2.
\]

前者让连续生成元的局部导数正确，后者要求潜空间改进真正返回物理场。

**稳定性修改。** 对 (A) 的对称部分中不稳定方向施加惩罚：

\[
\mathcal L_{\rm stab}
=\left[\max\!\left(\lambda_{\max}\!\left(\frac{A+A^T}{2}\right),0\right)\right]^2.
\]

同时采用带阻尼倾向的生成元初始化，并提高生成元相对学习率。

**结果。** 频率误差降至约 0.2392%，短/中/长 RMSE 降至约 0.002994/0.002184/0.013580；但 physics/no-physics 配对门控尚未完全通过。

### V0.5 物理损失和选择修正：`v05-final-20260814`

**PDE 物理修正。** 旧的差分/梯形残差与数据生成器的谱传播不完全一致。周期平流扩散改为与生成数据一致的 Fourier 一步算子：

\[
\widehat U^{n+1}_{k_x,k_y}
=\exp\left(
[-\nu(k_x^2+k_y^2)-i(c_xk_x+c_yk_y)]\Delta t
\right)
\widehat U^n_{k_x,k_y}.
\]

这不是让物理项“更容易通过”，而是确保训练物理残差和基准 PDE 使用同一个离散数学对象。

**尺度修正。** 质量损失改成无量纲相对漂移，重新标定各物理权重，避免物理项因量纲差异压倒预测项。

**模型选择修正。** 分离 `best_forecast`、`best_physics` 和 `best_forecast_post_warmup`，防止物理 curriculum 尚未生效时过早选择 checkpoint。

**结果。** 频率误差约 0.2455%，短/中/长 RMSE 约 0.002841/0.002170/0.013767；三种子 physics/no-physics 配对审查后全部门控通过。物理模型并非每个原始误差都优于 no-physics，但满足预先约定的绝对精度与非劣门控。

### V0.5 最终去留

保留全部核心修正。V0.5 在单一解析周期平流扩散问题上通过；该结论不能外推为复杂流场通用通过。

---

## 3.6 V0.6：JEPA 增强 Koopman 表征

### 修改内容

V0.6 不替换 Koopman，而是在 V0.5 上增加 online/target 自监督预测：

\[
\mathcal L_{\rm V06}
=\mathcal L_{\rm V05}
+\lambda_{J1}\mathcal L_{J1}
+\lambda_{Jm}\mathcal L_{Jm}.
\]

online 编码器产生 $z_t=E_\theta(U_t)$，Koopman 预测器给出

\[
\hat z_{t+h}=e^{A_0\Delta t_{t:t+h}}z_t,
\]

target 编码器提供停止梯度的目标

\[
\bar z_{t+h}=\operatorname{sg}(E_{\bar\theta}(U_{t+h})),
\]

并使用指数滑动平均更新：

\[
\bar\theta\leftarrow\tau\bar\theta+(1-\tau)\theta.
\]

JEPA 的作用是让潜空间更强调“可预测的动力学结构”，降低仅靠像素/场重构形成的无关自由度。推理时不需要 target encoder。

### 正式结果：`v06-final-20260816T030842Z`

相对 no-JEPA 控制组：短、中、长时域 RMSE 分别改善约 40.13%、28.21%、29.96%；长时域算子误差改善约 49.15%；频率和衰减率误差分别改善约 32.75% 和 30.87%。质量漂移平均恶化约 82.58%，但仍满足固定绝对门控。训练时间增加约 19.81%，显存增加约 2.82%。

### 最终去留

保留 JEPA 作为**Koopman 表征训练项**。结论仍限于当时的窄解析问题；它说明 JEPA 有效，不说明后续自适应算子必然有效。

---

## 3.7 V0.7：名义 Koopman 残差的可学习性与记忆分类

### 初始设计

冻结 V0.6 主干，定义名义残差：

\[
r^0_{t+1}
=E_\theta(U_{t+1})
-e^{A_0\Delta t}E_\theta(U_t).
\]

比较以下 probes：零预测、线性、瞬时 MLP、历史 MLP 和打乱历史；历史长度 $H\in\{1,2,4,8,16\}$。

核心统计量为：

\[
P_R=1-\frac{E_{\rm best}}{E_0+\epsilon},
\qquad
G_H=\frac{E_M-E_H}{E_M+\epsilon},
\]

其中 $P_R$ 衡量残差相对零预测是否可学习，$G_H$ 衡量历史模型相对瞬时模型是否有稳定增益。

### 第一轮：`v07-final-20260816T123014Z`

残差可学习性为 `MODERATE`，闭环效果为 `POSITIVE`，但历史记忆为 `INCONCLUSIVE`。问题不是模型完全无效，而是历史优势没有通过参数匹配和跨种子稳定性检查。

### 第二轮：`v07-final-20260817T051348Z`

加入稳定性和统计修正：

- 参数量匹配的瞬时控制；
- 打乱历史负控制；
- 验证集锁定模型、历史长度和路由；
- teacher-forced 与 closed-loop 同时报告；
- 多 backbone seed × closure seed 的嵌套统计。

结果变为残差可学习性 `STRONG`、闭环效用为正，但记忆仍 `INCONCLUSIVE`。这说明“能预测残差”不等于“必须用历史预测残差”。

### 第三轮：残差显著性路由 `v07-revised-final`

引入显著性指标：

\[
S_R=
\frac{\mathbb E\|r^0\|_2^2}
{\mathbb E\|z_{t+1}-z_t\|_2^2+\epsilon}.
\]

当时把小 $S_R$ 归为 R0，即忽略残差。该数据上因一步残差很小而得到 R0，尽管残差本身可学习且历史包含信息。

### 第四轮：删除 R0，形成当前 R1–R3 路由

后续审查发现：一步残差小不代表长期不重要。若误差具有相干方向，累计项

\[
\sum_{j=0}^{H-1}K^{H-1-j}r^0_{t+j+1}
\]

在长时域仍可显著影响轨迹。因此 $S_R$ 只保留为诊断量，不再作为“丢弃残差”的硬门。

当前路由定义为：

- **R1**：残差没有稳定可学习性；保留并报告，但不声称可闭合；
- **R2**：残差可学习，但历史没有稳定增益；使用瞬时上下文；
- **R3**：残差可学习且历史有稳定增益；使用因果历史上下文，Attention 是候选而非默认结论；
- **INCONCLUSIVE**：证据不足，不强行归类。

### 最终去留

保留残差定义、嵌套 probe、负控制和 R1–R3；废弃 R0。旧 V0.7 结果是在旧路由规则下产生，不能把其中的 R0 当作当前有效分类，若要正式引用需按新规则重新聚合或重跑必要部分。

---

## 3.8 V0.8：残差监督的动态上下文学习

### 问题升级

前期解析行波过于接近可线性化动力学，不适合检验“历史是否带来新状态信息”。V0.8 转向二维圆柱绕流问题，并训练新的 V0.6 兼容 backbone，而不是直接复用旧问题权重。

### 数学修改

冻结 $E,D,A_0$，继续以

\[
r^0_{t+1}=E(U_{t+1})-e^{A_0\Delta t}E(U_t)
\]

作为监督目标。上下文编码器从当前或历史潜变量得到 $c_t$：

\[
c_t=C_\phi(z_{t-H+1:t},\Delta t,\mu),
\]

并训练两个头：

\[
\hat r_{t+1}=R_\rho(c_t,z_t,\Delta t,\mu),
\qquad
\hat m_t=Q_\chi(c_t),
\]

其中 $\hat m_t$ 估计名义 Koopman 在当前状态下是否充分。损失使用只由训练集计算的残差尺度进行标准化；各输出头零初始化，使训练初始状态严格等于名义模型。

### 正式结果：`v08-final-20260820T111016Z`

- 物理问题：二维圆柱绕流；
- 路由：R3；选中模型：`history_mlp`；
- 残差可学习、历史增益、动态上下文和闭环效用得到支持；
- H80 平均增益约 4.53%，物理门控通过；
- 但仅 2/3 backbone seed 联合通过，因此严格 V0.9 handoff 要求的 3/3 未满足，V0.9 readiness 为 `NOT_READY`。

### 后续 handoff 修正

旧 checkpoint 曾因“保存配置”和“当前解析配置”哈希比较对象不一致而触发 `config hash mismatch`。修正后使用 checkpoint 自身保存的 resolved config 验证 provenance，并对旧结果进行重新评估。严格 readiness 仍未被改写为通过；后续 V0.9 仅以 `supported` 的**探索性条件证据**继续。

### 最终去留

保留 V0.8 的 R3 上下文和残差监督。V0.8 没有实现 $A_t$，它只回答“上下文能否预测缺失动力学”，不回答“能否安全改变 Koopman 生成元”。

---

## 3.9 V0.9 原始路线：低秩自适应 Koopman

### 初始数学结构

上下文被映射为低秩算子坐标：

\[
\eta_t=G(c_t),
\qquad
A_t=A_0+U\operatorname{diag}(\eta_t)V^T,
\]

并继续使用精确传播

\[
z_{t+1}=e^{A_t\Delta t}z_t.
\]

低秩结构的目的不是减少所有参数，而是把“工况/历史变化”限制在少量可解释方向，避免直接学习一个任意稠密时变矩阵。

### 第一轮：`v09-full-20260821T033247Z`

工作流完成，但 H32/H80 和物理门控失败；rank sweep 达到上限 8；自适应机制 `NOT_SUPPORTED`。这说明单纯把上下文接到低秩算子上，不足以保证闭环稳定和物理改善。

### 第二轮稳定化：`v09-stabilized-20260821T114835Z`

引入有界坐标和信任门：

\[
g_t=\sigma(h(c_t,U_t)),
\qquad
\eta_t=g_t\eta_{\max}\tanh(q_t),
\]

以及相对名义算子的传播增长、算子平滑、teacher-free 多时域 curriculum 和物理项。核心含义是：自适应修正只能在受控幅度内工作，并且不能比 $A_0$ 引入更强的非物理增长。

结果：长时潜空间稳定性改善，但物理门控仍失败；选中 rank 12，自适应机制仍未得到支持。

### 第三轮可观测量增强：`v09-observable-20260821T145859Z`

从冻结解码器构造速度、涡量、散度、边界、升力和阻力等观测量，并加入非劣约束。通用非劣门采用分辨率地板：

\[
x_{\rm adapt}
\le x_0(1+\rho)+\delta_{\rm abs}+\delta_{\rm res},
\]

避免当基线接近零时相对误差无穷放大。

结果：算子解释残差和长时稳定性得到支持，但观测量与物理门仍失败，整体机制仍 `NOT_SUPPORTED`。这表明主要瓶颈已从“数值爆炸”转向“潜空间收益不能可靠转成物理收益”。

### 最终去留

保留有界低秩修正、相对名义稳定性、teacher-free rollout 和可观测量审查；不接受原始 V0.9 为科学通过。

---

## 3.10 V0.9 Added Phase 1：让优化目标与物理门控一致

### Phase 1 初始轮：`v09-added-phase1-20260822T034843Z`

目标是判断问题属于“算子优化不充分”还是“表示本身不适合”。增加四层误差归因：

1. 数据/不可约误差；
2. 编码—解码重构误差；
3. 名义 Koopman 误差；
4. 自适应 Koopman 误差。

同时引入由训练集拟合的稳健 observable scale、Huber 损失、升阻力窗口损失和不等式增广拉格朗日。

对门控 $g_k(\theta)\le0$，使用

\[
\mathcal L_{\rm AL}
=\mathcal L_{\rm primary}
+\sum_k\lambda_k[g_k]_+
+\frac{\rho_k}{2}[g_k]_+^2.
\]

初始轮显示算子残差有可解释空间，但 observable、物理地板和长时稳定性失败，表示路线被阻塞。

### Phase 1 r1：`v09-added-phase1-r1-20260823T070404Z`

针对第一轮的目标竞争和门控时序，修改为：

- 零基线物理量使用绝对阈值，不做病态相对比；
- physical curriculum 完全生效后才启用 PCGrad；
- rollout 明确加入 H80，并让 rank 选择看到长时表现；
- 用最坏时域约束代替加权平均，防止 H4 改善掩盖 H80 失败；
- burden 使用最大值审查；
- dual ascent 延迟、阻尼并设上限；penalty 增长减慢并封顶。

PCGrad 只在梯度冲突时投影：

\[
g_i\leftarrow g_i-
\frac{g_i^Tg_j}{\|g_j\|^2+\epsilon}g_j,
\quad g_i^Tg_j<0.
\]

结果：严格物理/observable 通过数从 4/18 提升到 14/18，H80 从 1/18 提升到 13/18；已知工况得到支持，但 latent-inferred 未通过，历史动态控制为 0/9，整体仍 `NOT_SUPPORTED`。

### 最终去留

保留稳健尺度、多时域最坏项、延迟约束优化和误差归因。结论转向：算子在已知工况下可优化，真正瓶颈可能是工况可辨识性与历史创新的分离。

---

## 3.11 V0.9 Added Phase 2：静态工况与动态历史创新分解

### Phase 2 数学主线

将一个混合修正拆为静态工况分支和动态历史分支：

\[
A_t=A_0+\Delta A_s(\hat q_t)+\Delta A_d(h_t,\hat q_t),
\]

\[
\Delta A_s
=\sum_{j=1}^{r_s}\phi_j(\hat q_t)u_j^s(v_j^s)^T,
\qquad
\Delta A_d
=\sum_{k=1}^{r_d}\xi_{t,k}u_k^d(v_k^d)^T.
\]

其中 $q_t=(Re,U_\infty,\dot{Re})$。`known` 路径直接使用真实工况，`latent_inferred` 路径使用观察器 $\hat q_t=Q(c_t,o_t)$。两条低秩基要求近似 Frobenius 正交，并要求条件于 $q$ 的动态坐标近似零均值，以减少静态与动态分支互相抢解释权。

### 第一轮数值失败：`v09-added-p2-identifiable-20260823T100133Z`

训练在 seed 47、latent-inferred、rank 2、operator seed 907 处产生非有限上下文，工作流 `FAILED_INCOMPLETE`。

修正包括：观察器输出平滑有界并改用 Huber；算子消费停止梯度的 (hat q)；两个分支增加基于对称 Frobenius/对数范数的信任区；在损失和梯度处提前拒绝非有限值。

### 第二轮稳定但过约束：`v09-added-p2-stable-20260823T110904Z`

18/18 数值稳定，但修正负担仅约 0.55%–0.67%；known 一步增益约 0.64%，latent-inferred 约 0.02%；观察器 0/18，physics 12/18。说明“稳定”是通过把修正压得过小得到的，科学效果不足。

### 第三轮连续三阶段训练：`v09-added-p2-continuous-20260823T133237Z`

训练改成连续且职责明确的三阶段：

1. `static_oracle`：用真实 (q) 识别静态分支；
2. `dynamic_residual_oracle`：冻结/停止静态分支梯度，拟合剩余历史创新；
3. `observer_calibration`：单独校准 (Q(c,o))，通过 observer gate 后才允许 latent joint refinement。

总修正采用连续预算 $0.05\rightarrow0.10\rightarrow0.15$，rank 只由 known-oracle 路径按 burden、H80、目标和最小秩的词典序选择。结果 H80 增益约 2.09%、operator-explained residual 约 5.54%，但 H8/H16/H32 均低于 1%，观察器仍未就绪，physics 仅 7/18；rank 8 还越过物理可行边界。

### 第四轮 Phase 2 fix2：物理优先选择与可辨识性修正

修改为：

- rank/checkpoint 首先满足物理可行性，再比较 burden、H80、目标和秩；若无可行解，最小化最坏物理违反；
- checkpoint patience 只在 curriculum 成熟后计数；阶段切换恢复上一阶段选中状态并重置优化器；
- 使用 batch×horizon 的最坏项，而非平均抵消；
- 观察器只用因果特征
  \[
  o_t=[z_t,\operatorname{mean}(z_{t-H:t}),\operatorname{trend}(z_{t-H:t})];
  \]
- 先从历史中减去工况可解释部分：
  \[
  \hat h(q_t)=M_\theta(q_t),\qquad
  h_t^\perp=h_t-\hat h(q_t),\qquad
  \xi_t=H_\psi(h_t^\perp,q_t);
  \]
- rollout 损失改为相对名义误差的无量纲形式。

### 最终 Phase 2：`v09-added-p2-physical-20260824T105209Z`

工作流完成，选中 rank 2；物理 16/18、observable 得到支持，但 rollout skill 0/18，观察器、动态分支和历史辨识均未得到支持。最终分类为 `LATENT_CONDITION_NOT_IDENTIFIABLE`，完整机制 `NOT_SUPPORTED`。

### 最终去留

保留静态/动态分解作为诊断结构和负结果；停止继续调 Phase 2。原因不是双算子分解在数学上必然错误，而是当前冻结表示中，工况在潜历史里不可稳定辨识，继续调权重无法修复信息缺失。

---

## 3.12 V0.9 Added Phase 3：表示是否阻塞自适应算子

### Phase 3.0/3.1 审计：`v09-added-p3-audit-20260826T043840Z`

审计三个问题：

1. 重构物理是否可接受；
2. 潜空间 round-trip 是否接近闭合；
3. 名义切向 $A_0z$ 解码后是否仍留在物理流形附近。

round-trip 定义为

\[
\varepsilon_{\rm rt}
=\frac{\|E(D(z))-z\|_{\rm RMS}}
{\|z\|_{\rm RMS}+\epsilon}.
\]

最初报告把散度 RMS 与 MSE 阈值直接比较，量纲不一致，错误判为 reconstruction physics fail。修正为 RMS 与

\[
\sqrt{\text{max\_divergence\_mse}}
\]

比较后，重构物理为 `PASS`、round-trip 为 `FAIL`、名义切向为 `PASS`，所以下一步应是联合 Markov 表示，而不是先换物理解码器。原报告保留，修正通过 addendum 记录。

### Phase 3.2 候选物理解码器

准备了基于流函数的候选结构：

\[
u=\partial_y\psi,
\qquad
v=-\partial_x\psi,
\]

它可在离散内部自然满足无散条件，固壁施加 no-slip，入口/远场显式约束，压力单独解码。由于修正后的审计表明原解码器的重构物理已经通过，该候选没有被强行替换进主线。

### Phase 3.3 初始联合训练：`v09-added-p3-joint-20260826T053347Z`

训练 encoder、decoder、context 和 adaptive operator，同时冻结 $A_0$。所有 rollout 都从 raw field 在线重编码：

\[
U_t\xrightarrow{E_\theta}z_t
\xrightarrow{e^{A_t\Delta t}}\hat z_{t+1}
\xrightarrow{D_\psi}\hat U_{t+1}
\xrightarrow{E_\theta}z_{t+1}^{\rm online}.
\]

这一点防止表示变化后继续使用 V0.8 缓存 latent/residual，后者会成为陈旧监督。

初始 18 个 run 都在 epoch 49 停止，因为 early-stop patience 在 curriculum 成熟前已经累计；当时的 0.444 feasibility 也只计算物理和 drift，门控不完整。因此该轮只能视为流程诊断。

### Phase 3.3 r1：`v09-added-p3-joint-r1-20260827T070153Z`

修正为：curriculum 成熟后才开始 patience 和 checkpoint 选择；完整门控顺序为物理、表示漂移、round-trip、观察器、全部预测时域；增加解码场、速度、涡量、散度和边界指标。

结果：预测 18/18、物理 18/18，但表示可行 0/18、严格通过 0/18。潜空间相对冻结基线的 H8/H16/H32/H80 增益约 21.67%/20.64%/15.52%/8.80%，解码场仅约 1.10%/0.82%/0.58%/0.77%；表示漂移约 0.254–0.527，高于 0.10 固定上限；观察器仅 3/9。

**关键诊断。** 自适应动力学在潜空间确实变好，但 encoder/decoder 的坐标变化和解码瓶颈使收益不能传到物理场。

### Phase 3.4/3.5 匹配三路线：`v09-added-p3-routes-20260829T025754Z`

为区分“冻结主干限制了适应”与“联合训练只是坐标重排”，进行完全匹配的三路线：

- `frozen`：冻结 V0.8 表示；
- `joint`：在同一主干上联合微调；
- `from_scratch`：重新初始化 encoder/decoder/context/EMA target，并训练可投影为非扩张的 $A_0$。

三条路线匹配数据拆分、3 backbone seeds × 3 operator seeds × 2 condition modes、预算、门控和负控制。表示比较增加坐标不变指标：linear CKA、正交 Procrustes NRMSE 和有效秩。

完整科学门要求：内部物理、round-trip、预测、观察器、H8/H16/H32/H80 每个时域的解码场增益均至少 2%、速度/涡量非劣以及嵌套种子支持。

结果：joint matched 0，from-scratch matched 0，结论 `NO_PHASE3_ROUTE_SUPPORTED`。from-scratch 的场误差在 H8/H16/H32/H80 反而恶化约 8.5%/9.6%/10.8%/22.9%，round-trip 0/18、观察器 0/9。因此 from-scratch 被保留为已完成负控制，不再重复。

### Phase 3.6 解码物理联合精修：`v09-added-p3-physical-joint-20260830T085348Z`

为把潜空间收益直接传到物理场，加入仅由训练数据计算的多时域解码监督：

\[
\mathcal L_{\rm dec}
=\sum_h\omega_h\left[
2\frac{\|\hat U_h-U_h\|_2^2}{\|U_h\|_2^2+\epsilon}
+\frac{\|\hat{\boldsymbol u}_h-\boldsymbol u_h\|_2^2}
{\|\boldsymbol u_h\|_2^2+\epsilon}
+0.2\frac{\|\hat\omega_h-\omega_h\|_2^2}
{\|\omega_h\|_2^2+\epsilon}
\right].
\]

表示漂移约束改为按固定阈值 0.10 归一化的违反量：

\[
\mathcal L_{\rm drift}
=\left[\max\left(\frac{d_{\rm rep}}{0.10}-1,0\right)\right]^2.
\]

encoder/decoder 在物理监督开始前冻结，随后平滑解冻；context/operator 从 epoch 0 训练。checkpoint 按“硬可行性 → 解码误差 → 潜空间增益”排序，而不是优先选择漂亮的 latent 指标。

**最新结果。** 18/18 完成；物理和预测均 18/18。相对上一联合路线，平均 representation drift 从约 0.416 降到 0.152（约 -63.4%），forecast loss 约 -51.6%，JEPA consistency 约 -82.5%；linear CKA 约 0.9988、Procrustes NRMSE 约 0.042；H8–H32 涡量改善约 5%–6%。

但 representation drift 仍 0/18 通过（最佳仍约 0.108），round-trip 12/18，latent-only observer 1/9；H8/H16/H32/H80 解码场平均改善仅约 0.90%/0.83%/0.70%/0.71%，全部低于预先固定的 2% 门。因此 matched route 0/18、嵌套 backbone 支持 0%，科学结论仍为 `JOINT_REFINEMENT_NOT_SUPPORTED`，V1.0 `NOT_READY`。

### Phase 3 最终去留

- 保留 raw-field online re-encoding、冻结 $A_0$、坐标不变表示审查和解码物理监督；
- 保留 frozen 与 from-scratch 作为固定参考/负控制，不重复训练；
- Phase 3.6 是**有明确局部改进但未达到完整门控**，不能写成失败无价值，也不能写成路线通过；
- 当前瓶颈已缩小为：表示漂移/round-trip、latent condition observer，以及 latent 改进到 decoded field 改进的传递效率。

---

## 3.13 V0.9 Added Phase 3.7：物理对齐表示与观察器准入（已实现，GPU 待验证）

针对 Phase 3.6 的三个剩余瓶颈，新增以下结构，但不改变冻结参考、(A_0)、from-scratch
负控制和 2% decoded-field 门槛：

1. 使用冻结参考 decoder 的 Jacobian-vector product，把潜误差映射到 field、velocity 和
   vorticity 的局部物理度量后训练；
2. 在 Procrustes 对齐后增加 (|A_0T-TA_0|_F/|A_0|_F)，区分无害坐标旋转和与名义
   动力学不相容的表示变化；
3. condition observer 在 operator/representation 训练前独立比较真实历史、瞬时、打乱历史和
   均值控制。未通过准入时，latent-inferred 路径不允许 (hat q_t) 控制算子，而降级为明确标记的
   history-only dynamic route。

这一阶段当前只有本地代码和针对性测试证据，尚无正式 GPU 科学结果，因此不能据此更新
`JOINT_REFINEMENT_NOT_SUPPORTED` 或 V1.0 `NOT_READY`。完整数学与验收合同见
`docs/v0_9_added/phase_3_7_implementation.md`。

---

## 4. 验证基础设施的独立迭代

这些修改不改变模型数学，但决定结果是否可信。

| 迭代 | 出现的问题 | 修正后的契约 |
|---|---|---|
| GPU 脚本导入 | 直接运行脚本时项目根目录不在模块搜索路径，出现 `No module named gpu_validation` | 所有入口自行解析并加入项目根目录，可从仓库根目录一行启动 |
| fail-fast / resume | 早期测试失败与后续训练的关系不够明确 | 关键阶段失败立即生成 failure artifact；允许显式 resume，但正式重跑可使用新 ID 从头开始 |
| checkpoint provenance | V0.5/V0.8 曾出现当前 config 与 checkpoint 保存 config 哈希对象不一致 | checkpoint 按自身 resolved config、schema、Git commit 和数据指纹核验；不通过时拒绝加载 |
| ID 冲突 | 重复 ID 会覆盖或阻塞结果 | 请求 ID 自动解析为原 ID、`-r1`、`-r2`……；旧证据不可覆盖 |
| 输出策略 | 长训练阶段看似“没有运行” | 大阶段必须显示 `START` 和最终 `PASS/FAIL`；GPU 正式训练不逐 epoch 刷屏，只输出阶段最终结果并保留日志 |
| 结果布局 | `runs/` 有 checkpoint，但 `results/` 缺审阅报告 | `runs/v0.X/<resolved-id>/...` 保存重资产；`gpu_validation/v0_X/results/<resolved-id>/` 必须形成 completion、聚合结果和 Markdown 报告 |
| 干净工作区 | results 被生成后参与 dirty-source 检查 | 正式验证只锁定源代码/配置相关路径；生成结果与训练资产不应被误判为源代码修改 |
| 测试收敛 | 每次修改重复全部历史测试，代价过高 | 新功能运行新增测试和最小必要回归；完整 GPU 科学矩阵只在跨模块、高风险或正式验收时运行 |

---

## 5. 截至目前最新的整合技术路线

下面是当前应作为后续设计基准的统一结构。它是**整合后的设计合同**，不等于所有模块已经联合科学通过。

```text
raw physical state U_t
        │
        ▼
train-only normalization ──► online encoder E_theta ──► Koopman latent z_t
                                                         │
                           JEPA target encoder (train only, stop-gradient/EMA)
                                                         │
                                                         ▼
                                  nominal continuous generator A0
                                      z^0_{t+1}=exp(A0 dt) z_t
                                                         │
                                                         ▼
                                nominal residual r^0_{t+1}
                                                         │
                                   R1 / R2 / R3 evidence router
                                                         │
                                causal context c_t and adequacy m_t
                                                         │
                         ┌───────────────────────────────┴──────────────────────┐
                         ▼                                                      ▼
              static condition branch                              dynamic innovation branch
                Delta A_s(q_hat)                            Delta A_d(h_perp, q_hat)
                         └───────────────────────────────┬──────────────────────┘
                                                         ▼
                                  A_t = A0 + Delta A_s + Delta A_d
                                      z_{t+1}=exp(A_t dt) z_t
                                                         │
                              raw-field online re-encoding during joint training
                                                         │
                                                         ▼
                               decoder D_psi ──► physical field U_hat
                                                         │
                                 field / velocity / vorticity / divergence /
                                  boundary / lift / drag / long-horizon gates
```

### 5.1 当前统一数学表达

1. **名义表征与动力学**

   \[
   z_t=E_\theta(U_t),\qquad
   z_{t+1}^{0}=e^{A_0\Delta t}z_t.
   \]

2. **JEPA 表征约束**

   \[
   \mathcal L_{\rm JEPA}
   =\sum_h\left\|
   P_h(z_t)-\operatorname{sg}(E_{\bar\theta}(U_{t+h}))
   \right\|^2.
   \]

   当前实现中预测器与连续 Koopman 传播相结合，target 分支只服务训练稳定性。

3. **残差与路由**

   \[
   r^0_{t+1}=E(U_{t+1})-e^{A_0\Delta t}E(U_t),
   \]

   用 $P_R$ 与 $G_H$ 选择 R1/R2/R3；$S_R$ 仅诊断，不再丢弃残差。

4. **上下文与充分性**

   \[
   c_t=C(z_{t-H+1:t},\Delta t,\mu),
   \qquad
   \hat m_t=Q(c_t).
   \]

5. **低秩自适应生成元**

   \[
   A_t=A_0+\Delta A_s(\hat q_t)+\Delta A_d(h_t^\perp,\hat q_t),
   \]

   两个分支都必须满足幅度预算、稳定性/物理门和可辨识性检查；初始化时修正为零，因此 $A_t=A_0$。

6. **联合表示仅作为受控补救路线**

   当冻结表示被证据证明阻塞时，才允许训练 $E,D$，并要求 raw-field online re-encoding、$A_0$ 冻结、drift/round-trip/CKA/Procrustes 审查，以及解码物理监督。

7. **最终剩余残差闭合暂缓**

   理论上可写成

   \[
   r^0=r^{\rm op}+r^{\rm rem},
   \]

   但当前只验证 $r^{\rm op}$ 能否由自适应生成元可靠解释。由于该机制尚未完整通过，`r_rem` 的额外闭合或持久残差状态仍应禁用，以免无法区分收益来自算子还是额外黑箱修正。

---

## 6. 当前方向是否合理

### 合理且应继续保留的方向

1. **Koopman 是动力学主轴，JEPA 是表征增强。** 两者功能不冲突：JEPA 让潜变量可预测，Koopman 给出连续时间可解释传播。
2. **先诊断残差，再决定是否使用历史。** R1–R3 避免所有问题都强行使用 Attention，也承认残差类型与问题有关。
3. **低秩、零初始化、相对 $A_0$ 的信任区。** 使自适应部分是可审计的小修正，而不是重新学习全部动力学。
4. **静态工况与动态历史分离。** 即使 Phase 2 没通过，这个分解揭示了“已知工况可优化、潜变量工况不可辨识”的真实瓶颈。
5. **把潜空间和物理场分别验收。** Phase 3 证明 latent gain 可以很大而 field gain 很小；若只看 latent loss，会得出错误结论。
6. **保留冻结和 from-scratch 控制。** 它们使联合训练的收益有可归因参照，不应因负结果而删除。

### 当前仍需警惕的结构问题

1. **潜空间不是天然可观测工况坐标。** 当前 observer 通过率低，说明 $Re,U_\infty,\dot Re$ 未稳定编码在可被简单读出的历史中。
2. **decoder 是收益传递瓶颈。** Phase 3.6 已把 latent rollout 明显改善，但解码场收益仍不足 1%，说明潜误差与物理误差的度量几何没有完全对齐。
3. **round-trip 与 drift 仍冲突。** 强化物理预测时表示会移动；限制移动时又可能压制 Markov 化。需要把表示规范自由度和真实语义漂移继续分开，而不是只调一个权重。
4. **压力/力学通道可能弱约束。** 流函数结构可保证速度无散，但压力、升阻力与尾迹动力学仍需要独立可观测和边界证据。
5. **当前结论仍是单问题证据。** 圆柱绕流比解析行波更合适，但尚不能证明跨 PDE、跨几何或未见工况的通用性。

### 下一轮修改的决策边界

后续若继续，不应回到 Phase 2 继续调双分支权重，也不应重跑已经完成的 from-scratch 负控制。更合理的目标应直接针对三个剩余瓶颈：

1. 让 decoded-field 多时域目标直接约束可训练表示，同时保持坐标不变的 drift/round-trip 约束；
2. 将工况观察器与 operator 梯度进一步解耦，先证明工况信息在原始/潜历史中可辨识，再允许其控制算子；
3. 单独审查 decoder/pressure/force 通道是否限制 latent gain 的物理传递。

在这些条件满足前，不进入最终残差闭合，也不宣称 V1.0 ready。

---

## 7. 当前证据分级

| 层级 | 当前判断 | 说明 |
|---|---|---|
| V0.1–V0.4 软件/数学骨架 | 保留 | 接口、连续生成元、lifting 和闭环均为后续基础 |
| V0.5 物理 Koopman | 通过（窄问题） | 解析周期平流扩散、三种子、物理/无物理配对 |
| V0.6 Koopman + JEPA | 通过（窄问题） | 多时域和谱误差明显改善，物理绝对门保持 |
| V0.7 残差方法论 | 方法保留，旧 R0 结论失效 | 当前必须使用 R1–R3；旧结果需按新规则解释 |
| V0.8 动态上下文 | 局部支持 | 圆柱绕流 R3 和 H80 增益成立；严格 3/3 handoff 未通过 |
| V0.9 原始自适应算子 | 未支持 | 稳定性改善，但物理/observable 未联合通过 |
| V0.9 Added Phase 1 | 部分支持 | known-condition 可优化；latent-inferred 和独立历史失败 |
| V0.9 Added Phase 2 | 未支持，已冻结 | 最终分类为 latent condition not identifiable |
| V0.9 Added Phase 3.3 | 机制局部有效 | latent 改善显著，field 传递不足且 drift 失败 |
| V0.9 Added Phase 3.4/3.5 | 三路线均未支持 | from-scratch 是完成的负控制，不再重复 |
| V0.9 Added Phase 3.6 | 技术通过、科学未支持 | 物理/预测 18/18；2% field、drift、observer 和嵌套支持未通过 |
| V0.9 Added Phase 3.7 | 已实现、GPU 待验证 | 物理拉回度量、动力学 gauge 和独立 observer admission |
| V1.0 readiness | `NOT_READY` | 不能用局部指标替代完整 matched gate |

---

## 8. 文档与结果索引

### 路线与版本文档

- 根路线：`revised_koopman_attention_adaptive_framework.md`
- V0.1：`docs/implementation_notes_v0_1.md`
- V0.2：`docs/v0_2_code_walkthrough.md`
- V0.3：`docs/v0_3_code_walkthrough.md`
- V0.4：`docs/v0_4_code_walkthrough.md`
- V0.5：`study/V0_5_Koopman_adjustments_and_validation_20260816.md`
- V0.6：`study/V0_6_JEPA_validation_20260816.md`
- V0.7：`docs/v0_7/status.md`、`docs/v0_7/residual_decision_report.md`
- V0.8：`docs/v0_8/status.md`、`docs/v0_8/v0_8_scientific_report.md`
- V0.9：`docs/v0_9/status.md`、`docs/v0_9/v0_9_scientific_report.md`
- V0.9 Added：`docs/v0_9_added/phase_1_implementation.md`、`phase_2_implementation.md`、`phase_3_implementation.md`

### 最关键的现有 GPU 结果

- `gpu_validation/v0_9/results/v09-added-phase1-r1-20260823T070404Z/`
- `gpu_validation/v0_9/results/v09-added-p2-physical-20260824T105209Z/`
- `gpu_validation/v0_9/results/v09-added-p3-audit-20260826T043840Z/`
- `gpu_validation/v0_9/results/v09-added-p3-joint-r1-20260827T070153Z/`
- `gpu_validation/v0_9/results/v09-added-p3-routes-20260829T025754Z/`
- `gpu_validation/v0_9/results/v09-added-p3-physical-joint-20260830T085348Z/`

### 文档时效提醒

部分阶段文档是“实现时状态”，例如 `phase_3_implementation.md` 仍可能写着 Phase 3.6 GPU pending。对当前结论，应优先使用本文和对应 validation ID 下的 `completion.json`、聚合 JSON 与 `report.md`；不要只看较早的 status 文本。

---

## 9. V0.9 Added 的 Git 迭代定位

以下提交用于在换电脑后恢复主要阶段，不代表每个提交都单独构成科学结论：

| 阶段 | 主要提交/结果提交 |
|---|---|
| Phase 1 初始 | `c7f62e0` / `ec16c8c` |
| Phase 1 fix1 | `995d68c` / `525cfea` |
| Phase 2 初始与非有限值修复 | `4a78a14`、`d2124b5` / `c70b6c1` |
| Phase 2 连续三阶段 | `b642e47` / `08c814c` |
| Phase 2 fix2 与最终结果 | `34b74c1` / `69d78d5` |
| Phase 3 表示审计 | `a47b28b` / `190ef67` |
| Phase 3.3 初始联合训练 | `41c188c` / `fa8a22f` |
| Phase 3.3 r1 | `3721438` / `371bf5e` |
| Phase 3.4/3.5 匹配三路线 | `6c2f2eb` / `8b6de07` |
| Phase 3.6 | `65d2634` / `bb8be54` |

如果 Git 提交说明与正式结果报告冲突，以验证目录中绑定 Git commit 的 `completion.json` 和锁定测试聚合结果为准。

---

## 10. 一句话总览

本项目已经从“学习一个线性潜动力学”推进到“由 JEPA 稳定表征、由残差证据选择上下文、由低秩时变生成元进行受控修正、并在真实物理场上进行匹配验收”的统一路线；当前真正尚未解决的不是潜空间能否优化，而是**如何在不破坏表示一致性和工况可辨识性的前提下，把潜空间优化稳定地转移为跨时域、跨种子的物理场优化**。
