# Koopman-Structured Physical JEPA World Model

> 面向流体、燃烧与一般 PDE 动力系统的结构化潜在世界模型设计文档  
> 目标：将 **JEPA 的潜在预测、Koopman 的结构化动力学、Attention 的跨模态/长时耦合建模** 与 **物理约束、可选择物理解码和控制接口** 统一到一个可直接实现的软件架构中。

> **Revision v2.2**：重新整理 latent 的数学角色。核心状态不再采用旧版 `z_phys + z_K + z_R` 三个并列 latent，而改为 **PhysicsConstraint / optional PhysicalProbe + Koopman dynamical state `z_K` + history-dependent closure state `z_R`**。物理性由 governing equations、边界条件、守恒/耗散、状态可容许性与 constitutive relations 定义，而不是通过穷举物理量形成第三个 latent。`z_R` 只在冻结 Koopman 表示后，由同一 Koopman 坐标系中的真实未来 latent 与 Koopman base prediction 的差值构造监督目标。新增训练用轻量 decoder，使物理方程能够直接约束 decoded prediction；该 decoder 可在控制类部署中丢弃。工程契约继续沿用 v2.1 的时间对齐、EMA、closed-loop 防泄漏、checkpoint、梯度所有权和版本化验收规则。

---

# 第一部分：全局架构与数学定义（凝练版）

## 1. 研究目标

考虑受控、参数化、可能部分可观测的物理动力系统

\[
\frac{\partial U}{\partial t}=\mathcal F(U,a;\mu),
\]

或离散形式

\[
U_{t+1}=\Phi_{\Delta t}(U_t,a_t;\mu),
\]

其中：

- \(U_t\in\mathcal X\)：高维物理状态，例如二维/三维流场 \([\rho,u,v,w,p,T,Y_i,\ldots]\)；
- \(a_t\in\mathcal A\)：控制量，例如质量流量、入口压力、阀门开度、喷注参数、外力等；
- \(\mu\)：静态或慢变物理参数，例如 Reynolds 数、Mach 数、几何参数、边界条件、材料参数等；
- \(\Phi_{\Delta t}\)：真实但昂贵、未知或仅能通过 CFD/FEM/实验访问的演化算子。

目标不是直接学习一个完全黑箱的

\[
\hat U_{t+1}=F_\theta(U_t,a_t),
\]

而是构造一个**结构化潜在世界模型**：

\[
\boxed{
U_t
\xrightarrow{E_\theta}
z_t
\xrightarrow{\text{Koopman backbone + Attention residual}}
\hat z_{t+1}
\xrightarrow{\text{task-dependent readout}}
\hat y_{t+1}\;\text{or}\;\hat U_{t+1}
}
\]

并要求潜在状态同时满足四个性质：

\[
\boxed{
\text{Predictive}
+\text{Physically grounded}
+\text{Dynamically structured}
+\text{Controllable}
}
\]

核心研究命题可表述为：

> **Can a world model learn a latent state that is simultaneously predictive, physically identifiable, dynamically structured, and controllable?**

---

## 2. 总体设计原则

本框架不让 Transformer/Attention 从原始高维场中“自行发现全部物理规律”，而采用职责分离：

\[
\boxed{
\text{Encoder/JEPA}:\ \text{形成可预测的物理表示}
}
\]

\[
\boxed{
\text{Koopman}:\ \text{提供低阶、谱结构化的主要动力学骨架}
}
\]

\[
\boxed{
\text{Attention}:\ \text{只学习骨架无法解释的历史依赖与非线性耦合}
}
\]

\[
\boxed{
\text{PhysicsConstraint}:\ \text{定义并约束物理可容许状态与演化}
}
\]

\[
\boxed{
\text{Training/Selective decoder}:\ \text{训练时映射回物理空间施加约束，部署时按任务选择保留}
}
\]

因此，模型的基本哲学不是

\[
\text{large model}\rightarrow\text{physics emerges},
\]

而是

\[
\boxed{
\text{physical structure first}\rightarrow\text{learn unresolved complexity}
}
\]

---

## 3. 核心状态设计：PhysicsConstraint + `z_K` + `z_R`

### 3.1 不再把物理性定义成第三个 latent

设离散后的真实物理状态为

\[
U_t\in\mathcal X,
\]

但系统真实可访问的状态并不是整个任意欧氏空间，而是满足 governing equations、边界条件、constitutive relations、守恒/耗散和状态可容许性的集合：

\[
\boxed{
\mathcal M_{\rm phys}
=
\left\{
U:\;
\mathcal R_{\rm PDE}(U)=0,\;
\mathcal B(U)=0,\;
\mathcal C(U)=0,\;
\mathcal A(U)\ge 0
\right\}.
}
\]

因此本框架不再定义一个需要穷举或学习的

\[
z_t^{\rm phys}.
\]

物理信息被拆成两类对象：

1. **PhysicsConstraint**：定义哪些状态/演化是物理允许的，是核心约束；
2. **PhysicalProbe（可选）**：少量任务相关的可观测量，仅用于诊断、grounding、控制目标或消融，不是模型状态的一部分。

可选 probe 记为

\[
\boxed{
q_t=G_{\rm probe}(U_t),
}
\]

例如能量、升阻力、主频、局部压力传感器值等。`q_t` 不要求穷尽全部物理量，也不要求形成完备坐标；它只是可解释的观测接口。

### 3.2 这不是“枚举所有物理量”的问题

对于一个闭合的 PDE 系统，应首先选择一个足够描述演化的基本状态，例如可压缩反应流可采用

\[
U=[\rho,\rho\mathbf u,\rho E,\rho Y_1,\ldots,\rho Y_{N_s-1}],
\]

压力、温度、焓、涡量、Mach 数等大量派生物理量都可以写成

\[
q_i=G_i(U,\nabla U,\mu).
\]

因此不需要枚举它们。需要人工给定的是**状态定义与物理算子**，机器学习负责寻找适合描述其动力学的低维坐标。

这使问题成为

\[
\boxed{
\text{physics-defined admissible state space}
+\text{learned dynamical coordinates}
}
\]

而不是物理量枚举问题。

### 3.3 `z_K`：真正的 learned dynamical coordinates

定义可学习 lifting

\[
\boxed{
z_t^K=E_K(U_t^{\rm model},\mu),
\qquad
z_t^K\in\mathbb R^{d_K}.
}
\]

这里 `U_model` 是经过 normalization/preprocessing 后的网络输入；物理约束仍在 raw-unit physical space 中定义。

`E_K` 的目标不是简单压缩，而是寻找一个坐标图，使主要动力学尽量具有有限维 Koopman closure：

\[
\boxed{
E_K\big(\Phi_{\Delta t}(U_t,a_t)\big)
\approx
\mathcal K_{\Delta t}\big(E_K(U_t),a_t\big).
}
\]

连续时间写成

\[
\dot z_t^K=A(\mu)z_t^K+B(\mu)a_t,
\qquad
K_{\Delta t}=e^{A\Delta t}.
\]

从几何上，可以把

\[
E_K:\mathcal M_{\rm phys}\rightarrow\mathcal M_K
\]

理解为学习一张适合动力学传播的低维坐标图；理想情况下其分量接近 Koopman observables/eigenfunction-like coordinates，而不是任意压缩特征。

### 3.4 `z_R`：不是第二张平级流形，而是 closure/memory state

有限维 `z_K` 一般无法对复杂 PDE 完全闭合。即使 `E_K` 与 Koopman core 已训练完成，仍会存在

\[
\epsilon_{t+1}
=
E_K(U_{t+1})
-
\mathcal K_{\Delta t}(E_K(U_t),a_t).
\]

如果这个误差具有历史依赖，则定义

\[
\boxed{
z_t^R=M_\psi(\mathcal H_t),
}
\]

其中

\[
\mathcal H_t=
\{z_{t-H+1:t}^{K},a_{t-H+1:t},\Delta t_{t-H+1:t},\mu\},
\]

可选地加入少量 probe `q_hist`，但默认 MVP 不依赖 probe。

`z_R` 更接近一个用于**把非 Markov 的 reduced dynamics 扩展成近似 Markov 系统的隐藏记忆状态**：

\[
\boxed{
(z_t^K,z_t^R)\in\mathcal M_{\rm ext}.
}
\]

因此 `z_R` 不与 `z_K` 统计独立，也不追求与 `z_K` 形成另一张独立物理流形。它的存在完全依赖于 `z_K` 的有限维 closure error。

### 3.5 `z_R` 必须在 Koopman 完成后训练

先训练并冻结

\[
E_K,\quad \mathcal K,\quad D_{\rm train}.
\]

然后在**同一冻结 Koopman 坐标系**中构造 residual label：

\[
\boxed{
r_{t+1}
=
\operatorname{sg}\left[
E_K(U_{t+1}^{\rm model})
-
\mathcal K_{\Delta t}(E_K(U_t^{\rm model}),a_t)
\right].
}
\]

这可以离线预计算成 residual dataset：

```text
Input : z_K[t-H+1:t], action history, dt history, parameters
Label : r[t+1]
```

然后训练

\[
z_t^R=M_\psi(\mathcal H_t),
\qquad
\Delta z_{t+1}^K=W_Rz_t^R,
\]

使

\[
\boxed{
\mathcal L_R
=
\|g_t\Delta z_{t+1}^K-r_{t+1}\|_2^2.
}
\]

最终动力学为

\[
\boxed{
\hat z_{t+1}^K
=
\underbrace{\mathcal K_{\Delta t}(z_t^K,a_t)}_{\text{structured backbone}}
+
\underbrace{g_t\Delta z_{t+1}^K}_{\text{memory/closure correction}}.
}
\]

因此：

- `z_K` 与 `z_R` **训练阶段可以分开**；
- `z_K` 与 `z_R` **数学作用并不独立**；
- `z_R` 是在固定 `z_K` 表示下，为预测 residual 所学习出的 hidden representation。

### 3.6 物理方程如何真正进入训练

为了让物理约束直接作用于预测结果，训练阶段引入轻量 decoder

\[
\boxed{
\hat U_t^{\rm train}=D_{\rm train}(z_t^K,\mu).
}
\]

它首先提供重构约束

\[
\mathcal L_{rec}
=
\|\hat U_t^{\rm train}-U_t\|_W^2,
\]

防止 `E_K` 只保留“容易线性传播但无法区分真实状态”的变量；同时对预测未来状态

\[
\hat U_{t+1}^{\rm train}
=D_{\rm train}(\hat z_{t+1}^K)
\]

施加

\[
\boxed{
\mathcal L_{physics}
=
\lambda_R\|\mathcal R_{\rm PDE}(\hat U)\|^2
+
\lambda_B\|\mathcal B(\hat U)\|^2
+
\lambda_C\|\mathcal C(\hat U)\|^2
+
\lambda_A\mathcal P_{\rm admissible}(\hat U).
}
\]

如果存在精确/高效投影，也可使用 hard constraint：

\[
\hat U\leftarrow\Pi_{\mathcal M_{\rm phys}}(\tilde U).
\]

`D_train` 是**训练与验证工具**。后续若任务只做 latent control/MPC，可以丢弃它；若需要高分辨率全场 surrogate，再在 V1.x 增加独立高保真 decoder。

### 3.7 可选 PhysicalProbe 的角色

PhysicalProbe 不进入核心状态定义。它只在以下场景出现：

1. 诊断 `z_K` 是否保留任务相关物理信息；
2. 用作控制/MPC 的目标量；
3. 做跨模型可解释指标；
4. 作为 residual memory 的可选辅助输入进行消融。

定义

\[
q_t=G_{\rm probe}(U_t),
\qquad
\hat q_t=H_q(z_t^K,z_t^R).
\]

对应辅助损失

\[
\mathcal L_{probe}=\|\hat q_t-q_t\|^2.
\]

但 **MVP 不要求枚举或穷尽 `q_t`**，也不允许因为 probe 缺失而改变 `z_K + z_R` 的核心动力学定义。

### 3.8 最终代码语义

核心运行状态只保留：

```python
@dataclass
class LatentState:
    z_k: Tensor              # [B, d_k], instantaneous Koopman state
    z_r: Tensor | None       # [B, d_r], history-dependent closure memory
```

物理对象单独存在：

```python
class PhysicsConstraint: ...   # PDE / BC / conservation / admissibility
class PhysicalProbe: ...       # optional deterministic diagnostics q(U)
class TrainingDecoder(nn.Module): ...  # latent -> physical state for training constraints
```

禁止把 `q_phys`、probe 或其它物理诊断量重新包装成第三个必需 latent。

## 4. Koopman 谱骨架的推荐参数化

对于振荡、波传播、流体模态问题，推荐将连续时间生成元 \(A\) 参数化成稳定的旋转—伸缩块。

对第 \(i\) 个复共轭模态：

\[
A_i=
\begin{bmatrix}
-\alpha_i & -\omega_i\\
\omega_i & -\alpha_i
\end{bmatrix},
\qquad \alpha_i\ge0.
\]

则

\[
e^{A_i\Delta t}
=e^{-\alpha_i\Delta t}
\begin{bmatrix}
\cos(\omega_i\Delta t)&-\sin(\omega_i\Delta t)\\
\sin(\omega_i\Delta t)&\cos(\omega_i\Delta t)
\end{bmatrix}.
\]

这使 latent mode 具有直接动力学解释：

\[
\boxed{
\omega_i\leftrightarrow\text{frequency},\qquad
\alpha_i\leftrightarrow\text{decay/growth rate}
}
\]

若允许不稳定增长，可使用有界参数化 \(\alpha_i\in[-\alpha_{max},\alpha_{max}]\)，而不是完全无约束特征值。

对非振荡衰减模态可加入实数块

\[
\dot z_i=-\beta_i z_i,
\qquad \beta_i\ge0.
\]

参数依赖可写成

\[
A(\mu)=A_0+\sum_{k=1}^{d_\mu}\phi_k(\mu)A_k,
\]

或由一个小型 hypernetwork 输出谱参数 \(\{\alpha_i,\omega_i\}\)，避免直接输出无结构的大矩阵。

---

## 5. Attention residual dynamics：由 `z_K` 历史形成 `z_R`，只学习 closure

Attention 不负责从零学习完整演化算子，也不直接在原始物理场上做 full transition。它接收已经由 Koopman encoder 压缩的历史：

\[
\mathcal H_t=
\left\{
 z_{t-H+1:t}^{K},
 a_{t-H+1:t},
 \mu,
 \Delta t_{t-H+1:t}
\right\}.
\]

可选 PhysicalProbe 历史 `q_hist` 只作为消融输入，不是默认依赖。

定义

\[
\boxed{
z_t^R=M_\psi(\mathcal H_t)
}
\]

并将 `z_R` 映射到 Koopman latent 空间中的 closure correction：

\[
\Delta z_{t+1}^{K}=W_Rz_t^R.
\]

Koopman backbone 给出

\[
\tilde z_{t+1}^{K}
=\mathcal K_{\Delta t}(z_t^K,a_t;\mu).
\]

最终

\[
\boxed{
\hat z_{t+1}^{K}
=
\tilde z_{t+1}^{K}
+g_t\odot\Delta z_{t+1}^{K},
}
\]

其中

\[
g_t=\sigma(G_\eta(z_t^R))\in[0,1].
\]

### 5.1 residual target 必须来自冻结的同坐标 Koopman 表示

Residual warm-up 开始前冻结 `E_K` 与 Koopman core。定义

\[
\boxed{
r_{t+1}
=\operatorname{sg}\left[
E_K(U_{t+1})-\tilde z_{t+1}^K
\right].
}
\]

EMA teacher 不参与这个差分。

\[
\mathcal L_R
=\|g_t\Delta z_{t+1}^{K}-r_{t+1}\|_2^2.
\]

### 5.2 residual budget

为了防止 Attention 接管主动力学，使用

\[
\mathcal L_{budget}
=\|g_t\Delta z_{t+1}^K\|_2^2
\]

以及更稳定的相对诊断量

\[
\boxed{
C_{closure}
=\frac{\|g_t\Delta z_{t+1}^K\|}
{\|g_t\Delta z_{t+1}^K\|+\|\tilde z_{t+1}^K-z_t^K\|+\epsilon}
\in[0,1].
}
\]

稳定区域长期接近 1 说明 Koopman backbone 被架空；瞬态/切换附近上升则可能是有意义的 closure activity，需要通过实验验证。

### 5.3 数学解释

`z_R` 不是第二套瞬时编码，而是有限维投影后的 memory/closure variable。其目标与 Mori--Zwanzig 意义下的非 Markov memory term 接近：

\[
\text{reduced dynamics}
=\text{Markov backbone}+\text{memory closure}+\text{unresolved noise}.
\]

本项目首先用确定性 `z_R` 近似可预测的 memory closure；随机 unresolved component 留到 V2.x。

## 6. JEPA 训练外壳：只服务于 `z_K` 的 predictive representation

JEPA 不产生新的 latent 类型。它的作用是让 `E_K` 学到的表示不仅可重构、可被 Koopman 传播，而且对未来具有稳定 predictive semantics。

定义 online encoder

\[
z_t^K=E_\theta(U_t,\mu),
\]

以及 target encoder

\[
\boxed{
z_{t+k}^{K,tar}
=\operatorname{sg}[E_{\bar\theta}(U_{t+k},\mu)].
}
\]

Target encoder 使用 EMA：

\[
\bar\theta\leftarrow\tau\bar\theta+(1-\tau)\theta.
\]

Koopman 或 Koopman+closure predictor 给出 \(\hat z_{t+k}^{K}\)，JEPA loss 为

\[
\boxed{
\mathcal L_{JEPA}
=\sum_{k=1}^{H_J}\omega_k\,
d(\hat z_{t+k}^{K},z_{t+k}^{K,tar}).
}
\]

JEPA target 仅用于 representation alignment；**Residual target 必须在冻结阶段使用同一个 frozen online Koopman encoder 计算，不能把 EMA coordinate drift 当成 residual。**

因此：

- `z_K` 是 JEPA/Koopman 共享的 predictive state；
- `z_R` 没有独立 target encoder；
- PhysicsConstraint 不需要 target encoder；
- PhysicalProbe 只提供可选监督/诊断。

## 7. PhysicsConstraint 与可选 PhysicalProbe

### 7.1 强物理性来自方程和可容许状态，而不是 probe 枚举

定义统一物理约束接口

\[
\boxed{
\mathcal L_{physics}
=\mathcal L_{PDE}+\mathcal L_{BC}+\mathcal L_{conservation}+\mathcal L_{admissibility}+\mathcal L_{constitutive}.
}
\]

具体系统只实现适用的项。例如可压缩流至少需要考虑守恒形式、边界通量和 EOS consistency；不可压流可包含 divergence-free constraint。

PhysicsConstraint 应作用在 decoded raw-unit state 或其时空窗口上，而不是作用在一个人为枚举的 physical-latent vector 上。

### 7.2 PhysicalProbe 只是可选观测接口

对于任务确实关心的少量量，可以定义

\[
q_t=G_{probe}(U_t)
\]

以及可学习 readout

\[
\hat q_t=H_q(z_t^K,z_t^R).
\]

辅助损失

\[
\mathcal L_{probe}=\|\hat q_t-q_t\|_W^2.
\]

Probe 的选择应由任务和诊断需求决定，而不是试图穷尽所有物理变量。

### 7.3 Counterfactual identifiability

若有控制动作，需要避免不同动作在 latent 中被压成同一未来：

\[
\hat z_{t+1}^{(1)}=P(z_t,a_t^{(1)}),\qquad
\hat z_{t+1}^{(2)}=P(z_t,a_t^{(2)}).
\]

可以使用 action-held-out evaluation 或 margin/counterfactual loss，具体放在 action-conditioned 版本实现。

## 8. 两级 Decoder：训练约束 Decoder 与任务 Decoder

### 8.1 `D_train`：训练阶段的轻量物理解码器

为了避免 Koopman encoder 得到无法区分真实状态的任意 representation，并让 PDE/BC/守恒可以直接作用于预测，基础训练阶段需要

\[
\boxed{
\hat U_t^{train}=D_{train}(z_t^K,\mu).
}
\]

使用

\[
\mathcal L_{rec}=\|\hat U_t^{train}-U_t\|^2
\]

以及对未来预测 decoded state 的 `L_physics`。

`D_train` 不承担时间演化，只承担 latent 到物理状态的映射；它可以是低容量、低分辨率或仅在训练阶段存在。

### 8.2 高保真 field decoder：后续可选

若最终任务需要 CFD/FEM surrogate，再训练

\[
\hat U_t=D_{field}(z_t^K,z_t^R,\mu).
\]

它可以独立于 `D_train`，并在 V1.x 以后增加容量。

### 8.3 控制/规划任务

若只需要控制，可以部署

\[
(z_t^K,z_t^R)\rightarrow H_q\rightarrow q_t
\]

或直接输入 policy/MPC，而不部署 field decoder。因此“训练时需要物理映射”与“部署时必须输出完整物理场”是两个不同问题。

## 9. 完整损失函数：按职责分组，而不是一次性全部开启

最终联合训练时可写成

\[
\boxed{
\mathcal L_{total}
=
\lambda_K\mathcal L_K
+\lambda_M\mathcal L_{K,multi}
+\lambda_V\mathcal L_{var}
+\lambda_S\mathcal L_{spec}
+\lambda_J\mathcal L_{JEPA}
+\lambda_R\mathcal L_R
+\lambda_B\mathcal L_{budget}
+\lambda_{rec}\mathcal L_{rec}
+\lambda_P\mathcal L_{physics}
+\lambda_Q\mathcal L_{probe}
+\lambda_{cf}\mathcal L_{cf}
+\lambda_F\mathcal L_{field}.
}
\]

但**实现时禁止从第一个版本就同时打开全部 loss**。损失必须按版本和训练阶段逐步加入。

### 9.1 Koopman one-step consistency

\[
\mathcal L_K
=
\left\|
\operatorname{sg}(z_{t+1}^{K,tar})
-
\mathcal K_{\Delta t}(z_t^K,a_t)
\right\|_2^2.
\]

### 9.2 Koopman multi-step rollout

\[
\boxed{
\mathcal L_{K,multi}
=
\sum_{k=1}^{H_K}w_k
\left\|
\operatorname{sg}(z_{t+k}^{K,tar})
-
\hat z_{t+k}^{K,base}
\right\|_2^2.
}
\]

### 9.3 Koopman non-collapse

\[
\mathcal L_{var}
=
\frac1{d_K}\sum_j
\max(0,\sigma_{min}-\sigma_j(z^K))^2.
\]

可选 covariance penalty：

\[
\mathcal L_{cov}
=\|\operatorname{offdiag}(\operatorname{Cov}(z^K))\|_F^2.
\]

### 9.4 Spectral regularization

对于连续时间生成矩阵 `A`，根据具体系统使用软约束：

\[
\mathcal L_{spec}=\sum_i \phi(\lambda_i(A)).
\]

不应无条件强迫所有模态稳定；如果研究对象包含分岔和不稳定增长，应允许与物理一致的正增长率。

### 9.5 JEPA predictive alignment

\[
\mathcal L_{JEPA}
=
\sum_{k=1}^{H_J}
\omega_k
\,d\left(
\hat z_{t+k}^{K},
\operatorname{sg}(E_{\bar\theta}(U_{t+k}))
\right).
\]

第一版 `d(\cdot,\cdot)` 直接使用归一化 MSE 或 cosine distance；不要过早引入复杂对比学习。

### 9.6 Residual target loss

\[
r_{t+1}^{tar}
=
\operatorname{sg}\left[
z_{t+1}^{K,tar}-\tilde z_{t+1}^{K}
\right],
\]

\[
\boxed{
\mathcal L_R
=
\|\Delta z_{t+1}^{K}-r_{t+1}^{tar}\|_2^2.
}
\]

### 9.7 Residual budget / gate penalty

\[
\mathcal L_{budget}
=
\frac{\|g_t\Delta z_{t+1}^K\|_2^2}
{\|\tilde z_{t+1}^K\|_2^2+\epsilon}.
\]

必须持续记录

\[
\boxed{
R_{res}(t)
=\frac{\|g_t\Delta z_{t+1}^{K}\|_2}
{\|\tilde z_{t+1}^{K}\|_2+\epsilon}
}
\]

作为关键诊断量。

### 9.8 Reconstruction / state sufficiency

\[
\boxed{
\mathcal L_{rec}=\|D_{train}(z_t^K)-U_t\|_W^2.
}
\]

该项不是为了把模型退化成重构型 autoencoder，而是防止 `z_K` 只保留“容易线性传播但不足以区分真实物理状态”的信息。第一版可对降采样场或关键 state channels 使用低容量 decoder。

### 9.9 Optional PhysicalProbe loss

若任务需要少量可解释量：

\[
q_t=G_{probe}(U_t),\qquad
\hat q_t=H_q(z_t^K,z_t^R),
\]

\[
\boxed{
\mathcal L_{probe}=\|\hat q_t-q_t\|_2^2.
}
\]

该项默认可关闭；不存在 `z_phys` 与跨 latent covariance penalty。

### 9.10 Physics constraint

根据任务定义。例如有 decoder 时可以使用：

\[
\mathcal L_{mass}
=|M(\hat U_{t+1})-M(U_t)-\Delta M_{boundary}|^2,
\]

\[
\mathcal L_{energy}
=|E(\hat U_{t+1})-E_{expected}|^2.
\]

基础 Koopman representation 阶段至少保留一个 `D_train`，使物理约束能够落到 raw-unit state。若完整 PDE residual 计算昂贵，可先从 BC、守恒、EOS/admissibility 等廉价约束开始，再逐步增加 PDE residual。

### 9.11 Action counterfactual loss

对于同一初态下物理后果明显不同的两个动作 `a1, a2`，要求预测 latent 也能区分：

\[
\mathcal L_{cf}
=
\max\left(
0,
 m-
 \|\hat z_{t+1}^{K}(a^{(1)})-\hat z_{t+1}^{K}(a^{(2)})\|_2
\right).
\]

该项只在 action-conditioned 版本加入。

## 10. 推荐数据流

对于规则网格二维流体，batch 推荐：

```text
U_context_raw   : [B, H+1, C_u, Nx, Ny] or explicit transition-aligned contract
U_context_model : [B, H+1, C_u, Nx, Ny]
a_context       : [B, H, d_a]        # optional
U_future_raw    : [B, Kf, C_u, Nx, Ny]
U_future_model  : [B, Kf, C_u, Nx, Ny]
a_future        : [B, Kf, d_a]       # optional
mu              : [B, d_mu]          # optional
dt              : [B, H+Kf] or scalar
```

Koopman representation：

```text
z_k_context       : [B, H, d_k]
z_k_future_target : [B, Kf, d_k]
```

Residual closure：

```text
z_r_t       : [B, d_r]
delta_z_k   : [B, d_k]
gate        : [B, 1] or [B, d_k]
```

可选物理 probe：

```text
q_context : [B, H, d_q]   # optional; deterministic from raw state
q_future  : [B, Kf, d_q]  # metric/target only, never required for rollout
```

`z_R` 不应为每个 raw frame 独立编码；它由 `z_K` 历史窗口产生。第一版建议使用全局 latent vector，而不是大量 spatial tokens。

## 11. 推荐的最小可行模型（MVP）

最终研究 baseline 的最小结构是：

\[
\boxed{
E_K
+
D_{train}
+
\text{PhysicsConstraint}
+
\text{EMA JEPA target}
+
\text{continuous-time Koopman core}
+
\text{small residual memory/Attention}
}
\]

可选增加 `PhysicalProbe` 和 probe readout，但它们不属于核心状态。

工程开发仍必须按第三部分 V0.1--V1.0 逐步组装。

暂时不做：

- 大型高保真 field decoder；
- RL；
- mixture-of-Koopman experts；
- stochastic latent；
- 多尺度 spatial token hierarchy；
- 3D combustion/detonation。

V1.0 必须回答：

1. `z_K` 是否形成稳定、可传播的动力学坐标？
2. JEPA 是否改善 `z_K` 的 predictive representation，而不是只降低表面 loss？
3. `z_R` 是否只在 Koopman closure 失败处承担可预测 memory，而不是接管主动力学？
4. decoded prediction 是否满足预先定义的物理约束，并且这一结果不依赖穷举物理 probe？

# 第二部分：各模块选择原因、理论原理与代码实现说明（详细版）

## 12. 为什么不直接使用 Transformer 作为完整物理世界模型

Transformer 具有很强的通用函数逼近和长程依赖建模能力，其 attention 机制

\[
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt d}
\right)V
\]

非常适合寻找序列中不同 token 的相关性。但在物理系统中，“相关性建模”并不等于“动力学结构建模”。

物理系统通常额外具有：

- 守恒律；
- 对称性与不变性；
- 因果时间演化；
- 稳定/不稳定谱结构；
- 多尺度；
- 边界条件；
- 明确的控制输入；
- 长时间积分误差累积。

如果直接用 Transformer 学

\[
U_{t-H:t}\mapsto U_{t+1},
\]

模型必须同时从数据中重新学习：

1. 什么信息值得保留；
2. 什么是状态变量；
3. 哪些动力学近似线性；
4. 哪些模式具有主导频率；
5. 哪些模式会衰减/增长；
6. 哪些非线性耦合是关键；
7. 哪些关系只是高维观测空间中的冗余相关性。

这会产生较差的数据效率和可解释性，而且一步误差很容易在 autoregressive rollout 中放大。

因此本框架的观点是：

\[
\boxed{
\text{Attention 应是 residual reasoning mechanism，
而不是唯一的 dynamics prior。}
}
\]

---

## 13. 为什么使用 JEPA，而不是以重构为中心的 Autoencoder

传统 Autoencoder 目标：

\[
U\xrightarrow{E}z\xrightarrow{D}\hat U,
\qquad
\min\|U-\hat U\|^2.
\]

它优化的是“保留能够重构当前观测的信息”。但世界模型真正需要的是“保留能够预测未来、区分动作后果、支撑控制的信息”。两者并不等价。

例如流场中高频小尺度、网格噪声或局部纹理可能对 MSE 重构很重要，但对主模态演化和控制无关；反过来，一个能量很低但决定 bifurcation 的方向可能对重构 MSE 贡献很小，却对长期预测非常重要。

JEPA 将目标改为

\[
\boxed{
\text{predict target representation rather than reconstruct target observation}
}
\]

即

\[
E(U_t)\rightarrow\hat z_{t+1},
\qquad
\hat z_{t+1}\approx E_{target}(U_{t+1}).
\]

这为“任务相关压缩”提供了天然框架。

### 13.1 但为什么 JEPA 本身还不够

JEPA 可以避免显式像素/场重构，但 latent 仍可能只是“统计可预测”，并不保证：

- 相同 physical state 映射一致；
- 不同 physical state 可识别；
- 不同 action 产生可区分未来；
- latent 距离对应物理意义；
- 守恒量和稳定性在 latent 中被保留。

因此本框架必须同时使用 state-sufficiency/reconstruction、decoded PhysicsConstraint、Koopman dynamics constraint；进入控制版本后再加入 action counterfactual consistency。

---

## 14. 为什么选择 Koopman 作为主要动力学骨架

非线性系统

\[
x_{t+1}=F(x_t)
\]

对 observable \(g(x)\) 定义 Koopman operator

\[
\mathcal K g(x)=g(F(x)).
\]

即使 \(F\) 非线性，\(\mathcal K\) 对 observable 是线性的。若能找到有限维近似不变子空间

\[
z=g(x),
\]

则有

\[
z_{t+1}\approx Kz_t.
\]

它特别适合本框架的原因并不是“Koopman 能完美线性化所有物理”，而是它提供了几种非常有价值的 inductive bias：

### 14.1 谱结构

\[
\lambda_i=e^{(-\alpha_i+i\omega_i)\Delta t}
\]

直接对应衰减率和振荡频率，因此比一个任意 MLP transition 更容易检查 long-term stability。

### 14.2 多步传播简单

若无控制

\[
z_{t+k}=K^k z_t,
\]

或连续时间

\[
z(t+\tau)=e^{A\tau}z(t).
\]

这为 long-horizon rollout 提供显式结构，而不是每一步重新做完全非线性映射。

### 14.3 与降阶模型和控制天然兼容

Koopman latent 可以自然连接：

- DMD/EDMD；
- POD/ROM；
- LQR；
- MPC；
- system identification；
- spectral/modal analysis。

这对物理世界模型比“只预测更准”更有价值。

---

## 15. 为什么不能只用 Koopman

Koopman operator 在一般系统上是无限维的。有限维 \(K\) 只是近似，因此复杂 PDE、湍流、燃烧、冲击波、强 bifurcation 系统可能出现：

- closure error；
- continuous spectrum；
- 非平稳模态；
- latent manifold drift；
- spectral collapse；
- latent norm explosion；
- 模态切换无法由固定 \(K\) 表达。

因此不应把 Koopman 当作“物理真理层”，而应视为：

\[
\boxed{
\text{dominant structured dynamics prior}
}
\]

这正是 residual branch 存在的原因。

---

## 16. 为什么使用 Attention 学 residual，而不是直接学 full transition

定义真实 latent transition

\[
z_{t+1}=F_z(z_t,z_{t-1},\ldots,a_t,\mu).
\]

将其分解为

\[
\boxed{
F_z=\mathcal K+\mathcal R
}
\]

其中：

\[
\mathcal K\quad\text{捕获可谱化、主要、长期可传播的动力学},
\]

\[
\mathcal R\quad\text{捕获 history-dependent nonlinear residual}.
\]

因此

\[
\hat z_{t+1}=\mathcal K(z_t,a_t)+R_\psi(z_{t-H:t},a_{t-H:t}).
\]

Attention 在这里最有价值，因为 residual 往往恰好具有：

- 非局部 temporal dependency；
- 跨 mode coupling；
- intermittency；
- transition precursor；
- action-conditioned interaction。

### 16.1 residual gate 必须存在

若没有限制，训练会发现最简单的方法是让大 Transformer 直接学习所有 dynamics，并令 Koopman branch 退化。

因此推荐：

1. residual 最后一层零初始化；
2. gate 初始偏置设为负值，使 \(g\approx0\)；
3. 加 residual norm penalty；
4. 先训练 Koopman，再打开 residual；
5. 监控

\[
r_{res}
=
\frac{\|\Delta z\|}
{\|z^{base}_{t+1}\|+\varepsilon}.
\]

如果稳定周期阶段 \(r_{res}\) 仍长期很大，说明结构分工失败。

---

## 17. 为什么核心状态只保留 `z_K` 与 `z_R`

原先把 `z_phys`、`z_K`、`z_R` 并列的设计容易混淆三个不同层次：物理定律、动力学坐标和 memory state。v2.2 将其分离：

\[
\boxed{
\text{PhysicsConstraint}\quad\neq\quad\text{latent state}
}
\]

\[
\boxed{
z_K=\text{learned dynamical coordinates}
}
\]

\[
\boxed{
z_R=\text{closure/memory state conditioned on }z_K\text{ history}
}
\]

### 17.1 PhysicsConstraint：定义状态空间，不参与 latent 竞争

它回答：

> 什么样的状态和演化是物理允许的？

例如 PDE residual、boundary flux、divergence-free、EOS、质量/能量守恒、正密度/正温度等。它们直接约束 `D_train(z_K)` 或预测后的 decoded state，不需要形成一个 trainable latent branch。

### 17.2 `z_K`：通过可传播性、可重构性和物理一致性共同学习

\[
\boxed{
z_t^K=E_K(U_t),
\qquad
z_{t+1}^K\approx\mathcal K(z_t^K)
}
\]

其训练信号来自：

- one/multi-step Koopman consistency；
- reconstruction/state sufficiency；
- JEPA predictive target；
- non-collapse；
- spectral diagnostics；
- decoded physical constraints。

### 17.3 `z_R`：冻结 `z_K` 后，由 residual supervision 学出

\[
\boxed{
r_{t+1}=\operatorname{sg}[E_K(U_{t+1})-\mathcal K(E_K(U_t))]
}
\]

\[
\boxed{
z_t^R=M_\psi(z_{t-H:t}^K,a_{hist},dt_{hist}),
\quad
W_Rz_t^R\approx r_{t+1}.
}
\]

因此 `z_R` 没有人工定义的分量含义；它学习的是“为了预测 Koopman closure error，历史中还需要保留什么信息”。

### 17.4 几何解释

`E_K` 学习

\[
\mathcal M_{phys}\rightarrow\mathcal M_K.
\]

当 `z_K` 不是严格 Markov sufficient state 时，引入 `z_R` 得到扩展状态

\[
\boxed{
(z_K,z_R)\in\mathcal M_{ext},
}
\]

其目的不是建立第二张独立物理流形，而是用 memory coordinate 使 reduced dynamics 更接近 Markov closure。

## 18. Encoder 应该怎么选

Encoder 必须尊重输入数据拓扑。

### 18.1 第一版二维规则网格

推荐按实现难度排序：

1. Conv encoder；
2. FNO/UNO-style encoder；
3. patch/token encoder；
4. multi-scale operator encoder。

第一版若重点验证“Koopman + JEPA + residual Attention”，建议使用结构清楚的 CNN 或 FNO encoder，不要一开始把创新点分散在复杂 encoder 上。

可写：

\[
h_t=E_{spatial}(U_t),
\qquad
z_t=W_z\operatorname{Pool}(h_t).
\]

若要保留 spatial tokens，则

\[
Z_t=[z_t^{(1)},\ldots,z_t^{(M)}],
\]

但这应放到第二阶段。

### 18.2 非结构网格

后续可替换为：

- MeshGraphNet/GNN；
- graph neural operator；
- mesh transformer；
- coordinate-conditioned neural field。

世界模型上层接口保持

\[
E(U_t)\rightarrow z_t
\]

不变，因此 encoder 可独立替换。

---

## 19. Target encoder 为什么用 EMA

若 online encoder 和 target encoder 同时自由优化

\[
\hat z\approx z^{target},
\]

存在所有表示一起塌缩到常数的风险。

使用 stop-gradient target 与 EMA：

\[
\bar\theta\leftarrow\tau\bar\theta+(1-\tau)\theta
\]

使 target 表示变化更慢，为 online predictor 提供相对稳定的学习目标。

第一版建议：

```yaml
target_ema:
  tau_start: 0.99
  tau_end: 0.9999
  schedule: cosine
```

并同时监控每个 latent dimension 的 variance，避免非全局但局部的 collapse。

---

## 20. Koopman 模块的工程参数化

不推荐第一版直接学习完全自由矩阵

\[
K\in\mathbb R^{d_K\times d_K},
\]

因为：

- 特征值容易越过稳定边界；
- 难解释；
- 参数多；
- 训练时可能通过非正规矩阵产生巨大 transient amplification。

推荐 continuous-time generator + structured blocks。

### 20.1 API

```python
class KoopmanCore(nn.Module):
    def forward(
        self,
        z_k,        # [B, d_K]
        action,     # [B, d_a]
        dt,         # [B, 1] or scalar
        params=None # [B, d_mu]
    ) -> Tensor:   # [B, d_K]
        ...
```

### 20.2 第一版结构

```text
[d_K/2] oscillator blocks
A_i = [[-alpha_i, -omega_i],
       [ omega_i, -alpha_i]]
```

其中

```text
alpha_i = softplus(raw_alpha_i)
omega_i = omega_max * sigmoid(raw_omega_i)
```

然后直接用解析形式计算 \(e^{A_i\Delta t}\)，无需每次调用通用 `matrix_exp`。

### 20.3 参数条件化

在多 Reynolds 数/多工况下，可令

```text
(raw_alpha, raw_omega) = spectral_hypernet(mu)
```

但 hypernetwork 要小，避免它自己变成另一套黑箱 dynamics。

---

## 21. Action-conditioned Koopman 设计

对于主动控制，推荐逐步增加复杂度。

### Level 0：无控制

\[
z' = Kz.
\]

### Level 1：线性控制

\[
z'=Kz+B a.
\]

### Level 2：双线性控制

\[
z'=Kz+Ba+\sum_j a_jN_jz.
\]

### Level 3：Koopman + residual action attention

\[
z'=
Kz+Ba+\sum_ja_jN_jz
+R_\psi(z_{hist},a_{hist}).
\]

推荐只有在 Level 1 无法表达 action response 时才进入 Level 2。

---

## 22. Attention / ResidualMemory 应该输入什么

ResidualMemory 的输入不应是 raw CFD field，也不应重复做一个大型 spatial encoder。默认输入只包含 `z_K` 历史、action、dt 和参数；PhysicalProbe 仅作为可选消融输入。

推荐历史 token：

\[
\boxed{
\xi_i
=P_\xi\left(
[z_i^K,a_i,\mu,\Delta t_i]
\right)
}
\]

组成

\[
X_t=[\xi_{t-H+1},\ldots,\xi_t].
\]

然后

\[
\boxed{
z_t^R=\operatorname{Transformer}_{small}(X_t)_{last}.
}
\]

第一版只使用 temporal tokens，不使用 spatial attention。

原因：

1. spatial compression 已由 `E_K` 完成；
2. residual branch 的科学问题是 closure/memory，不是重新学习空间表征；
3. 限制 residual 容量可以降低其接管 Koopman backbone 的风险；
4. temporal attention 权重和 gate 更容易作为诊断量分析。

第一版建议：

```text
history H : 8-16
d_model   : 64-128
heads     : 4
layers    : 2-3
d_R       : 32-64
```

后续如果证明确实存在强空间局部 closure，再进入 spatial Koopman tokens。

## 23. Multi-step rollout 为什么必须从第一版就考虑

单步 loss

\[
\|\hat z_{t+1}-z_{t+1}\|
\]

不能保证 rollout 稳定，因为推理时模型输入的是自己的预测：

\[
\hat z_{t+1}\rightarrow\hat z_{t+2}\rightarrow\cdots.
\]

训练中需要逐渐加入 open-loop rollout：

\[
\hat z_{t+k+1}=P(\hat z_{t+k},a_{t+k}).
\]

并优化

\[
\mathcal L_{multi}
=
\sum_{k=1}^{K_f}w_kd(\hat z_{t+k},z_{t+k}^{target}).
\]

建议 curriculum：

```text
Kf = 1 -> 2 -> 4 -> 8 -> 16
```

而不是第一天就训练 100 步 rollout。

---

## 24. PhysicalProbe 如何选择（可选，不属于核心状态）

不要试图穷尽所有物理变量。PhysicalProbe 只用于诊断、控制目标或额外可解释监督；强物理性由 PhysicsConstraint 保证。

选择原则：

\[
\boxed{
\text{少量、稳定、物理关键、与任务有关}
}
\]

对不可压二维尾流可选：

- total kinetic energy；
- enstrophy；
- lift coefficient \(C_L\)；
- drag coefficient \(C_D\)；
- dominant shedding frequency；
- circulation；
- selected pressure probes。

对可压缩流/燃烧可选：

- total mass；
- total energy；
- integrated heat release；
- chamber pressure；
- dominant frequency；
- wave number / mode indicator；
- shock/detonation front position；
- outlet thrust/mass flow。

这样 \(z\) 必须保留“会影响动力学和任务”的物理信息。

---

## 25. PhysicsConstraint 放在什么位置

物理约束应优先作用于**物理空间预测**，而不是通过枚举 latent observable 间接实现。

### 层 A：状态可容许性与 constitutive consistency

例如正密度、正温度、质量分数 simplex、EOS consistency：

\[
\mathcal L_{adm}+\mathcal L_{EOS}.
\]

### 层 B：守恒与边界条件

\[
\mathcal L_{cons}+\mathcal L_{BC}.
\]

这类约束通常比完整 PDE residual 更便宜，适合首先加入。

### 层 C：PDE residual / discrete operator consistency

对 decoded prediction

\[
\hat U=D_{train}(\hat z_K)
\]

计算

\[
\mathcal R_{PDE}(\hat U,a,\mu,\Delta t).
\]

如果原数值离散算子可调用，优先比较离散更新/flux consistency，而不是强行使用连续自动微分 PINN residual。

### 层 D：可选 PhysicalProbe readout

\[
H_q(z_K,z_R)\approx q(U)
\]

只用于 interpretability、控制目标和诊断，不承担“保证物理性”的主要责任。

## 26. Selective decoder 的设计原则

### 完整场预测

推荐 decoder 只承担

\[
z_t\rightarrow\hat U_t
\]

而不要把 future dynamics 也塞进 decoder。

二维网格可用：

- transpose-conv decoder；
- FNO decoder；
- coordinate-conditioned neural field。

### 控制

优先用 physical probe：

\[
z_t\rightarrow q_t
\]

而不是

\[
z_t\rightarrow U_t\rightarrow q_t.
\]

后者增加不必要误差路径。

因此代码建议把：

```text
WorldModel.transition()
WorldModel.decode_field()
WorldModel.read_probe()  # optional
```

写成三个独立接口。

---

## 27. 推荐训练流程：先学 Koopman 坐标，再冻结生成 residual dataset

核心原则：

\[
\boxed{
\text{physics contracts}
\rightarrow
\text{Koopman representation}
\rightarrow
\text{JEPA refinement}
\rightarrow
\text{freeze + residual dataset}
\rightarrow
\text{closure memory}
\rightarrow
\text{controlled joint fine-tune}
}
\]

### Stage 0：数据与 PhysicsConstraint

完成：

1. trajectory-level train/val/test；
2. window sampler 与 time/action/dt alignment；
3. raw/model normalization；
4. `PhysicsConstraint` / `ProblemSpec`；
5. 可选 `PhysicalProbe`；
6. Persistence、DMD/POD 等 baseline。

### Stage 1：直接验证 KoopmanCore

使用 oscillator / Duffing / Lorenz 等已知低维状态，只测试 `A/K`、`matrix_exp`、irregular dt、spectrum、multi-step rollout 和 checkpoint。

### Stage 2：训练 `E_K + D_train + KoopmanCore`

关闭 residual。训练

\[
\mathcal L_{stage2}
=
\lambda_K\mathcal L_K
+\lambda_M\mathcal L_{K,multi}
+\lambda_{rec}\mathcal L_{rec}
+\lambda_P\mathcal L_{physics}
+\lambda_V\mathcal L_{var}
+\lambda_S\mathcal L_{spec}.
\]

`D_train` 只负责 latent 到物理空间的映射，不承担未来 dynamics。

目标：先证明 `z_K` 本身是可传播且保留足够物理状态的信息。

### Stage 3：引入 JEPA target

建立 online / EMA target encoder。Residual 仍关闭。使用 JEPA predictive alignment 进一步组织 `z_K`，但 Koopman dynamics target 与 residual target 的“同坐标”要求保持不变。

### Stage 4：冻结结构分支并生成 residual dataset

完成 Stage 3 后：

```text
freeze E_K
freeze KoopmanCore
freeze D_train
hard-sync/pause EMA for residual stage
```

对训练集离线计算

\[
r_{t+1}=E_K(U_{t+1})-\mathcal K(E_K(U_t),a_t).
\]

缓存：

```text
z_k_history
action_history
dt_history
mu
residual_target
trajectory_id / time_index
```

Residual dataset 必须带 `encoder_hash + koopman_hash`；若结构分支权重变化，旧 residual cache 自动失效。

先用 tiny MLP/GRU 验证这个 supervised closure target 是否可学。

### Stage 5：Attention closure / `z_R`

替换 tiny baseline：

\[
z_t^R=M_\psi(z_{t-H:t}^K,a_{hist},dt_{hist},\mu),
\]

\[
\Delta z_{t+1}=W_Rz_t^R.
\]

只优化 residual branch。加入 gate 与 residual budget；默认不依赖任何 PhysicalProbe。

### Stage 6：Closed-loop multi-step closure

从单步 residual supervision 过渡到

\[
\hat z_{t+1}\rightarrow\hat z_{t+2}\rightarrow\cdots
\]

未来历史只压入模型自己的 `z_k_next`、给定 future action/dt/parameters；不得重新编码未来真实 `U` 作为 transition 输入。

### Stage 7：Controlled joint fine-tune

只有前面阶段稳定后才允许小学习率解冻 `E_K/KoopmanCore`。此时离线 residual cache 不再作为唯一 target；必须在线重新计算 detached residual target，避免 encoder 坐标变化后标签过期。

学习率满足

```text
LR_KoopmanEncoder << LR_ResidualAttention
LR_KoopmanCore    << LR_ResidualAttention
LR_TargetEncoder = 0  # EMA only
```

### Stage 8：任务接口

- 控制/规划：`H_q(z_K,z_R)` 或直接 latent MPC/policy；
- 高保真全场：额外训练 `D_field`；
- action-conditioned Koopman：最后加入 `B a` / bilinear terms / counterfactual test。

## 28. 代码目录建议：按职责与版本边界模块化

```text
project/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data/
│   │   ├── oscillator.yaml
│   │   └── pde2d.yaml
│   ├── model/
│   │   ├── koopman_only.yaml
│   │   ├── jepa_koopman.yaml
│   │   └── full_closure.yaml
│   └── train/
│       ├── stage_koopman.yaml
│       ├── stage_jepa.yaml
│       ├── stage_residual.yaml
│       └── stage_joint.yaml
├── src/
│   ├── data/
│   │   ├── datasets.py
│   │   ├── windows.py
│   │   ├── splits.py
│   │   └── normalization.py
│   ├── physics/
│   │   ├── constraints.py
│   │   ├── probes.py
│   │   ├── operators.py
│   │   └── invariants.py
│   ├── config/
│   │   ├── schema.py             # strict structured config + validation
│   │   └── hashing.py            # resolved config hash
│   ├── models/
│   │   ├── types.py              # ProblemBatch / ProblemSpec / latent outputs dataclasses
│   │   ├── koopman_encoder.py
│   │   ├── koopman_core.py
│   │   ├── target_encoder.py
│   │   ├── residual_memory.py
│   │   ├── residual_head.py
│   │   ├── gate.py
│   │   ├── training_decoder.py
│   │   ├── probe_readout.py
│   │   ├── field_decoder.py
│   │   └── world_model.py
│   ├── losses/
│   │   ├── koopman.py
│   │   ├── jepa.py
│   │   ├── collapse.py
│   │   ├── residual.py
│   │   ├── reconstruction.py
│   │   ├── probe.py
│   │   ├── physics.py
│   │   └── counterfactual.py
│   ├── rollout/
│   │   ├── koopman_rollout.py
│   │   ├── latent_rollout.py
│   │   └── mpc.py
│   ├── metrics/
│   │   ├── forecast.py
│   │   ├── spectral.py
│   │   ├── residual_burden.py
│   │   └── physical.py
│   ├── train/
│   │   ├── stages.py             # TrainStage + configure_trainable
│   │   ├── common.py
│   │   ├── train_koopman.py
│   │   ├── train_jepa.py
│   │   ├── train_residual.py
│   │   └── train_joint.py
│   └── utils/
│       ├── checkpoint.py
│       ├── seed.py
│       └── logging.py
├── scripts/
│   ├── generate_oscillator.py
│   ├── generate_advection_diffusion_2d.py
│   ├── train.py
│   ├── evaluate_rollout.py
│   ├── analyze_spectrum.py
│   └── visualize_latent.py
└── tests/
    ├── test_windows.py
    ├── test_physics_constraints.py
    ├── test_physical_probe_optional.py
    ├── test_koopman_core.py
    ├── test_matrix_exp_grad.py
    ├── test_ema.py
    ├── test_residual_stopgrad.py
    ├── test_world_model_shapes.py
    ├── test_checkpoint_roundtrip.py
    ├── test_problem_batch_alignment.py
    ├── test_optimizer_gradient_ownership.py
    ├── test_target_hard_sync_and_ema_order.py
    ├── test_same_coordinate_residual_target.py
    ├── test_causal_residual_memory.py
    ├── test_no_future_leakage.py
    ├── test_residual_cache_fingerprint.py
    └── test_closed_loop_rollout_history_update.py
```

设计要求：

1. `physics/` 不依赖神经网络，且 `PhysicalProbe` 不属于 latent state；
2. `koopman_core.py` 不依赖 JEPA 或 Transformer；
3. `residual_memory.py` 不允许内部直接调用 Koopman core；
4. `world_model.py` 只负责编排，不堆放具体数学实现；
5. 每个版本新增模块时，不允许无理由重写已通过测试的旧模块；
6. 所有模块必须可以单独实例化和单元测试。

## 29. 推荐接口契约

本节属于公共 API 规范。数学文中写 `z^K,z^R`，Python API 一律使用 `z_k,z_r`。

### 29.1 PhysicsConstraint

```python
class PhysicsConstraint(Protocol):
    def loss(self, pred_state_raw, *, prev_state_raw=None,
             action=None, dt=None, spec=None, metadata=None) -> dict[str, Tensor]:
        """Return named physical losses in raw physical units."""
        ...
```

可以组合多个 constraint；禁止在 constraint 内部重新训练网络。

### 29.2 PhysicalProbe（可选）

```python
class PhysicalProbe(Protocol):
    def compute(self, state_raw, spec, metadata=None) -> Tensor:
        """Deterministic diagnostic q(U); never part of required latent state."""
        ...
```

### 29.3 KoopmanEncoder

```python
class KoopmanEncoder(nn.Module):
    def forward(self, state_model, mu_static=None) -> Tensor:
        """Normalized state -> z_k [B,d_k]."""
        ...
```

### 29.4 TrainingDecoder

```python
class TrainingDecoder(nn.Module):
    def forward(self, z_k, mu_static=None) -> Tensor:
        """z_k -> normalized/model-space state; caller converts to raw units for physics loss."""
        ...
```

### 29.5 KoopmanCore

```python
class KoopmanCore(nn.Module):
    def step(self, z_k, action=None, dt=None, mu_static=None):
        ...

    def rollout(self, z_k0, future_actions=None, future_dts=None,
                horizon=None, mu_static=None):
        ...

    @torch.no_grad()
    def spectrum(self):
        ...
```

### 29.6 ResidualMemory

```python
class ResidualMemory(nn.Module):
    def forward(
        self,
        z_k_hist,
        history_actions=None,
        history_dts=None,
        current_action=None,
        current_dt=None,
        mu_static=None,
        probe_hist=None,  # optional ablation only
    ) -> Tensor:
        """Causal z_k history -> closure memory z_r."""
        ...
```

### 29.7 ResidualHead

```python
class ResidualHead(nn.Module):
    def forward(self, z_r):
        """z_r -> delta_z_k [B,d_k], gate [B,1]."""
        ...
```

### 29.8 StructuredPhysicalJEPA

```python
class StructuredPhysicalJEPA(nn.Module):
    def encode_koopman(self, state_model, mu_static=None): ...
    def base_step(self, z_k, action, dt, mu_static=None): ...

    def transition(
        self, *, z_k_hist,
        history_actions, history_dts,
        current_action, current_dt,
        mu_static=None,
        probe_hist=None,
    ): ...

    def rollout_closed_loop(self, batch, horizon=None):
        """Transition loop never accesses future states."""
        ...
```

```python
@dataclass
class TransitionOutput:
    z_k_base: Tensor
    z_r: Tensor | None
    delta_z_k: Tensor
    gate: Tensor
    z_k_next: Tensor
```

### 29.9 TrainStage

```python
class TrainStage(Enum):
    KOOPMAN = "koopman"
    JEPA = "jepa"
    RESIDUAL = "residual"
    JOINT = "joint"
```

冻结/解冻只能由统一状态机产生。

## 30. 分阶段训练伪代码

### 30.1 Koopman representation + training decoder

```python
z_k_t = online_encoder(U_t_model)
z_k_next = online_encoder(U_next_model)

z_k_base = koopman_core.step(z_k_t, current_action, current_dt, mu_static)

U_t_hat_model = training_decoder(z_k_t, mu_static)
U_next_hat_model = training_decoder(z_k_base, mu_static)
U_t_hat_raw = normalizer.inverse(U_t_hat_model)
U_next_hat_raw = normalizer.inverse(U_next_hat_model)

physics_terms = physics_constraints.loss(
    U_next_hat_raw,
    prev_state_raw=U_t_raw,
    action=current_action,
    dt=current_dt,
    spec=problem_spec,
)

loss = (
    lambda_k * mse(z_k_base, z_k_next)
    + lambda_rec * reconstruction_loss(U_t_hat_model, U_t_model)
    + lambda_phys * sum(physics_terms.values())
    + lambda_var * variance_loss(z_k_t)
)
```

### 30.2 JEPA + Koopman

```python
z_k_ctx = online_encoder(U_context_model)
z_k_future_online = online_encoder(U_future_model)

with torch.no_grad():
    z_k_future_jepa = target_encoder(U_future_model)

z_k_pred = koopman_core.rollout(...)

loss = (
    lambda_j * jepa_loss(z_k_pred, z_k_future_jepa)
    + lambda_k * koopman_consistency(z_k_pred, z_k_future_online)
    + lambda_rec * reconstruction_loss(...)
    + lambda_phys * physics_loss_on_decoded_prediction(...)
    + lambda_var * variance_loss(z_k_ctx)
)

optimizer.step()
update_ema_after_optimizer_step(target_encoder, online_encoder)
```

### 30.3 冻结后生成 residual dataset

```python
freeze(online_encoder)
freeze(koopman_core)
freeze(training_decoder)
pause_ema()

with torch.no_grad():
    z_k_hist = online_encoder(U_context_model)
    z_k_next = online_encoder(U_next_model)
    z_k_base = koopman_core.step(
        z_k_hist[:, -1], current_action, current_dt, mu_static
    )
    residual_target = z_k_next - z_k_base

cache.write(
    z_k_hist=z_k_hist,
    actions=history_actions,
    dts=history_dts,
    mu=mu_static,
    residual_target=residual_target,
    encoder_hash=hash_module(online_encoder),
    koopman_hash=hash_module(koopman_core),
)
```

### 30.4 Residual closure training

```python
z_r = residual_memory(
    z_k_hist,
    history_actions,
    history_dts,
    current_action,
    current_dt,
    mu_static,
)
delta_z_k, gate = residual_head(z_r)
correction = gate * delta_z_k

loss = (
    mse(correction, residual_target)
    + lambda_budget * closure_budget(correction, z_k_hist[:, -1], z_k_base)
)
```

### 30.5 Closed-loop rollout

```python
z_k_hist = online_encoder(batch.context_states_model)
predictions = []

for k in range(horizon):
    out = model.transition(
        z_k_hist=z_k_hist,
        history_actions=history_actions_for_current_queue,
        history_dts=history_dts_for_current_queue,
        current_action=batch.future_actions[:, k] if batch.future_actions is not None else None,
        current_dt=batch.future_dts[:, k],
        mu_static=batch.mu_static,
    )
    predictions.append(out.z_k_next)
    z_k_hist = append_and_crop(z_k_hist, out.z_k_next)
    update_action_dt_history(...)
```

`batch.future_states_*` 只能在 rollout 完成后用于 loss/metric，不能进入 transition history。

### 30.6 Joint fine-tune

如果 `E_K/KoopmanCore` 解冻，旧 residual cache 立即失效。Joint 阶段必须在线计算 detached residual target：

```python
with torch.no_grad():
    z_next_target = online_encoder(U_next_model)
    z_base_detached = koopman_core.step(...)
    residual_target = z_next_target - z_base_detached
```

同时保留 JEPA、reconstruction、physics constraints 和 closure budget；Koopman LR 远小于 residual LR。

## 31. 必做消融实验

为了证明贡献来自“结构化动力学 + 物理约束 + closure memory”，而不是单纯参数更多，至少比较：

| ID | Representation | Dynamics | Closure | PhysicsConstraint | Probe |
|---|---|---|---|---|---|
| B0 | AE | MLP | No | No | No |
| B1 | AE | Transformer full transition | Full | No | No |
| B2 | JEPA | Transformer full transition | Full | No | No |
| B3 | Koopman AE | Koopman | No | No/weak | No |
| B4 | JEPA + Koopman | Koopman | No | Yes | No |
| B5 | JEPA + Koopman | Koopman | Attention residual | Yes | No |
| B6 | B5 | Koopman | Attention residual | Yes | Optional probe input/readout |

这里 B5 是核心模型；B6 用于判断少量 physical probe 是否真的提供额外价值，不能把 probe 当成模型成立的必要条件。

正式报告至少包括

\[
\boxed{
\text{short error}
+\text{long rollout}
+\text{spectrum}
+\text{physics-constraint violation}
+\text{closure burden}
}
\]

控制阶段再增加 counterfactual/control metrics。

## 32. 评价指标

### 32.1 场预测

\[
\epsilon_U(k)
=
\frac{\|\hat U_{t+k}-U_{t+k}\|_2}{\|U_{t+k}\|_2}.
\]

### 32.2 latent rollout

\[
\epsilon_z(k)
=\|\hat z_{t+k}-z_{t+k}^{tar}\|.
\]

### 32.3 谱误差

\[
\epsilon_f
=\frac{|\hat f-f|}{|f|+\epsilon}.
\]

并比较 learned Koopman spectrum 与 FFT/DMD/POD/linear stability result。

### 32.4 物理缺陷

守恒误差：

\[
\epsilon_I(k)
=|I(\hat U_{t+k})-I(U_{t+k})|.
\]

### 32.5 residual burden

\[
R_{burden}
=\mathbb E\left[
\frac{\|\Delta z_t\|}{\|z_t^{base}\|+\epsilon}
\right].
\]

这是本框架非常重要的诊断指标。

### 32.6 控制相关

- MPC success rate；
- accumulated cost；
- energy/control effort；
- constraint violation；
- model-predicted vs real closed-loop trajectory error。

---

## 33. 最适合的实验路线

### Phase A：低维单元测试

先验证代码正确性：

1. damped harmonic oscillator；
2. Van der Pol oscillator；
3. Lorenz-63（用于检查有限 Koopman + residual 的必要性）。

目标不是论文结果，而是检查：

- Koopman block 频率是否正确；
- residual 是否在非线性增强时上升；
- rollout 是否数值稳定。

### Phase B：二维流体主验证

优先推荐二维圆柱尾流，多 Reynolds 数 + 可选吹吸控制。

原因：

- 主导 vortex shedding frequency 清楚；
- 有成熟 POD/DMD baseline；
- Koopman 谱结构直观；
- 从稳态到极限环的瞬态可测试；
- 可构造 action-conditioned control；
- 数据生成成本远低于燃烧/爆轰。

### Phase C：更强非线性 PDE

例如：

- Kuramoto–Sivashinsky；
- 2D Navier–Stokes 高 Re；
- compressible wake；
- shock-containing flow。

测试固定有限维 Koopman 的 closure limit。

### Phase D：燃烧/爆轰模态

最终可进入：

- 多模态 RDE；
- mode switching；
- frequency detuning；
- wave number transitions；
- control-conditioned modal selection。

这一阶段最有科学意义，但不适合作为软件架构的首个 debugging case。

---

## 34. 当前研究趋势对本框架的启示（截至 2026-08）

### 34.1 JEPA 正从“语义 representation”走向“action-conditioned world state”

V-JEPA 2 已经展示：先从大规模视频学习 representation，再以少量机器人交互数据训练 action-conditioned latent world model，可以用于规划。这支持“预测 latent 而非重构全部观测”的路线。

### 34.2 研究焦点正在转向 latent 是否真正具有物理意义

2026 年的 Phys-JEPA 将 known physical variables 和 physics consistency 直接放入 latent state/transition，而非只在 decoded output 上约束；PhyLatent 更进一步指出：仅防止 JEPA 全局 collapse，并不等于 latent 保留 physical state identity 和 action consequences。

这支持本框架把 physical-state identity 问题显式拆成：`D_train` 的 state sufficiency、decoded PhysicsConstraint、可选 PhysicalProbe diagnostics，以及 action-conditioned 版本的 counterfactual consistency。

### 34.3 Koopman 世界模型正在强调谱稳定性和 long-horizon imagination

Koopman Dreamer 将 spectrally constrained Koopman latent backbone 与 action terms、local correction 和 multi-step/open-loop objectives 结合，核心问题就是减少 latent imagination 长时间滚动中的误差累积。

这支持本框架不使用无约束自由 \(K\)，而使用显式谱参数化。

### 34.4 “物理约束应进入 latent dynamics”正在成为独立方向

Physics-conforming Latent Twins 强调守恒量、不变量、admissibility 和 dissipative structure 应直接通过 latent flow map 满足，并讨论原空间约束如何转移到 latent space。

因此本框架的 physics loss 不应只放在 decoder 输出端。

### 34.5 仅靠 Transformer + Koopman 的组合已不够构成贡献

Koopman embedding + Transformer 早在 2020–2022 年即已有工作，因此真正创新点不能描述为“我们第一次组合 Koopman 和 Transformer”。

本框架应把贡献集中在：

\[
\boxed{
\text{JEPA predictive representation}
+\text{physics-constrained state sufficiency}
+\text{spectral Koopman backbone}
+\text{bounded residual attention closure}
}
\]

也就是**latent state design principle**，而不是模块拼接。

---

## 35. 当前主要痛点与对应设计

| 当前痛点 | 后果 | 本框架对应机制 |
|---|---|---|
| latent collapse / partial collapse | 表示无信息 | EMA target + variance monitoring |
| state insufficiency | `z_K` 易传播但丢失真实状态信息 | `D_train` reconstruction/state discrimination |
| physical inadmissibility | decoded rollout 违反 PDE/BC/守恒 | PhysicsConstraint / hard projection where possible |
| counterfactual collapse | 不同 action 未来 latent 过近 | action-conditioned Koopman + counterfactual tests |
| reconstruction bias | latent 过度保存细节 | 低容量 `D_train` + JEPA predictive objective |
| finite Koopman closure | 固定有限维 `K` 无法解释复杂瞬态 | frozen-base residual dataset + `z_R` memory closure |
| spectral instability | rollout 爆炸/坍缩 | structured generator / bounded spectrum |
| long-horizon error accumulation | 单步准、长期错 | multi-step closed-loop curriculum |
| attention takeover | Koopman 变成装饰 | staged freeze + gated residual + residual budget |
| residual labels stale | joint fine-tune 后 offline cache 坐标失效 | encoder/core hash guard + online detached targets in joint stage |
| generalization across parameters | 新 Re/Ma/geometry 失效 | parameter-conditioned spectral core |
| full-field decoding expensive | 控制效率差 | training decoder 与 deployment field decoder 分离 |
| partial observation | 观测不构成 Markov state | history/memory extension，后续观测模型 |

## 36. 第一版超参数建议

仅作为启动值，不是理论常数：

```yaml
latent:
  d_koopman: 64
  d_residual: 32

context:
  history: 16
  future_horizon_start: 1
  future_horizon_max: 16

koopman:
  continuous_time: true
  oscillator_blocks: 32
  parameter_conditioned: false
  bilinear_action: false

training_decoder:
  enabled: true
  low_capacity: true

physical_probe:
  enabled: false       # core model does not depend on probes
  use_as_closure_input: false

attention:
  d_model: 128
  layers: 2
  heads: 4
  dropout: 0.05
  residual_gate_init_bias: -3.0

jepa:
  ema_tau_start: 0.99
  ema_tau_end: 0.9999

loss:
  lambda_koopman: 1.0
  lambda_multistep: 0.5
  lambda_reconstruction: 0.2
  lambda_physics: 0.1       # tune by gradient scale; V0.5 onward
  lambda_jepa: 1.0          # V0.6 onward
  lambda_residual: 1.0      # V0.7 onward
  lambda_residual_budget: 1.0e-3
  lambda_probe: 0.0         # optional ablation
  lambda_counterfactual: 0.0
```

实际权重必须通过 gradient-scale、validation rollout 和 physical violation 共同调节。

## 37. 必须记录的训练诊断

每个 epoch/validation 至少记录：

```text
loss/koopman
loss/multistep
loss/reconstruction
loss/physics_total
loss/jepa                 # when enabled
loss/residual             # when enabled
loss/residual_budget      # when enabled
loss/probe                # optional

latent/z_k_variance_min
latent/z_k_variance_mean
latent/z_k_cov_condition

koopman/alpha_min_mean_max
koopman/omega_min_mean_max
koopman/spectral_radius_discrete

residual/correction_norm
residual/closure_fraction_mean
residual/gate_mean
residual/gate_p95

forecast/base_error_1_4_8_16
forecast/full_error_1_4_8_16

physics/constraint_total
physics/conservation_error
physics/bc_error
physics/admissibility_error
physics/probe_error          # optional
```

否则很容易得到一个预测 MSE 看起来很好、但动力学分工或物理约束已经失效的模型。

## 38. 关键失败模式与调试顺序

### 失败 A：`z_K` variance 接近 0

说明 representation collapse。优先检查：EMA/stop-gradient、variance loss、decoder 是否失去 state discrimination、normalization。

### 失败 B：`z_K` 可线性传播但 reconstruction 很差

说明 encoder 找到了容易传播但不 sufficient 的退化坐标。提高 state-sufficiency 约束，检查 `D_train` 容量与数据覆盖；不要先增加 Attention。

### 失败 C：decoded state reconstruction 好但 PhysicsConstraint 很差

检查 raw/model inverse transform、BC/flux 离散、constraint 实现和单位。若约束本身无误，再调 physics loss 或采用硬投影。

### 失败 D：Koopman-only 很差，Attention 打开后突然很好

可能 Attention 完全接管。检查 `C_closure`、gate、base error；必要时回退到 V0.7 tiny closure，重新判断 residual 是否真的可预测。

### 失败 E：一步预测很好，16 步后爆炸

检查谱半径、non-normal transient、residual feedback、closed-loop curriculum 和 teacher-forcing 泄漏。

### 失败 F：V0.7 residual dataset 可学，joint fine-tune 后突然失效

优先检查离线 cache 是否仍对应当前 encoder/core。Joint 阶段禁止继续把旧 cache 当作唯一标签，应在线重算 detached residual target。

### 失败 G：稳定态好，模态切换错

可能说明固定有限维 Koopman closure 不足。先检查 residual activity 是否在 transition 前上升，再考虑延长 history、regime-conditioned Koopman 或 mixture experts。

## 39. 第二阶段可扩展方向

在 MVP 成功后，可逐步探索：

### 39.1 Mixture-of-Koopman Experts

对不同 dynamical regimes：

\[
\hat z_{t+1}
=\sum_{m=1}^{M}\pi_m(z_t)K_mz_t+R_\psi.
\]

适合明显存在多 attractor / 多模态切换的问题。

### 39.2 Spatial Koopman tokens

不再把整个场压成单一向量，而形成

\[
Z_t\in\mathbb R^{M\times d},
\]

使 Attention 研究不同物理区域/模态之间的交互。

### 39.3 Stochastic JEPA world model

对不确定系统学习

\[
p(z_{t+1}|z_t,a_t)
\]

而不只是均值预测。

### 39.4 Operator-valued decoder

用 Neural Operator 将 latent 映射回任意坐标上的场：

\[
D:(z,x)\mapsto U(x).
\]

### 39.5 Koopman-MPC / RL

世界模型训练完成后，可用 latent imagination 做：

- CEM/MPPI/MPC；
- SAC/PPO actor-critic；
- model-based RL；
- stability-aware control。

---

## 40. 对研究价值的最终定位

本方向的价值不应建立在“同时使用了三个热门组件”上，而应建立在以下科学问题上：

\[
\boxed{
\text{What internal coordinates should an AI use to understand a physical dynamical system?}
}
\]

一个优秀的物理世界模型，其 latent 不应只是压缩得好，而应满足：

1. **Sufficiency**：保留未来预测和任务所需信息；
2. **Identifiability**：不同重要物理状态可区分；
3. **Dynamics relevance**：latent 的几何与真实演化相关；
4. **Spectral structure**：主导频率/衰减/增长可被组织和分析；
5. **Controllability**：动作后果在 latent 中可区分和规划；
6. **Physical admissibility**：不轻易产生明显违反物理的状态；
7. **Efficient closure**：不能由结构模型描述的部分由有限 residual model 补偿。

因此，该框架最强的表述不是：

> Koopman + JEPA + Transformer。

而是：

\[
\boxed{
\textbf{A physics-constrained predictive latent-state architecture with}
\textbf{spectrally structured dynamics and attention-based memory closure.}
}
\]

对应中文：

> **一种由物理方程约束、具有预测性 Koopman 潜在坐标、谱结构动力学与 Attention 记忆闭合的物理世界模型。**

若在流体/燃烧问题上能证明：

\[
\text{更稳定的 long-horizon rollout}
+\text{更正确的谱结构}
+\text{更低的物理缺陷}
+\text{更好的控制可用性},
\]

那么它的贡献将明显强于“换一个网络使一步 MSE 再下降几个百分点”。

---

## 41. 推荐首先实现的版本：不要直接实现“完整模型”

第一阶段开发目标不是得到最终论文模型，而是建立一条**每一步都可以被独立证伪和调试**的证据链：

```text
V0.1  工程骨架与接口
  ↓
V0.2  数据窗口 + PhysicsConstraint contracts
  ↓
V0.3  Direct-state KoopmanCore
  ↓
V0.4  Learned KoopmanEncoder
  ↓
V0.5  2D/PDE Koopman-only baseline
  ↓
V0.6  JEPA target/online shell
  ↓
V0.7  Residual target + tiny closure baseline
  ↓
V0.8  Attention closure / z_R
  ↓
V0.9  Joint fine-tune + diagnostics
  ↓
V1.0  Stable research baseline
```

然后再进入：

```text
V1.1  selective field decoder
V1.2  action-conditioned Koopman
V1.3  counterfactual + MPC
V2.x  mixture Koopman / stochastic latent / combustion & detonation
```

完整逐版本任务书见第三部分。


# 第三部分：版本化开发路线与 Codex 任务书

这一部分不是论文描述，而是**工程执行协议**。后续将本 Markdown 交给 Codex 时，应一次只实现一个版本；当前版本验收通过前，不进入下一版本。

## 42. Codex 开发总规则

### 42.1 一次只完成一个版本

每次给 Codex 的任务必须明确：

```text
只实现 Vx.y。
不要预实现后续版本。
不要重构与本版本无关且已通过测试的模块。
```

### 42.2 每个版本必须包含六类输出

1. **代码**：只新增/修改该版本必要文件；
2. **单元测试**：至少覆盖 shape、gradient、numerical behavior；
3. **最小运行脚本**；
4. **配置文件**；
5. **README/CHANGELOG 更新**；
6. **验收报告**：输出核心指标，不以“程序能运行”作为完成标准。

### 42.3 模块兼容原则

- 新模块通过明确接口连接，不直接访问其它模块内部成员；
- tensor shape 必须在 docstring 标明；
- 所有重要对象支持 CPU 单元测试；
- GPU 只能是加速选项，不能成为测试前提；
- 所有训练脚本必须支持 `--seed`；
- checkpoint 必须保存 model/config/normalizer/version；
- 禁止隐藏全局变量决定模型行为。

### 42.4 每个版本的 Definition of Done

版本只有同时满足以下条件才算完成：

\[
\boxed{
\text{unit tests pass}
+\text{smoke train pass}
+\text{acceptance metrics pass}
+\text{checkpoint reload pass}
}
\]

如果验收失败，先修复当前版本，不允许通过增加后续模块“掩盖”问题。


### 42.5 工程验收与科学验收必须分开

Codex 可以判断“代码是否符合工程契约”，但**不能自行宣布一个研究假设已经成立**。因此每个版本必须分成两层验收：

**Engineering Gate（Codex 可自动判断）**：

- 单元测试、shape、gradient ownership、checkpoint round-trip 全部通过；
- smoke train 无 NaN/Inf；
- 相同 seed 在 CPU 测试条件下可复现；
- 不发生未来信息泄漏；
- 已冻结模块参数和 buffer 在训练后保持不变；
- 配置、数据指纹、代码版本与运行结果被完整记录。

**Scientific Gate（输出报告，由研究者判断）**：

- 与前一版本/基线相比的 open-loop、observable、频谱和 closure 指标；
- 至少 3 seeds 时报告 median、mean、std，而不是只报最好一次；
- synthetic system 可以有硬阈值；真实 PDE 数据优先报告相对改进，不让 Codex 擅自把任意阈值解释为“科学成功”。

因此后文中的“验收指标”若涉及研究性能，应理解为**必须计算并报告**，除非文档明确给出 synthetic ground-truth 的硬阈值。

### 42.6 唯一时间语义：状态、动作与 `dt` 的对齐

这是整个项目最容易出现 silent bug 的地方，必须在 V0.1/V0.2 固定后永不改变。

定义单步转移：

\[
\boxed{
U_i \xrightarrow{\;a_i,\,\Delta t_i\;} U_{i+1}
}
\]

因此底层 trajectory 的唯一合法存储语义是：

```text
states  : [T+1, ...]       # U_0 ... U_T
actions : [T, d_a]         # a_i drives U_i -> U_{i+1}; optional
dts     : [T]              # dt_i belongs to U_i -> U_{i+1}
```

对 context 长度 `H`、预测长度 `K`，一个标准 window 定义为：

```text
context_states  : states[t-H+1 : t+1]        # [H, ...], ends at U_t
history_actions : actions[t-H+1 : t]          # [H-1, d_a]
history_dts     : dts[t-H+1 : t]              # [H-1]
future_actions  : actions[t : t+K]             # [K, d_a]
future_dts      : dts[t : t+K]                 # [K]
future_states   : states[t+1 : t+K+1]          # [K, ...]
```

其中 `future_actions[:,0]` 与 `future_dts[:,0]` **必须且只允许**用于 `U_t -> U_{t+1}`。

禁止使用含义模糊的 `a_context: [B,H,d_a]` 而不说明最后一个 action 对应哪一个 transition。

### 42.7 标准 `ProblemBatch` 与网格/单位元数据

V0.1 必须定义统一 Batch contract，后续 Dataset 只负责构造它，模型不得直接猜测字段含义。

推荐：

```python
@dataclass
class ProblemBatch:
    context_states_raw: Tensor      # [B,H,C,*spatial]
    future_states_raw: Tensor       # [B,K,C,*spatial]
    context_states_model: Tensor    # normalized/model input, same shape
    future_states_model: Tensor     # normalized/model target, same shape
    history_actions: Tensor | None  # [B,H-1,d_a]
    future_actions: Tensor | None   # [B,K,d_a]
    history_dts: Tensor             # [B,H-1]
    future_dts: Tensor              # [B,K]
    mu_static: Tensor | None        # [B,d_mu]
    coordinates: Tensor | None      # grid coordinates / mesh coordinates
    cell_weights: Tensor | None     # quadrature weights / cell volumes
    valid_mask: Tensor | None       # geometry/domain mask
    trajectory_id: Tensor | list
```

必须同时维护 `raw` 与 `model` 两套状态：

- `raw`：有真实物理单位，用于 `PhysicalProbe`、物理损失、最终指标；
- `model`：经过训练集统计量归一化，用于 neural encoder。

\[
\boxed{
G_{phys}\text{ 必须作用于 raw physical state，不能直接对 normalized field 计算质量、能量、涡量等物理量。}
}
\]

规则网格还必须保存 `dx/dy` 或等价坐标；积分量使用 `cell_weights`，不能默认“所有网格点等权”适用于所有数据。

### 42.8 变量语义、单位和通道顺序必须进入配置

禁止模型通过位置猜测 `channel 0` 是密度、`channel 1` 是速度。数据配置必须至少包含：

```yaml
channels:
  - {name: rho, unit: kg/m^3}
  - {name: u,   unit: m/s}
  - {name: v,   unit: m/s}
grid:
  layout: channels_first
boundary:
  type: periodic   # example only
```

`PhysicalProbe` 根据 channel name/ProblemSpec 查找物理变量。缺少必需变量时应明确报错或跳过该 observable，禁止悄悄用错误 channel。

### 42.9 配置必须严格校验，禁止 silent defaults

推荐在 `src/config/` 中定义 structured dataclass 配置并显式 validate。核心原则：

- 未识别字段报错；
- 负的 loss weight 报错；
- `d_model % n_heads != 0` 报错；
- `history < 2`、`horizon < 1` 报错；
- action model 开启但 dataset 无 action 时立即报错；
- variable-`dt` 数据下 residual attention 若未接收 `dt` conditioning，应立即报错。

Codex 不得为了“让程序能跑”自动填充关键维度、物理单位或数据路径。

### 42.10 数值精度与设备契约

V0.x 默认以 `float32` 训练；CPU 单元测试可对关键解析问题使用 `float64`。

以下操作属于 **precision islands**：

- `torch.matrix_exp`；
- eig/eigenvalue diagnostics；
- 高条件数 covariance / condition number；
- 物理积分的参考测试。

第一版禁止在这些操作外层盲目开启 AMP/autocast。若未来启用 mixed precision，必须在 precision island 中显式关闭 autocast。

另外：

- `forward()` 内部禁止隐式 `.cpu()` / NumPy 转换；
- tensor 的 device/dtype 跟随输入；
- V0.x 不要求 `torch.compile`、DDP、多 GPU；这些在 V1.0 稳定后再加入。

### 42.11 EMA teacher 与同坐标动力学 target 必须分开

这里是 v2.1 的重要修正。

**EMA target encoder 的作用是 JEPA representation target；它不应被默认当作所有 Koopman/residual loss 的右端 target。**

原因是：

\[
E_{\bar\theta}(U) \neq E_\theta(U)
\]

即使二者参数很接近，也可能存在小的坐标漂移。差分

\[
E_{\bar\theta}(U_{t+1}) - K E_\theta(U_t)
\]

会混入“teacher 坐标漂移”，这不是物理 residual。

因此统一规定：

1. **Koopman consistency target**：默认使用同一个 online encoder 对未来状态编码；训练时可按具体 loss 决定是否对未来分支 stop-gradient；
2. **Residual target**：必须使用同坐标的 detached online future latent：

\[
\boxed{
r_{t+1}^{tar}
=\operatorname{sg}\left[E_\theta(U_{t+1})\right]
-\operatorname{sg}\left[z_{t+1}^{K,base}\right]
}
\]

3. **JEPA target**：才使用 EMA encoder：

\[
z_{t+1}^{jepa,tar}=E_{\bar\theta}(U_{t+1}).
\]

在 V0.7/V0.8 residual warm-up 开始前，执行一次：

```text
target_encoder <- hard_copy(online_encoder)
freeze online_encoder
freeze target_encoder
pause EMA update
```

这样 residual target 的坐标系完全固定。V0.9 joint fine-tune 后才恢复 EMA。

### 42.12 EMA 更新的唯一合法顺序

JEPA stage 中每一步顺序固定为：

```text
zero_grad
-> online forward
-> target forward under no_grad
-> loss backward
-> optimizer.step()
-> scheduler.step() if step-based
-> EMA update(target <- online)
-> logging/checkpoint
```

target encoder：

- 初始化时必须 exact hard copy online encoder；
- 永远不加入 optimizer；
- `requires_grad=False`；
- EMA 必须包含 parameters 和需要同步的 buffers；
- 第一版 encoder 禁止使用 BatchNorm，优先 LayerNorm/GroupNorm，减少 EMA buffer 歧义。

### 42.13 冻结/解冻必须由训练状态机统一管理

禁止在多个训练脚本里零散写 `param.requires_grad_(False)`。

实现：

```python
class TrainStage(Enum):
    KOOPMAN = "koopman"
    JEPA = "jepa"
    RESIDUAL = "residual"
    JOINT = "joint"
```

以及唯一入口：

```python
configure_trainable(model, stage)
assert_optimizer_matches_trainable_params(model, optimizer)
```

必须自动检查：

- 每个 trainable parameter 恰好出现在一个 optimizer group；
- frozen parameter 不在 optimizer 中；
- optimizer step 前后 frozen parameter 数值完全不变；
- 每个参数组有明确名称和日志中的 grad norm。

阶段切换时默认**重新创建 optimizer/scheduler**，不继承前一阶段 momentum，除非文档明确要求。

### 42.14 Closed-loop rollout 是唯一正式预测指标，禁止未来真值泄漏

v2.2 的默认 closure state **不包含 PhysicalProbe**。正式 closed-loop rollout 在 context 结束后只能使用：

- 自己预测得到的 `z_k` 历史；
- 已知 future action；
- 已知 `dt`；
- static/known parameters `mu`。

PhysicalProbe 若存在，只能用于 rollout 完成后的 metric/diagnostic，或在明确的 ablation 中作为额外输入；MVP 不依赖它。

正式 rollout 禁止访问：

```text
future_states_raw
future_states_model
true future physical probes
true future encoded z_k
```

除非只是计算 loss/metric，且该 tensor 不进入下一步 transition path。

必须增加 `test_no_future_leakage.py`：将未来真值与未来 probe 随机打乱后，closed-loop prediction 必须保持不变。

### 42.15 Attention 必须是因果的，并显式条件化真实时间间隔

ResidualMemory 不是普通 bidirectional encoder。第一版强制 causal mask。

若 `dt` 可变，token 必须包含与状态到下一状态对应的 `dt`：

\[
\boxed{
\xi_i=P([z_i^K,a_i,\mu,\Delta t_i])
}
\]

仅使用位置 index 不能表达真实时间间隔。

必须增加 causal test：修改未来 token 不得改变当前位置输出 `z_R`。

第一版 gate 统一为**每个 sample/step 一个 scalar**：

```text
gate       : [B,1]
delta_z_k  : [B,d_k]
correction : gate * delta_z_k
```

vector gate 留到后续版本。

### 42.16 Closure burden 的稳定定义

不建议只用 `||correction|| / ||z_base||`，因为 latent 原点和尺度会影响解释。第一版同时记录：

\[
\Delta z_{base}=z_{t+1}^{base}-z_t^K,
\]

以及有界 closure fraction：

\[
\boxed{
C_{closure}
=\frac{\|g\Delta z\|_2}
{\|g\Delta z\|_2+\|\Delta z_{base}\|_2+\epsilon}
\in[0,1].
}
\]

另记录：

\[
R_{closure}=\frac{\|g\Delta z\|_2}{\|\Delta z_{base}\|_2+\epsilon}.
\]

研究阶段主要看 `C_closure` 的分布与 transient/mode switching 的关联，不规定一个跨所有系统都成立的硬阈值。

### 42.17 训练时的 teacher forcing 与 rollout curriculum

V0.7/V0.8 可以先用真实 context latent 训练单步 residual；但 V0.9 必须加入 closed-loop multi-step loss，否则 closure 很容易只会纠正“真实历史下的一步误差”。

推荐 curriculum：

```text
horizon 1 -> 2 -> 4 -> 8 -> 16
```

每一阶段先保持 `KoopmanCore` 稳定，再逐渐增加 full-model rollout horizon。

正式 multi-step loss 必须用预测 latent 更新历史队列；不能每一步重新编码真实 `U_{t+k}` 作为下一步输入。

### 42.18 Checkpoint schema 与恢复语义

checkpoint 不只是 `model.state_dict()`。至少保存：

```text
schema_version
architecture_revision
stage
online_model_state
target_encoder_state
optimizer_state
scheduler_state
amp_scaler_state (if used)
epoch/global_step
normalizer_state
physics_constraint_spec
resolved_config
config_hash
data_fingerprint
split_manifest
python_random_state
numpy_random_state
torch_cpu_rng_state
torch_cuda_rng_state (if available)
git_commit (if available)
```

V0.x 对“精确恢复”的保证定义为**epoch boundary resume**。若未来需要 mid-epoch exact resume，需要额外保存 sampler/DataLoader 状态，不允许文档含糊地声称已经 exact resume。

读取 checkpoint 时：

- schema/version 不兼容必须报错；
- 禁止 `strict=False` 静默吞掉缺失参数，除非存在显式 migration 函数；
- resume 后第一个 validation 输出应与保存前一致到数值容差。

### 42.19 数据指纹、split manifest 与复现

每个数据集必须生成：

- trajectory ID 列表；
- train/val/test split manifest；
- 数据文件路径/大小/mtime 或内容 hash（按成本选择）；
- channel/unit/grid metadata；
- normalizer statistics hash。

窗口必须在 trajectory split **之后**生成，严禁先切 window 再随机分 train/val/test。

运行目录中必须复制保存 resolved config 与 split manifest。

### 42.20 日志、NaN guard 与 Debug Hooks

每个训练 step/epoch 至少支持记录：

- total loss 与每个 loss component；
- module-wise grad norm / parameter norm；
- latent mean/std/covariance diagnostics；
- base/full rollout error；
- `C_closure`、gate mean/std；
- spectral diagnostics；
- physical observable error；
- learning rate。

出现 NaN/Inf 时必须：

1. 停止 optimizer update；
2. 输出最近 batch 的 trajectory IDs / dt range / loss components；
3. 保存 `debug_nan.pt`（不包含超大原始数据副本）；
4. 明确报错，而不是自动把 NaN 替换为 0。

`torch.autograd.detect_anomaly()` 仅作为 debug flag，默认关闭。

### 42.21 最小依赖与 Codex 不得擅自扩大技术栈

V0.x 优先只依赖：

```text
Python >= 3.10
PyTorch
NumPy
PyYAML (若使用 YAML config)
pytest
matplotlib (仅分析脚本，可选)
```

Codex 不得因为方便自动引入 Lightning、Hydra、Ray、wandb、xformers、transformers、PhysicsNeMo 等大型依赖。若以后需要，应在独立版本中引入并解释必要性。

### 42.22 命名契约

核心状态数学记号使用 `z^K,z^R`。可选物理诊断记为 `q`。Python 代码统一 snake_case：

```text
z_k
z_r
z_k_base
delta_z_k
z_k_next
```

公共 API 禁止同时存在 `zK`、`z_K`、`z_k` 三种写法。

### 42.23 跨模型比较时禁止直接比较 latent MSE

两个不同 encoder 的 latent 空间可以经过旋转、缩放甚至一般可逆变换，因此

\[
\|z^{(A)}-z^{(B)}\|
\]

通常没有跨模型物理意义。

因此：

- latent rollout error 只用于**同一 frozen encoder 坐标系内部**的 ablation/训练诊断；
- 跨模型正式比较优先使用 raw-unit physical metrics/probes、field error、频率/谱、控制性能；
- 若确实比较 latent geometry，需要先定义明确 alignment/procrustes/CCA 等协议，V1.0 前不实现。

### 42.24 外部数据与最小 smoke dataset 分开

Codex 不应因为 V0.5 文档写了 cylinder wake 就自动联网下载数据或顺手写一个复杂 CFD solver。

统一规定：

- **内置 smoke dataset**：仓库自己生成、很小、确定性、用于 CI/接口验证；
- **research dataset adapter**：只定义加载接口，真实 cylinder wake / Navier–Stokes / combustion 数据由研究者显式提供路径。

V0.5 默认 smoke dataset 采用**二维周期 advection-diffusion 的解析/半解析场序列**，用于验证 `[B,H,C,Nx,Ny] -> encoder -> Koopman rollout` 全链路；它不承担证明复杂非线性世界模型有效的任务。

### 42.25 必须新增的关键单元测试

在原测试清单基础上，至少加入：

```text
test_problem_batch_alignment.py
test_raw_vs_model_physics_semantics.py
test_optimizer_gradient_ownership.py
test_target_hard_sync_and_ema_order.py
test_same_coordinate_residual_target.py
test_causal_residual_memory.py
test_no_future_leakage.py
test_closed_loop_rollout_history_update.py
test_variable_dt_conditioning.py
test_checkpoint_schema_and_resume.py
test_config_rejects_unknown_fields.py
test_frozen_parameters_do_not_change.py
```

这些测试比“网络能否 forward”更重要，因为它们直接保护研究逻辑。

---

## 43. V0.1 — Project Skeleton & Contracts

### 目标

只建立工程骨架、配置系统、数据类型和测试基础设施，不训练任何神经网络。

### 新增模块

```text
src/models/types.py
src/utils/seed.py
src/utils/checkpoint.py
src/utils/logging.py
configs/
tests/
```

### 核心数据类型

V0.1 即固定 Python 命名为 snake_case，并同时实现 `ProblemBatch`/`ProblemSpec`，避免 V0.2 再改变公共接口。

```python
@dataclass
class LatentState:
    z_k: Tensor
    z_r: Tensor | None = None

@dataclass
class TransitionOutput:
    z_k_base: Tensor
    z_r: Tensor | None
    delta_z_k: Tensor
    gate: Tensor
    z_k_next: Tensor

@dataclass
class ProblemBatch:
    context_states_raw: Tensor
    future_states_raw: Tensor
    context_states_model: Tensor
    future_states_model: Tensor
    history_actions: Tensor | None
    future_actions: Tensor | None
    history_dts: Tensor
    future_dts: Tensor
    mu_static: Tensor | None = None
    coordinates: Tensor | None = None
    cell_weights: Tensor | None = None
    valid_mask: Tensor | None = None
    trajectory_id: object | None = None
```

`ProblemSpec` 保存 channel names/units、grid、boundary 与 observable requirements；其字段必须可序列化进 checkpoint/config。

### 必测项目

- dataclass 创建和 device transfer；
- config load；
- seed reproducibility；
- checkpoint round-trip；
- CPU pytest 全通过。

### 验收

```bash
pytest -q
```

必须全绿。

### Codex 任务边界

**禁止**实现 Encoder、Koopman、JEPA、Transformer。

---

## 44. V0.2 — Data Windows & Physics Contracts

### 目标

建立严格的时间窗数据协议、raw/model 双状态语义以及物理约束接口。PhysicalProbe 只作为可选诊断模块。

### 输入协议

严格遵守第 42.6 节时间语义。底层 trajectory 使用

```text
states  : [T+1,...]
actions : [T,d_a]
dts     : [T]
```

且唯一语义为

\[
U_i\xrightarrow{a_i,\Delta t_i}U_{i+1}.
\]

### 新增模块

```text
src/data/datasets.py
src/data/windows.py
src/data/splits.py
src/data/normalization.py
src/physics/constraints.py
src/physics/operators.py
src/physics/probes.py        # optional
```

### 第一版 PhysicsConstraint

先实现通用 contract 和 toy/advection-diffusion 可验证约束：

- boundary-condition consistency；
- state admissibility/finite-value checks；
- 可用时的 conservation/discrete residual；
- custom constraint registry。

### PhysicalProbe（可选）

允许支持少量通用统计量，例如 channel mean/RMS、kinetic energy、enstrophy 等，但它们：

- 无 trainable parameter；
- 只读取 raw-unit state；
- 不属于 `LatentState`；
- 不作为后续 closure 的必需输入。

normalizer 只在 train split fit；val/test 禁止重新 fit。V0.2 保存 `split_manifest`、`data_fingerprint`、`normalizer_state` 和 physics-contract metadata。

### 测试

- window 不跨 trajectory；
- train/val/test 无 trajectory leakage；
- raw/model 不混用；
- PhysicsConstraint 输入单位正确；
- optional probe 可完全关闭；
- normalizer round-trip；
- shape/alignment tests。

### 验收

生成 toy dataset，运行数据窗口和 physics contract smoke test；关闭所有 probe 后流程仍完全可运行。

## 45. V0.3 — Direct-State Continuous-Time KoopmanCore

### 目标

**在没有 learned encoder 的情况下**先证明 Koopman 数值核心完全正确。

### 数据

至少两个系统：

1. damped harmonic oscillator；
2. 一个非线性系统，例如 Duffing oscillator。

线性 oscillator 用来做严格谱验证；Duffing 仅用于观察有限维线性近似的局限。

### 数学

\[
\dot z=Az,
\qquad
z(t+\Delta t)=e^{A\Delta t}z(t).
\]

### 新增模块

```text
src/models/koopman_core.py
src/rollout/koopman_rollout.py
src/metrics/spectral.py
scripts/generate_oscillator.py
scripts/analyze_spectrum.py
```

### API

```python
step(z, dt)
rollout(z0, dts, horizon)
spectrum()
```

### 必须支持

- scalar `dt`；
- batch `dt`；
- `torch.matrix_exp` gradient；
- continuous eigenvalue diagnostics。

数值契约：

- `step(z, dt)` 计算 `matrix_exp(dt * A) @ z`，不得用一阶 Euler 偷换；
- eigenspectrum 仅用于 detached diagnostics，第一版不通过复杂特征值排序反向传播；
- matrix exponential 在 autocast precision island 中执行；
- analytical oscillator test 使用 float64 reference。

### 验收指标

对线性 oscillator：

- learned frequency 与真值相对误差 < 1%；
- 100-step rollout 不出现数值爆炸；
- checkpoint reload 后输出一致。

### Codex 边界

禁止 learned encoder；禁止 JEPA；禁止 residual network。

---

## 46. V0.4 — Learned KoopmanEncoder + TrainingDecoder on Synthetic Data

### 目标

引入 `E_K` 与轻量 `D_train`，验证 learned lifting 同时具有：

- Koopman 可传播性；
- state sufficiency / 可重构性；
- non-collapse。

### 新增模块

```text
src/models/koopman_encoder.py
src/models/training_decoder.py
src/losses/koopman.py
src/losses/collapse.py
src/losses/reconstruction.py
src/train/train_koopman.py
```

### 训练目标

\[
\mathcal L
=\lambda_K\mathcal L_K
+\lambda_M\mathcal L_{K,multi}
+\lambda_{rec}\mathcal L_{rec}
+\lambda_V\mathcal L_{var}
+\lambda_S\mathcal L_{spec}.
\]

Synthetic 版本可先不加复杂 PDE residual，但 `D_train` 从本版本开始是正式接口，不再只是可选 warm-up head。

### 必须记录

- `z_k` mean/std 与 covariance condition；
- reconstruction error；
- one/multi-step latent rollout；
- learned spectrum。

### 验收

- `z_k` 不塌缩；
- reconstruction/state discrimination 正常；
- Koopman-only rollout 优于 persistence；
- V0.3 数值核心回归测试全部通过。

## 47. V0.5 — 2D/PDE Koopman-Only + PhysicsConstraint Baseline

### 目标

将 `E_K + D_train + KoopmanCore` 扩展到规则网格物理场，并第一次让物理约束直接作用于 decoded prediction；仍然不使用 JEPA 和 Attention。

### Encoder / Decoder

第一版使用小型 CNN encoder + 对称/轻量 decoder，避免同时引入多个复杂 backend。

\[
U_t\xrightarrow{E_K}z_t^K
\xrightarrow{\mathcal K} \hat z_{t+1}^K
\xrightarrow{D_{train}}\hat U_{t+1}.
\]

### 推荐测试问题

1. 仓库内置二维周期 advection-diffusion smoke dataset；
2. 研究者提供的 cylinder wake / 2D Navier--Stokes / reaction-diffusion。

### Loss

\[
\mathcal L
=\mathcal L_K
+\lambda_M\mathcal L_{K,multi}
+\lambda_{rec}\mathcal L_{rec}
+\lambda_P\mathcal L_{physics}
+\lambda_V\mathcal L_{var}.
\]

优先从便宜的 BC、守恒、离散 operator consistency 开始；完整连续 PDE residual 不是强制第一项。

### 验收

至少比较 persistence、DMD/POD（可实现时）、learned Koopman，并报告：

- latent rollout；
- raw-unit field/reconstruction error；
- physics constraint violation；
- frequency/spectrum；
- `z_k` variance。

PhysicalProbe 只作为可选额外 metric。

## 48. V0.6 — JEPA Online/Target Shell over Koopman Dynamics

### 目标

只加入 JEPA 训练外壳，不加入 residual Attention。

### 新增模块

```text
src/models/target_encoder.py
src/losses/jepa.py
src/train/train_jepa.py
```

### 结构

\[
z_t^K=E_\theta(U_t),
\qquad
z_{t+k}^{K,tar}=E_{\bar\theta}(U_{t+k}),
\]

\[
\bar\theta\leftarrow\tau\bar\theta+(1-\tau)\theta.
\]

预测仍然纯 Koopman：

\[
\hat z_{t+k}^{K}=\mathcal K^{(k)}(z_t^K).
\]

### 关键 target 语义

本版本开始必须区分两种 target：

- `z_k_online_future = E_theta(U_future)`：与当前动力学处于同一 latent 坐标系，用于 Koopman consistency 等结构 loss；
- `z_k_jepa_target = E_bar(U_future)`：仅用于 JEPA target。

禁止无条件用 EMA target 替代所有 Koopman future target。

### 测试

- target 初始化时与 online exact hard-sync；
- EMA 参数只在 `optimizer.step()` 之后通过 update 函数变化；
- target branch `requires_grad=False` 且不属于 optimizer；
- target 输出无 gradient graph；
- 保存/恢复 EMA state 与 update step；
- 第一版 encoder 不使用 BatchNorm。

### 验收

相较 V0.5：

- latent training 更稳定或 long-horizon 不下降；
- 不允许通过增加网络参数掩盖性能退化；
- 所有 KoopmanCore 老测试继续通过。

---

## 49. V0.7 — Residual Dataset Infrastructure + Tiny Closure Baseline

### 目标

验证“冻结 Koopman 后，closure residual 是否具有可学习的历史结构”。先不用 Transformer。

### 新增模块

```text
src/data/residual_cache.py
src/losses/residual.py
src/metrics/residual_burden.py
src/models/residual_head.py
src/train/train_residual.py
```

### 核心 target

进入 V0.7 时冻结：

```text
online encoder : frozen
Koopman core   : frozen
training decoder: frozen
EMA target     : paused / not used for residual difference
```

同坐标 residual：

\[
\boxed{
r_{t+1}
=\operatorname{sg}\left[
E_K(U_{t+1})-\mathcal K(E_K(U_t),a_t)
\right].
}
\]

离线 cache 至少保存：

```text
z_k_history
action_history
dt_history
mu
residual_target
trajectory_id/time_index
encoder_hash
koopman_hash
```

若 encoder/core 权重变化，cache 必须判定失效。

### Tiny closure baseline

先使用线性层、小 MLP 或 GRU summary：

```text
[z_k history summary, action/dt] -> delta_z_k
```

不依赖 PhysicalProbe。

### 测试

- residual target 无 gradient；
- frozen 参数 step 前后完全不变；
- cache hash/version guard；
- `delta_z_k=0` 时严格退化为 Koopman-only；
- residual burden metric 正确。

### 验收

如果 tiny baseline 连 residual 都完全不可预测，应先检查 `z_k` 是否足够、history 是否需要延长或 Koopman representation 是否不合适，而不是直接用更大的 Transformer 掩盖问题。

## 50. V0.8 — Attention Closure / History-Dependent `z_R`

### 目标

正式引入 closure memory state：

\[
\boxed{
z_t^R=M_\psi(\mathcal H_t)}.
\]

`z_R` 不是“第三个独立物理 latent”，而是为了预测冻结 Koopman residual 所形成的历史隐藏状态。

### 新增模块

```text
src/models/residual_memory.py
src/models/gate.py
configs/model/full_closure.yaml
```

### 第一版 Transformer 约束

```text
d_model: 64 or 128
layers : 2-3
heads  : 4
history: 8-16
```

### 输入 token

\[
\boxed{
\xi_t=P([z_t^K,a_t,\mu,\Delta t_t])
}
\]

无控制系统去掉 `a_t`。PhysicalProbe 默认不输入；仅在后续 ablation 通过 `use_probe_input=true` 显式加入。

### Closed-loop 历史更新

未来 transition history 只更新：

```text
predicted z_k
known future action
known future dt
known/static parameters
```

不得从 `U_future` 重算任何 transition 输入。

### 输出

\[
\Delta z_{t+1}^K=W_Rz_t^R,
\qquad
g_t=\sigma(w_g^Tz_t^R+b_g).
\]

第一版：

```text
z_r        : [B,d_r]
delta_z_k  : [B,d_k]
gate       : [B,1]
```

Residual head near-zero 初始化；gate 初始较小。

### 训练

继续冻结 Koopman branch：

\[
\mathcal L=\mathcal L_R+\lambda_B\mathcal L_{budget}.
\]

### 验收

报告：

- `C_closure(t)`；
- gate；
- base vs corrected error；
- residual 与瞬态/切换事件的关系；
- teacher-forced diagnostic vs closed-loop rollout gap。

必须通过 causal mask、no-future-leakage 和 residual-cache compatibility tests。

## 51. V0.9 — Controlled Joint Fine-Tuning

### 目标

第一次允许 Koopman 与 Attention 同时训练，但采用不同学习率和完整诊断。

### 参数组

```python
optimizer = AdamW([
    {"params": koopman_encoder.parameters(), "lr": 1e-5},
    {"params": koopman_core.parameters(),    "lr": 1e-5},
    {"params": residual_memory.parameters(), "lr": 1e-4},
    {"params": residual_head.parameters(),   "lr": 1e-4},
])
```

target encoder 在本阶段重新 hard-sync 一次后恢复 EMA；EMA 顺序严格遵守第 42.12 节。

Joint fine-tune 开始加入真正的 closed-loop multi-step curriculum：

```text
horizon: 1 -> 2 -> 4 -> 8 -> 16
```

每一步未来历史只由模型自己的 `z_k_next` 与给定 future action/dt/parameters 更新，禁止 teacher forcing 作为正式 rollout loss 的唯一来源。

### loss

开启：

- JEPA；
- Koopman consistency；
- multi-step；
- residual target；
- residual budget；
- non-collapse；
- reconstruction/state sufficiency；
- decoded PhysicsConstraint；
- optional PhysicalProbe loss（若配置启用）。

PDE/BC/守恒约束按 V0.5 已验证的接口继续使用。

### 必须比较

\[
E_{base}(k)
\quad\text{vs}\quad
E_{full}(k).
\]

且持续比较训练前后 Koopman spectrum，避免 joint training 破坏原结构。

### 验收

- full model long-horizon 优于 V0.6；
- base Koopman 性能不能严重退化；
- `C_closure/R_closure` 分布受控且 closure 不长期接管主动力学；
- decoded physical constraint violation 不恶化；
- 至少 3 个随机种子。

---

## 52. V1.0 — Stable Research Baseline

### 目标

把 V0.x 中已验证的组件固化为第一版可用于科研实验的基线。

### 必须具备

- 完整 config；
- deterministic training option；
- checkpoint/resume；
- metrics CSV/JSON；
- spectrum analysis；
- latent visualization；
- residual burden analysis；
- baseline comparison；
- ablation switches；
- resolved config + config hash；
- data fingerprint + split manifest；
- stage-aware checkpoint schema；
- closed-loop no-leakage regression test；
- module-wise gradient ownership report。

### 必备消融开关

```text
use_jepa
use_koopman
use_residual
use_gate
use_physics_constraints
use_physical_probe
freeze_koopman
```

### V1.0 不包含

- full-field high-resolution decoder；
- MPC；
- RL；
- mixture-of-Koopman experts；
- stochastic latent；
- 3D combustion。

这保证 V1.0 的科学问题非常纯粹：

\[
\boxed{
\text{structured latent dynamics}
+\text{JEPA}
+\text{attention closure}
}
\]

是否真的改善 physical world modeling？

---

## 53. V1.1 — High-Fidelity Physical Projection / Field Decoder

### 目标

解决 latent 到物理空间的映射，但不改变 latent dynamics 本体。

两种 readout：

### A. 低维 observables

\[
H_q(z_K,z_R)\rightarrow q.
\]

### B. 完整场

\[
D_{field}(z_K,z_R,\mu)\rightarrow \hat U.
\]

推荐 decoder 独立训练/微调，先冻结 world model，避免 decoder reconstruction pressure 破坏 latent dynamics。

验收必须同时报告 latent error 和 field/observable error。

---

## 54. V1.2 — Action-Conditioned Koopman

### 目标

把模型从 autonomous world model 扩展为 Physical AI/control world model。

逐级实现：

### Level 1

\[
z_{t+1}=Kz_t+Ba_t.
\]

### Level 2

\[
z_{t+1}=Kz_t+Ba_t+\sum_ja_jN_jz_t.
\]

只有 Level 1 通过后才实现 Level 2。

必须设计 action-held-out test，不能只在训练 action 分布内验证。

---

## 55. V1.3 — Counterfactual Dynamics & MPC

### 目标

验证 latent state 是否真正保留 action consequence。

加入：

\[
\mathcal L_{cf}
\]

并实现 latent MPC：

\[
\min_{a_{t:t+H-1}}
\sum_{k=1}^{H}
\ell(H_q(z_{t+k}),q_{target})
+\lambda_a\|a_{t+k}\|^2.
\]

此时仍不需要 RL。

只有当 MPC 已验证 world model 的 controllability 后，再考虑 RL。

---

## 56. V2.x — 研究扩展，不应提前实现

候选方向：

1. Mixture-of-Koopman experts：用于 bifurcation / mode switching；
2. spatial Koopman tokens：处理局部结构；
3. stochastic JEPA：处理多未来与不确定性；
4. Neural Operator decoder；
5. irregular mesh / graph encoder；
6. combustion/reaction source conditioning；
7. detonation mode switching；
8. MPC → RL policy distillation。

这些都应该建立在 V1.0/V1.3 证据充分后再做。

---

## 57. 推荐 Git/版本管理方式

每个版本建议一个独立 tag：

```text
v0.1-contracts
v0.2-data-physics-contracts
v0.3-koopman-core
v0.4-koopman-encoder
v0.5-pde-koopman
v0.6-jepa-shell
v0.7-residual-target
v0.8-attention-closure
v0.9-joint
v1.0-research-baseline
```

每个版本只允许若干小 commit，例如：

```text
feat: add continuous koopman core
 test: validate matrix exponential rollout
 docs: record v0.3 acceptance results
```

不要使用一个巨大 commit 完成多个阶段。

---

## 58. 每次交给 Codex 的标准任务模板

后续可以复制下面模板，并只替换 `<VERSION>`：

```text
你正在实现项目 <VERSION>。

必须先阅读仓库中的 architecture markdown，并严格遵守该版本边界。

目标：
- 只完成 <VERSION> 定义的功能。
- 不实现任何后续版本功能。
- 保持已有公共 API 向后兼容，除非本版本文档明确要求修改。

实现要求：
1. 先检查现有代码和测试，不重复实现已有功能。
2. 每个新增模块必须有类型标注、docstring、明确 tensor shape。
3. 新增或修改对应 pytest。
4. 提供最小 smoke-run 脚本。
5. 所有测试必须可在 CPU 上运行。
6. 训练脚本必须支持 seed、config、checkpoint/resume。
7. 关键诊断指标必须写入日志。
8. 不要为了让测试通过而删除已有断言或降低验收标准。
9. 不允许读取未来真值来更新 closed-loop rollout 历史；teacher-forced 与 closed-loop 必须分开实现。
10. 不允许擅自新增大型依赖、联网下载 research dataset 或实现后续版本组件。
11. 修改公共 API 前先检查 architecture contract；若文档与代码冲突，应报告冲突，不自行猜测。
12. 对 frozen/EMA/stop-gradient 逻辑必须增加 gradient ownership 单元测试。

完成后输出：
- 修改文件列表；
- 核心设计说明；
- 测试命令及结果；
- 本版本验收指标；
- 已知限制；
- 明确说明未实现的下一版本功能。
```

---

## 59. 第一条 Codex 实施指令建议

真正开始写代码时，第一条只给：

```text
实现 V0.1 — Project Skeleton & Contracts。
严格不要实现 V0.2 及之后任何模型或训练逻辑。
```

V0.1 验收通过后，再给 V0.2。这样可以最大程度避免 Codex 一次性生成大量耦合代码，后期无法定位问题。

---



## 60. Codex 实施前最终风险审计清单（v2.2 更新）

在把任何一个版本交给 Codex 前，先逐项确认：

### A. 数据是否闭合

- [ ] `states[T+1] / actions[T] / dts[T]` 对齐唯一；
- [ ] context/future window 不跨 trajectory；
- [ ] future action 第 0 个元素明确对应当前状态到下一状态；
- [ ] raw/model 两套状态没有混用；
- [ ] PhysicsConstraint 与 PhysicalProbe（若启用）只读取正确的 raw-unit state/geometry metadata；
- [ ] train split 先于 normalizer fit 和 window generation；
- [ ] data fingerprint 与 split manifest 已保存。

### B. latent target 是否在同一坐标系

- [ ] Koopman/residual target 使用 online encoder 同坐标表示；
- [ ] EMA teacher 仅承担 JEPA target；
- [ ] V0.7/V0.8 residual warm-up 前执行 hard-sync 并暂停 EMA；
- [ ] residual target 两端都 stop-gradient；
- [ ] 不存在 teacher drift 被误解释为 physical residual 的路径。

### C. rollout 是否真的可部署

- [ ] closed-loop future 不访问 `U_future`；
- [ ] future transition 不读取真实 `U_future` 或真实 PhysicalProbe；
- [ ] closure 默认只依赖预测 `z_k`、action、dt、parameters；
- [ ] attention 使用 causal mask；
- [ ] variable `dt` 已进入 token/transition；
- [ ] teacher-forced 与 closed-loop 指标分别命名、分别日志；
- [ ] no-future-leakage test 已通过。

### D. 优化器与梯度是否符合阶段职责

- [ ] train stage 通过统一 state machine 设置；
- [ ] frozen params 不在 optimizer；
- [ ] target params 不在 optimizer；
- [ ] trainable params 恰好属于一个 optimizer group；
- [ ] 每个 group 的 grad norm 有日志；
- [ ] stage transition 默认重建 optimizer；
- [ ] joint fine-tune 中 Koopman LR 明显小于 residual LR。

### E. 数值核心是否可信

- [ ] `matrix_exp` 不是 Euler 近似；
- [ ] analytic oscillator float64 test 通过；
- [ ] eig diagnostics detached；
- [ ] autocast 没有覆盖 precision islands；
- [ ] NaN/Inf guard 会 fail-fast；
- [ ] 100-step synthetic rollout 有稳定性回归测试。

### F. 研究指标是否没有被实现细节误导

- [ ] latent MSE 只在同一 frozen coordinate system 内比较；
- [ ] 跨模型比较使用 raw-unit physical metrics/probes/field metrics；
- [ ] closure 使用 `C_closure` + `R_closure`，不是单一不稳定比例；
- [ ] base Koopman 和 full model 的 error 同时报告；
- [ ] 至少 3 seeds 的阶段报告不只保留 best run；
- [ ] Codex 只输出 scientific evidence，不自行宣布论文结论成立。

### G. 代码是否仍保持模块化

- [ ] `world_model.py` 仅编排；
- [ ] `physics/` 无 trainable NN；
- [ ] `koopman_core.py` 不依赖 JEPA/Attention；
- [ ] `residual_memory.py` 不调用 Koopman 内部实现；
- [ ] `PhysicalProbe` 为可选诊断模块，不属于 LatentState；
- [ ] `TrainingDecoder` 与高保真 `FieldDecoder` 职责分离；
- [ ] 后续版本功能没有提前进入当前版本；
- [ ] public API 统一 snake_case。

如果其中任意一个涉及**数据泄漏、坐标系不一致、冻结失败或 checkpoint 无法恢复**的条目未满足，禁止进入下一版本，即使 loss 看起来更低。

## 61. 参考文献与近期方向（用于实现前阅读）

以下文献按与本框架的直接相关性排列：

1. **Assran et al., V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (2025).**  
   arXiv:2506.09985  
   https://arxiv.org/abs/2506.09985

2. **Nie et al., Phys-JEPA: Physics-Informed Latent World Models for Multivariate Time-Series Forecasting (2026).**  
   arXiv:2606.16076  
   https://arxiv.org/abs/2606.16076

3. **Li et al., Koopman Dreamer: Spectrally Constrained Latent Dynamics for Stable World-Model Imagination (2026).**  
   arXiv:2607.19719  
   https://arxiv.org/abs/2607.19719

4. **Chung, Bu & Verma, Physics-conforming Latent Twins (2026).**  
   arXiv:2606.15053  
   https://arxiv.org/abs/2606.15053

5. **Zeng, Ren & Song, PhyLatent: Learning Dynamics-Relevant Representations for JEPA World Models (2026).**  
   arXiv:2608.05720  
   https://arxiv.org/abs/2608.05720

6. **Grozavescu et al., Koopman Autoencoders with Continuous-Time Latent Dynamics (2026).**  
   arXiv:2602.02832  
   https://arxiv.org/abs/2602.02832

7. **Geneva & Zabaras, Transformers for Modeling Physical Systems (Neural Networks, 2022; initial preprint 2020).**  
   OpenReview: YbDGyviJkrL  
   https://openreview.net/forum?id=YbDGyviJkrL

8. **Lusch, Kutz & Brunton, Deep learning for universal linear embeddings of nonlinear dynamics (Nature Communications, 2018).**  
   经典 Koopman autoencoder / nonlinear embedding 工作。

9. **Colbrook et al., An Introductory Guide to Koopman Learning (2025).**  
   arXiv:2510.22002  
   https://arxiv.org/abs/2510.22002

10. **Kim et al., Latent State Design for World Models under Sufficiency ... (2026).**  
    arXiv:2605.01694  
    https://arxiv.org/abs/2605.01694

---

## 62. 一句话实现准则

在后续所有代码设计中，坚持下面这一条即可避免架构跑偏：

\[
\boxed{
\textbf{先让模型学习“什么是可演化的物理状态”，
再让 Koopman 描述其主要演化，
最后让 Attention 只负责无法被结构化动力学闭合的部分。}
}
\]

PhysicsConstraint 定义“什么状态/演化是物理允许的”，JEPA 与 reconstruction 共同塑造 `z_K` 的 predictive/sufficient representation，Koopman 组织主要动力学，Attention 只学习冻结 Koopman 后仍可预测的 memory/closure residual。

