# Koopman-Structured Physical JEPA World Model

> 面向流体、燃烧与一般 PDE 动力系统的结构化潜在世界模型设计文档  
> 目标：将 **JEPA 的潜在预测、Koopman 的结构化动力学、Attention 的跨模态/长时耦合建模** 与 **物理约束、可选择物理解码和控制接口** 统一到一个可直接实现的软件架构中。

> **Revision v2.1**：在 v2.0 的三类 latent 分层架构基础上，新增一套面向 Codex 实际编码的**强工程契约**：统一时间/动作对齐、严格 Batch 数据结构、raw/model 两套物理量语义、EMA 与同坐标 latent target 的分离、closed-loop rollout 防泄漏协议、训练阶段状态机、checkpoint schema、数值精度边界、配置校验、数据指纹、梯度所有权测试与版本级科学验收规则。特别修复一个关键闭环问题：未来时刻没有真实 `z_phys` 时，closure 必须使用 `PhysicalReadout(z_K)` 的预测物理锚点，禁止读取未来真值。

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
- \(a_t\in\mathcal A\)：控制量，例如质量流量、入口压力、阀门开度、喷注参数、外力等；（）
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
\text{Physics grounding}:\ \text{确保 latent 与真实物理量和物理规律相连}
}
\]

\[
\boxed{
\text{Selective decoder}:\ \text{仅在任务需要时恢复完整物理场}
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

## 3. 三类潜在状态：非对称、分层而非并列切片

本框架仍保留三类 latent 的概念：

\[
\boxed{
\mathcal S_t^{latent}
=
\left(
 z_t^{\mathrm{phys}},
 z_t^{K},
 z_t^{R}
\right)
}
\]

但 **v2.0 不再建议** 使用一个 Encoder 输出

\[
E(U_t)=[z_t^{\mathrm{phys}},z_t^K,z_t^R]
\]

然后让三部分一起自由反向传播。这样极易出现三种 representation 重新混合、复制信息，最终退化为一个不可解释的大黑箱。

三类 latent 应具有不同来源、不同训练信号和不同职责：

\[
\boxed{
\begin{aligned}
z_t^{\mathrm{phys}} &: \text{physical anchoring，尽量由已知物理量直接构造};\\
z_t^{K} &: \text{Koopman lifting，学习主要可结构化动力学};\\
z_t^{R} &: \text{closure/memory state，学习结构化模型剩余误差的历史依赖。}
\end{aligned}
}
\]

因此推荐的因果顺序是

\[
\boxed{
\text{Physical anchoring}
\rightarrow
\text{Koopman discovery}
\rightarrow
\text{Residual closure}
\rightarrow
\text{JEPA joint alignment}
}
\]

### 3.1 `z_phys`：显式物理锚点，不要求网络重新发现已知物理

定义一个尽可能确定性的物理映射

\[
\boxed{
z_t^{\mathrm{phys}}
=G_{\mathrm{phys}}(U_t,\mu)
}
\]

其中 `G_phys` 优先使用解析计算、离散积分、统计量、谱分析或已知传感器量，而不是神经网络。

例如二维流体可取

\[
z_t^{\mathrm{phys}}
=
[
M,
P_x,
P_y,
E_k,
\Omega,
C_D,
C_L,
p_{\mathrm{rms}},
f_{\mathrm{dom}},\ldots
].
\]

典型物理量包括

\[
M_t=\int_\Omega \rho\,d\Omega,
\]

\[
P_x=\int_\Omega \rho u\,d\Omega,
\]

\[
E_k=\int_\Omega \frac12\rho |\mathbf u|^2\,d\Omega,
\]

\[
\Omega=\int_\Omega \frac12|\boldsymbol\omega|^2\,d\Omega.
\]

第一版推荐

\[
\boxed{
z_t^{\mathrm{phys}}
=\operatorname{Normalize}(G_{\mathrm{phys}}(U_t))
}
\]

且 `G_phys` **无可训练参数**。这样 physical identifiability 从架构层面成立，而不是依赖 latent 后验解释。

如果后续确实需要一个可学习的 physical branch，可增加

\[
\tilde z_t^{\mathrm{phys}}=E_{\mathrm{phys}}(z_t^{\mathrm{phys}}),
\qquad
\hat q_t=D_{\mathrm{phys}}(\tilde z_t^{\mathrm{phys}}),
\]

并使用

\[
\mathcal L_{\mathrm{phys\text{-}recon}}
=\|\hat q_t-q_t\|_2^2,
\]

但这属于后续扩展，不属于 MVP。

### 3.2 `z_K`：Koopman-structured dynamical latent

定义可学习 lifting/encoder

\[
\boxed{
z_t^K=E_K(U_t,\mu),
\qquad
z_t^K\in\mathbb R^{d_K}.
}
\]

它的职责不是重构所有场细节，而是寻找一个有限维表示，使主要动力学尽量满足结构化演化：

离散时间：

\[
\boxed{
\tilde z_{t+1}^{K}
=K_{\Delta t}(\mu)z_t^K+B_d(\mu)a_t
}
\]

连续时间：

\[
\boxed{
\dot z_t^K=A(\mu)z_t^K+B(\mu)a_t,
\qquad
K_{\Delta t}=e^{A\Delta t}.
}
\]

进一步可扩展为双线性 action coupling：

\[
\dot z_t^K
=A z_t^K+B a_t+
\sum_{j=1}^{d_a}a_t^{(j)}N_j z_t^K.
\]

`z_K` 必须通过**多步动力学一致性、非塌缩和谱诊断**共同训练，而不能只优化一步误差。

建议目标：

\[
\mathcal L_K^{(1)}
=
\left\|
\operatorname{sg}(z_{t+1}^{K,tar})
-\mathcal K_{\Delta t}(z_t^K,a_t)
\right\|_2^2,
\]

\[
\boxed{
\mathcal L_K^{multi}
=
\sum_{k=1}^{H_K}w_k
\left\|
\operatorname{sg}(z_{t+k}^{K,tar})
-
\mathcal K_{\Delta t}^{(k)}(z_t^K,a_{t:t+k-1})
\right\|_2^2.
}
\]

还必须防止平凡解

\[
E_K(U)\equiv 0.
\]

因此至少加入方差约束

\[
\mathcal L_{var}
=
\frac1{d_K}\sum_j
\max\left(0,\sigma_{min}-\sqrt{\operatorname{Var}(z_{:,j}^K)+\epsilon}\right)^2,
\]

必要时再加入弱 covariance regularization。

### 3.3 `z_R`：Residual closure / memory latent，不是第二个自由 Encoder

`z_R` 的定义在 v2.0 中发生关键变化。

不推荐直接写

\[
z_t^R=E_R(U_t).
\]

更合理的定义是：**只有当有限维 Koopman 状态不能完全形成 Markov closure 时，才从历史信息中产生一个 closure/memory state。**

给定历史窗口

\[
\mathcal H_t=
\left\{
 z_{t-H+1:t}^K,
 z_{t-H+1:t}^{\mathrm{phys}},
 a_{t-H+1:t},
 \mu
\right\},
\]

定义

\[
\boxed{
z_t^R=M_\psi(\mathcal H_t)
}
\]

其中 `M_psi` 可以是小型 Transformer、Attention block、SSM 或 GRU；本项目主路线采用小型 Attention/Transformer。

`z_R` 本身不直接等价于物理场状态，而是一个**历史闭合状态**。再通过 residual head 映射到 Koopman latent 空间中的修正：

\[
\boxed{
\Delta z_{t+1}^K=W_R z_t^R.
}
\]

最终动力学为

\[
\boxed{
\hat z_{t+1}^{K}
=
\underbrace{\mathcal K_{\Delta t}(z_t^K,a_t)}_{\text{structured backbone}}
+
\underbrace{g_t\,\Delta z_{t+1}^{K}}_{\text{closure correction}}
}
\]

其中 gate

\[
g_t=\sigma(g_\psi(\mathcal H_t))\in[0,1]
\]

用于抑制 residual branch 无限制接管动力学。

### 3.4 `z_R` 的直接训练目标：学习 Koopman 无法解释的误差

先得到结构化预测

\[
\tilde z_{t+1}^{K}
=\mathcal K_{\Delta t}(z_t^K,a_t).
\]

再由 target encoder 得到未来目标

\[
z_{t+1}^{K,tar}=E_{\bar\theta}(U_{t+1}).
\]

定义 residual target

\[
\boxed{
r_{t+1}^{tar}
=
\operatorname{sg}\left[
 z_{t+1}^{K,tar}-\tilde z_{t+1}^{K}
\right].
}
\]

Attention/closure branch 只学习

\[
\boxed{
\Delta z_{t+1}^{K}
\approx
r_{t+1}^{tar}
}
\]

即

\[
\mathcal L_R
=
\left\|
\Delta z_{t+1}^{K}-r_{t+1}^{tar}
\right\|_2^2.
\]

这里的 `stop-gradient` 是架构核心：在 residual warm-up 阶段，Transformer **不能通过反向传播把 Koopman backbone 故意变差再由自己补偿**。

进一步增加 residual budget：

\[
\boxed{
\mathcal L_{budget}
=
\|g_t\Delta z_{t+1}^K\|_2^2
}
\]

或稀疏版本

\[
\mathcal L_{budget}^{L1}=\|g_t\Delta z_{t+1}^K\|_1.
\]

设计原则为

\[
\boxed{
\text{structured explanation first, neural correction second.}
}
\]

### 3.5 三类 latent 不追求完全统计独立，而追求 functional disentanglement

真实物理量之间本来就是耦合的，因此不要求

\[
z^{\mathrm{phys}}\perp z^K\perp z^R.
\]

真正要求的是：

- `z_phys` 负责**物理锚定**；
- `z_K` 负责**主要可谱分析动力学**；
- `z_R` 负责**未闭合历史效应和非线性残差**。

可以使用较弱的交叉协方差正则避免完全复制：

\[
\mathcal L_{cross}
=
\|\operatorname{Cov}(z^{\mathrm{phys}},z^K)\|_F^2
+
\eta_R\|\operatorname{Cov}(z^K,z^R)\|_F^2,
\]

但该项权重必须较小，不应破坏真实的物理相关性。

### 3.6 三类状态的最终代码语义

推荐在代码中明确区分：

```python
@dataclass
class LatentState:
    z_phys: Tensor   # [B, d_phys], explicit/anchored observables
    z_K: Tensor      # [B, d_K], instantaneous Koopman state
    z_R: Tensor | None  # [B, d_R], history-dependent closure memory
```

其中：

- `z_phys` 由 `PhysicalAnchor.compute()` 得到；
- `z_K` 由 `KoopmanEncoder.forward()` 得到；
- `z_R` 由 `ResidualMemory.forward(history)` 得到；
- 不允许 `latent_split.py` 简单把一个向量切成三段作为最终实现。

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

## 5. Attention residual dynamics：由历史产生 `z_R`，只修正 `z_K`

Attention 不负责从零学习完整演化算子，也不直接在原始物理场上做 full transition。它首先形成历史闭合状态

\[
\boxed{
z_t^R=M_\psi(\mathcal H_t)
}
\]

其中

\[
\mathcal H_t=
\left\{
 z_{t-H+1:t}^{K},
 z_{t-H+1:t}^{\mathrm{phys}},
 a_{t-H+1:t},
 \mu,
 \Delta t
\right\}.
\]

然后将 `z_R` 映射到 Koopman latent 空间中的 closure correction：

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
+g_t\odot\Delta z_{t+1}^{K}
}
\]

其中

\[
g_t=\sigma(G_\eta(z_t^R))\in[0,1].
\]

Residual branch 必须满足两个训练约束：

### 5.1 residual target

\[
r_{t+1}^{tar}
=
\operatorname{sg}\left[
 z_{t+1}^{K,tar}-\tilde z_{t+1}^{K}
\right],
\]

\[
\mathcal L_R
=\|g_t\Delta z_{t+1}^{K}-r_{t+1}^{tar}\|_2^2.
\]

### 5.2 residual budget

\[
\mathcal L_{budget}
=
\frac{
\|g_t\Delta z_{t+1}^{K}\|_2^2
}{
\|\tilde z_{t+1}^{K}\|_2^2+\varepsilon
}.
\]

因此 Attention 的职责被严格限定为：

1. 学习有限维 Koopman closure error；
2. 学习非 Markov 压缩产生的 memory effect；
3. 学习跨 latent mode 的非线性耦合；
4. 在瞬态、模态切换和控制扰动附近提供局部修正。

模型必须持续记录

\[
R_{res}(t)=
\frac{\|g_t\Delta z_{t+1}^{K}\|_2}
{\|\tilde z_{t+1}^{K}\|_2+\epsilon}.
\]

如果稳定区域长期出现

\[
R_{res}\gg1,
\]

则说明结构分工失败，Attention 已经接管主动力学。

## 6. JEPA 训练外壳：主要对齐 Koopman dynamical representation

JEPA 在本框架中的作用不是产生第三个任意 latent，而是为 `z_K` 提供稳定的未来 representation target。

定义 online encoder

\[
z_t^K=E_\theta(U_t,\mu),
\]

以及 target encoder

\[
\boxed{
z_{t+k}^{K,tar}
=
\operatorname{sg}\left[E_{\bar\theta}(U_{t+k},\mu)\right].
}
\]

Target encoder 不直接梯度更新，而使用 EMA：

\[
\boxed{
\bar\theta
\leftarrow
\tau\bar\theta+(1-\tau)\theta
}
\]

其中通常

\[
\tau\rightarrow1.
\]

Koopman + residual predictor 给出

\[
\hat z_{t+k}^{K}.
\]

JEPA loss：

\[
\boxed{
\mathcal L_{JEPA}
=
\sum_{k=1}^{H_J}\omega_k
\,d\left(
\hat z_{t+k}^{K},
 z_{t+k}^{K,tar}
\right).
}
\]

这样 JEPA 的职责是：

- 让 representation 对未来演化有用；
- 避免把训练目标完全绑定在像素/网格重构上；
- 为 Koopman 和 residual closure 提供共同预测空间。

`z_phys` 不需要 JEPA target，因为它由真实物理状态显式计算；`z_R` 也没有独立 target encoder，因为它是由历史生成的 closure memory，其监督来自 residual target。

## 7. 物理锚定与可识别性

仅有 JEPA non-collapse 不足以保证 latent 是真实的“物理状态”。因此定义物理 readout

\[
\hat q_t=H_\omega(z_t),
\]

并施加

\[
\boxed{
\mathcal L_{ground}
=
\|\hat q_t-q(U_t)\|_W^2
}
\]

其中 \(q(U_t)\) 选取少量真正重要的物理量，而不是完整场。

如系统存在守恒量/耗散量 \(I_m(U)\)，可进一步要求

\[
\hat I_m(z_t)=I_m(U_t),
\]

以及

\[
I_m(U_{t+1})-I_m(U_t)=0
\]

或耗散系统中

\[
I_m(U_{t+1})-I_m(U_t)\le0.
\]

若有控制动作，需要避免 counterfactual collapse：同一 \(z_t\) 下明显不同的动作应产生可区分的未来状态

\[
\hat z_{t+1}^{(1)}=P(z_t,a_t^{(1)}),\qquad
\hat z_{t+1}^{(2)}=P(z_t,a_t^{(2)}).
\]

可使用 margin loss：

\[
\mathcal L_{cf}
=
\max\left(
0,
\kappa\|a_t^{(1)}-a_t^{(2)}\|
-
\|\hat z_{t+1}^{(1)}-\hat z_{t+1}^{(2)}\|
\right).
\]

---

## 8. 可选择的物理解码

### 8.1 完整场预测任务

若需要 CFD/FEM surrogate：

\[
\boxed{
\hat U_t=D_\phi(z_t,\mu)
}
\]

并使用

\[
\mathcal L_{field}
=\|W_U(\hat U_t-U_t)\|^2.
\]

必要时增加 PDE residual、BC/IC consistency、flux/conservation loss。

### 8.2 控制/规划任务

若只需要控制，则不必恢复百万维流场，只需

\[
\boxed{
\hat q_t=H_\omega(z_t)
}
\]

并在 latent rollout 上进行 MPC：

\[
\boxed{
\mathbf a^*
=
\arg\min_{a_{t:t+H_c-1}}
\sum_{k=1}^{H_c}
\ell\left(
H_\omega(\hat z_{t+k}),q_{target}
\right)
+\lambda_a\|a_{t+k-1}\|^2
}
\]

或将 latent 直接输入 policy：

\[
a_t\sim\pi_\xi(a|z_t).
\]

因此 decoder 是**任务接口**，不是 JEPA 世界模型成立的必要条件。

---

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
+\lambda_G\mathcal L_{ground}
+\lambda_C\mathcal L_{cross}
+\lambda_P\mathcal L_{physics}
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

### 9.8 Physical grounding

若 `z_phys` 是显式计算量，则不需要对其自身训练；grounding 主要用于检验 `z_K`/combined latent 是否仍能恢复关键可观测量：

\[
\hat q_t=H_\omega(z_t^K,z_t^R),
\]

\[
\mathcal L_{ground}=\|\hat q_t-q_t\|_2^2.
\]

### 9.9 Functional non-redundancy

\[
\mathcal L_{cross}
=
\|\operatorname{Cov}(z^{\mathrm{phys}},z^K)\|_F^2
+\eta_R\|\operatorname{Cov}(z^K,z^R)\|_F^2.
\]

只使用小权重，目的仅是防止完全复制。

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

如果暂时没有 decoder，则优先对可观测量和 latent dynamics 加约束，而不是强行构造全场 PDE residual。

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
U_context : [B, H, C_u, Nx, Ny]
a_context : [B, H, d_a]        # optional
U_future  : [B, Kf, C_u, Nx, Ny]
a_future  : [B, Kf, d_a]       # optional
mu        : [B, d_mu]           # optional
dt        : [B, Kf] or scalar
```

显式物理锚点：

```text
zphys_context : [B, H, d_phys]
zphys_future  : [B, Kf, d_phys]
```

Koopman representation：

```text
zK_context      : [B, H, d_K]
zK_future_target: [B, Kf, d_K]
```

Residual closure memory：

```text
zR_t       : [B, d_R]
delta_zK   : [B, d_K]
gate       : [B, 1] or [B, d_K]
```

注意：`z_R` 不应预先为每个 raw frame 独立编码；它由历史窗口产生。

第一版建议使用全局 latent vector，而不是大量 spatial tokens。只有当 V1.0 已证明全局结构成立后，才进入 spatial Koopman token / region token。

## 11. 推荐的最小可行模型（MVP）

最终研究 baseline 的最小结构是：

\[
\boxed{
G_{phys}
+
E_K
+
\text{EMA target}
+
\text{continuous-time Koopman core}
+
\text{small residual memory/Attention}
+
\text{physical readout}
}
\]

但**工程开发的第一个版本不是这个完整 MVP**。必须按第三部分 V0.1–V1.0 逐步组装。

暂时不做：

- 大型 field decoder；
- RL；
- mixture-of-Koopman experts；
- stochastic latent；
- 多尺度 spatial token hierarchy；
- 3D combustion/detonation。

V1.0 必须回答四个科学问题：

1. `z_K` 是否真的形成比普通 representation 更稳定的 long-horizon dynamics？
2. JEPA target 是否改善 learned lifting 的 predictive quality，而不是只改善训练表面稳定性？
3. `z_R` / Attention 是否主要在 Koopman closure 失败处起作用，而不是接管主动力学？
4. physical anchors/readout 是否证明 latent 的变化仍对应真实物理状态变化？

---

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

因此本框架必须加入 physics grounding、action counterfactual consistency 与 Koopman dynamics constraint。

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

## 17. 为什么三类 latent 必须采用不同训练机制

三类 latent 的划分不是说真实物理世界天然严格分成三组变量，而是建立一种**可审计的内部状态设计**。

如果只使用单一自由 latent，容易出现：

- Koopman branch 只保留“容易线性传播但没有物理信息”的变量；
- residual network 复制全部状态并接管预测；
- reconstruction loss 迫使 representation 保存大量任务无关细节；
- physical probe 只能事后拟合，而不能证明 state representation 真正 grounded。

因此三类状态必须使用不同学习机制。

### 17.1 `z_phys`：锚定出来，而不是自由学习出来

\[
\boxed{
z^{phys}=G_{phys}(U)
}
\]

它回答：

> 当前 latent model 是否仍然和质量、能量、频率、涡量、力系数等真实物理量保持联系？

第一版固定，不参与梯度更新。

### 17.2 `z_K`：通过“可演化性”学出来

\[
\boxed{
z^K=E_K(U),
\qquad
z_{t+1}^K\approx\mathcal K(z_t^K)
}
\]

它回答：

> 哪组内部坐标最适合描述系统的主导频率、增长/衰减和长期演化？

其训练信号来自 Koopman consistency、multi-step rollout、JEPA target、non-collapse 和 physical grounding。

### 17.3 `z_R`：通过“结构模型剩余误差”学出来

\[
\boxed{
z_t^R=M_\psi(\mathcal H_t)}
\]

它不是另一个 frame encoder，而是 history-dependent closure state。

它回答：

> 有限维 Koopman 状态还漏掉了什么记忆效应、非线性耦合和瞬态信息？

训练目标直接来自

\[
r_{t+1}^{tar}=\operatorname{sg}(z_{t+1}^{K,tar}-z_{t+1}^{K,base}).
\]

对于模态切换问题，`R_res(t)`、gate 或 `||z_R||` 有可能进一步成为 transition indicator，但这必须通过实验验证，不能预先假定。

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

ResidualMemory 的输入不应是 raw CFD field，也不应重复做一个大型 spatial encoder。

推荐历史 token：

\[
\boxed{
\xi_i
=P_\xi\left(
[z_i^K,z_i^{phys},a_i,\mu,\Delta t_i]
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

## 24. Physical grounding 如何选择

不要把所有网格变量都作为 grounding target，否则又退化成 reconstruction。

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

## 25. 物理约束放在什么位置

推荐分为三层。

### 层 A：latent grounding

\[
H(z)\approx q(U).
\]

成本最低，第一版必须做。

### 层 B：latent dynamics constraints

例如 Koopman 谱约束、已知 invariant 的 latent readout consistency。

\[
I_z(\hat z_{t+1})\approx I_z(z_t).
\]

### 层 C：decoded field constraints

只有有 decoder 时才计算：

\[
\mathcal R_{PDE}(\hat U)=0,
\]

BC/IC、flux、mass/energy conservation 等。

不建议第一版把昂贵 PDE residual 当作主要训练信号，否则难以判断性能提升来自结构化 latent 还是 PINN-style regularization。

---

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
WorldModel.read_physics()
```

写成三个独立接口。

---

## 27. 推荐训练流程：严格分层训练，不允许一开始端到端“一锅训练”

训练策略本身属于架构设计的一部分。核心原则：

\[
\boxed{
\text{anchor first}
\rightarrow
\text{Koopman first}
\rightarrow
\text{closure second}
\rightarrow
\text{joint fine-tune last}
}
\]

### Stage 0：数据、物理锚点与基准

先完成：

1. trajectory-level train/val/test 划分；
2. window sampler；
3. `PhysicalAnchor` 及其 normalization；
4. baseline：Persistence、DMD/POD、AE+MLP/Transformer、Koopman-only；
5. 固定随机种子和配置保存机制。

此阶段不训练世界模型。

### Stage 1：直接验证 KoopmanCore，本阶段不要 learned encoder

先使用已知低维状态或合成动力系统，例如 damped oscillator / Duffing / Lorenz 的真实状态：

\[
x_t\rightarrow A/K\rightarrow \hat x_{t+k}.
\]

目的：验证

- `matrix_exp` / discrete propagation；
- irregular `dt`；
- eigenvalue/frequency extraction；
- multi-step rollout；
- gradient；
- checkpoint/reload。

如果这一阶段失败，禁止进入 field encoder。

### Stage 2：训练 Koopman encoder `E_K`

加入

\[
z_t^K=E_K(U_t).
\]

Attention residual 保持关闭：

\[
\Delta z=0.
\]

第一轮可使用 lightweight reconstruction/readout 防止 representation 完全自由漂移：

\[
\hat U_t^{low}=D_{warm}(z_t^K)
\]

或物理 probe：

\[
\hat q_t=H(z_t^K).
\]

优化

\[
\mathcal L_{K}^{(1)}
+\lambda_M\mathcal L_{K,multi}
+\lambda_V\mathcal L_{var}
+\lambda_G\mathcal L_{ground}.
\]

目标：证明 `z_K` 自身已经具有可用的 open-loop dynamics。

### Stage 3：引入 JEPA online/target encoder

建立：

\[
E_\theta\quad\text{online},
\qquad
E_{\bar\theta}\quad\text{EMA target}.
\]

更新：

\[
\bar\theta\leftarrow\tau\bar\theta+(1-\tau)\theta.
\]

此时仍然关闭 residual branch，只做：

\[
\mathcal L_{JEPA}
+\mathcal L_K
+\mathcal L_{K,multi}
+\mathcal L_{var}
+\mathcal L_{ground}.
\]

目的是先证明 **JEPA target + Koopman backbone** 可以稳定训练，而不是一次性引入 Attention。

### Stage 4：Residual target 机制验证，先不用 Transformer

冻结或近似冻结：

- Koopman encoder；
- Koopman core；
- target encoder。

进入本阶段时先执行 `target <- hard_copy(online)`，随后冻结 online/target/Koopman 并暂停 EMA。Residual target 使用**同坐标 online latent**：

\[
\boxed{
r_{t+1}^{tar}
=\operatorname{sg}[E_\theta(U_{t+1})]
-\operatorname{sg}[\tilde z_{t+1}^K].
}
\]

EMA teacher 仅保留为 JEPA 语义，不参与 residual 差分。先用一个极小的 MLP/linear residual head 从简单 history summary 预测 residual。

目的不是追求性能，而是验证：

- residual target 是否数值合理；
- stop-gradient 是否正确；
- residual branch 是否只补偿 structured model；
- `C_closure` / `R_closure` 指标是否能正常记录。

这一版通过后才换 Attention。

### Stage 5：Attention closure / `z_R`

将 residual MLP 替换为历史依赖模块：

\[
z_t^R=M_\psi(\mathcal H_t),
\qquad
\Delta z_{t+1}^K=W_R z_t^R.
\]

Koopman branch 继续冻结或使用极小学习率。

优化

\[
\boxed{
\mathcal L_{stage5}
=\mathcal L_R
+\lambda_B\mathcal L_{budget}
+\lambda_{gate}\mathcal L_{gate}.
}
\]

Residual head 使用 near-zero small initialization，gate bias 初始设为负值，使

\[
g_t\ll 1
\]

开始训练，同时避免严格全零输出层在首步完全阻断 memory branch 的梯度。ResidualMemory 使用 causal mask，并将真实 `dt` 作为 token 条件。

从本阶段起必须实现 closed-loop rollout：context 之后的 `z_phys` 由 `PhysicalReadout(\hat z_k)` 生成，禁止读取未来真实物理锚点。

### Stage 6：联合微调

只有前五个阶段都通过后，才联合优化：

\[
\mathcal L_{total}.
\]

学习率建议满足

```text
LR_KoopmanEncoder << LR_ResidualAttention
LR_KoopmanCore    << LR_ResidualAttention
LR_TargetEncoder = 0  # EMA only
PhysicalAnchor   = frozen
```

例如数量级：

```text
Koopman encoder : 1e-5
Koopman core    : 1e-5
Residual attn   : 1e-4
Readout/decoder : 1e-4
```

目标是防止 joint fine-tune 把已经学到的谱结构“洗掉”。本阶段恢复 EMA，并加入 closed-loop rollout horizon curriculum；所有冻结/解冻通过统一 `TrainStage` 状态机配置，阶段切换默认重新创建 optimizer/scheduler。

### Stage 7：任务头

根据任务选择：

#### A. 物理量/控制任务

只加入

\[
H_{phys}(z_K,z_R)\rightarrow q.
\]

#### B. 全场 surrogate

再加入

\[
D_{field}(z_K,z_R,\mu)\rightarrow \hat U.
\]

#### C. Action-conditioned control

最后才加入

\[
B a_t,
\quad
\sum_j a_jN_jz,
\quad
\mathcal L_{cf},
\quad
MPC/RL.
\]

不要在第一个可运行版本中同时做完整场 decoder 和 RL。

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
│   │   ├── anchors.py
│   │   ├── observables.py
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
│   │   ├── physical_readout.py
│   │   ├── field_decoder.py
│   │   └── world_model.py
│   ├── losses/
│   │   ├── koopman.py
│   │   ├── jepa.py
│   │   ├── collapse.py
│   │   ├── residual.py
│   │   ├── grounding.py
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
    ├── test_physical_anchor.py
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
    └── test_closed_loop_rollout_history_update.py
```

设计要求：

1. `physics/` 不依赖神经网络；
2. `koopman_core.py` 不依赖 JEPA 或 Transformer；
3. `residual_memory.py` 不允许内部直接调用 Koopman core；
4. `world_model.py` 只负责编排，不堆放具体数学实现；
5. 每个版本新增模块时，不允许无理由重写已通过测试的旧模块；
6. 所有模块必须可以单独实例化和单元测试。

## 29. 推荐接口契约

本节属于**公共 API 规范**。数学文中可以写 `z^K`，Python API 一律使用 `z_k`。

### 29.1 PhysicalAnchor

```python
class PhysicalAnchor:
    def compute(self, state_raw, spec, metadata=None):
        """Raw physical state -> explicit physical observables z_phys."""
        ...
```

输入/输出：

```text
state_raw : [B,C,*spatial]，必须为真实物理单位
z_phys    : [B,d_phys]
```

`PhysicalAnchor` 不接收 normalized `state_model`。

### 29.2 KoopmanEncoder

```python
class KoopmanEncoder(nn.Module):
    def forward(self, state_model, mu_static=None):
        """Normalized field/state -> instantaneous Koopman latent z_k."""
        ...
```

### 29.3 KoopmanCore

```python
class KoopmanCore(nn.Module):
    def step(self, z_k, action=None, dt=None, mu_static=None):
        """One structured latent step U_i --(a_i,dt_i)--> U_{i+1}."""
        ...

    def rollout(self, z_k0, future_actions=None, future_dts=None,
                horizon=None, mu_static=None):
        """Pure Koopman open-loop rollout; no residual/teacher access."""
        ...

    @torch.no_grad()
    def spectrum(self):
        """Detached continuous/discrete spectral diagnostics."""
        ...
```

### 29.4 PhysicalReadout

```python
class PhysicalReadout(nn.Module):
    def forward(self, z_k):
        """Predicted z_k -> normalized/predicted z_phys used for grounding and closed-loop history."""
        ...
```

训练指标最终必须 inverse-transform 回 raw physical units。

### 29.5 ResidualMemory

```python
class ResidualMemory(nn.Module):
    def forward(
        self,
        z_k_hist,
        z_phys_hist,
        history_actions=None,
        history_dts=None,
        current_action=None,
        current_dt=None,
        mu_static=None,
    ):
        """Causal history -> closure memory z_r."""
        ...
```

`current_action/current_dt` 对应最后一个历史状态到待预测状态的 transition。Variable-`dt` 数据时 `current_dt` 不得省略。

### 29.6 ResidualHead

```python
class ResidualHead(nn.Module):
    def forward(self, z_r):
        """z_r -> delta_z_k [B,d_k], gate [B,1]."""
        ...
```

### 29.7 StructuredPhysicalJEPA

```python
class StructuredPhysicalJEPA(nn.Module):
    def encode_koopman(self, state_model, mu_static=None):
        ...

    def compute_physical_anchor(self, state_raw, spec, metadata=None):
        ...

    def base_step(self, z_k, action, dt, mu_static=None):
        ...

    def closure_step(
        self, z_k_hist, z_phys_hist,
        history_actions, history_dts,
        current_action, current_dt,
        mu_static=None,
    ):
        ...

    def transition(self, *, z_k_hist, z_phys_hist,
                   history_actions, history_dts,
                   current_action, current_dt,
                   mu_static=None):
        """Return base/residual/gate/combined next z_k separately."""
        ...

    def rollout_closed_loop(self, batch, horizon=None):
        """No future-state access; future z_phys comes from PhysicalReadout."""
        ...

    def rollout_teacher_forced(self, batch, horizon=None):
        """Diagnostics only; never used as official forecasting metric."""
        ...
```

`transition()` 禁止只返回一个 tensor：

```python
@dataclass
class TransitionOutput:
    z_k_base: Tensor
    z_r: Tensor | None
    delta_z_k: Tensor
    gate: Tensor             # [B,1]
    z_k_next: Tensor
```

这样训练日志能够明确知道最终性能来自 Koopman backbone 还是 neural closure。

### 29.8 TrainingStage API

```python
class TrainStage(Enum):
    KOOPMAN = "koopman"
    JEPA = "jepa"
    RESIDUAL = "residual"
    JOINT = "joint"


def configure_trainable(model, stage: TrainStage) -> None:
    ...


def assert_optimizer_matches_trainable_params(model, optimizer) -> None:
    ...
```

冻结语义只能从这里产生，训练脚本不自行改 `requires_grad`。

## 30. 分阶段训练伪代码

以下伪代码只表达**梯度与 target 语义**；实际实现统一接收 `ProblemBatch`。

### 30.1 Koopman-only：同坐标 online dynamics

```python
z_k_t = online_encoder(U_t_model)
z_k_next_online = online_encoder(U_next_model)

z_k_base = koopman_core.step(
    z_k_t,
    action=current_action,
    dt=current_dt,
    mu_static=mu_static,
)

# Koopman representation learning may backprop through both online encodings.
loss = (
    lambda_k * koopman_one_step(z_k_base, z_k_next_online)
    + lambda_m * koopman_multistep_closed_or_encoded_targets(...)
    + lambda_v * variance_loss(z_k_t)
    + lambda_g * grounding_loss(
        physical_readout(z_k_t),
        z_phys_t_normalized,
    )
)

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

本阶段没有 EMA teacher。

### 30.2 JEPA + Koopman：EMA target 只用于 JEPA

```python
z_k_ctx = online_encoder(U_context_model)
z_k_future_online = online_encoder(U_future_model)   # same-coordinate dynamics target

with torch.no_grad():
    z_k_future_jepa = target_encoder(U_future_model) # EMA JEPA target only

z_k_pred = koopman_core.rollout(
    z_k_ctx[:, -1],
    future_actions=future_actions,
    future_dts=future_dts,
    horizon=K,
    mu_static=mu_static,
)

loss = (
    lambda_j * jepa_loss(z_k_pred, z_k_future_jepa)
    + lambda_k * koopman_consistency(z_k_pred, z_k_future_online)
    + lambda_v * variance_loss(z_k_ctx)
    + lambda_g * grounding_loss(...)
)

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
update_ema_after_optimizer_step(target_encoder, online_encoder)
```

### 30.3 Residual warm-up：hard-sync 后全部结构分支冻结

进入阶段时：

```python
hard_sync(target_encoder, online_encoder)
pause_ema()
configure_trainable(model, TrainStage.RESIDUAL)
```

单步 residual target：

```python
with torch.no_grad():
    z_k_hist = online_encoder(U_context_model)
    z_k_next_same_coord = online_encoder(U_next_model)
    z_k_base = koopman_core.step(
        z_k_hist[:, -1], current_action, current_dt, mu_static
    )
    residual_target = z_k_next_same_coord - z_k_base

z_phys_hist = physical_anchor.compute(U_context_raw, problem_spec)
z_r = residual_memory(
    z_k_hist,
    z_phys_hist,
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

residual_optimizer.zero_grad(set_to_none=True)
loss.backward()
residual_optimizer.step()
```

这里 residual target 两端都没有梯度，EMA teacher 不参与 target。

### 30.4 Closed-loop rollout：正式预测不能读取未来状态

```python
z_k_hist = online_encoder(batch.context_states_model)
z_phys_hist = physical_anchor.compute(batch.context_states_raw, problem_spec)

predictions = []
for k in range(horizon):
    current_action = None if batch.future_actions is None else batch.future_actions[:, k]
    current_dt = batch.future_dts[:, k]

    out = model.transition(
        z_k_hist=z_k_hist,
        z_phys_hist=z_phys_hist,
        history_actions=history_actions_for_current_queue,
        history_dts=history_dts_for_current_queue,
        current_action=current_action,
        current_dt=current_dt,
        mu_static=batch.mu_static,
    )

    z_k_next = out.z_k_next
    z_phys_next = physical_readout(z_k_next)  # NOT true future anchor

    predictions.append(z_k_next)
    z_k_hist = append_and_crop(z_k_hist, z_k_next)
    z_phys_hist = append_and_crop(z_phys_hist, z_phys_next)
    update_action_dt_history(...)
```

`batch.future_states_*` 只允许在循环完成后用于 loss/metric，不得进入上述 transition history。

### 30.5 Joint fine-tune

```python
configure_trainable(model, TrainStage.JOINT)
hard_sync(target_encoder, online_encoder)
resume_ema()

# full_pred is generated by rollout_closed_loop; no future-state input to transition.
full_pred = model.rollout_closed_loop(batch, horizon=current_curriculum_horizon)

with torch.no_grad():
    jepa_target = target_encoder(batch.future_states_model[:, :current_horizon])

# same-coordinate target for Koopman/residual diagnostics
z_k_future_online = online_encoder(batch.future_states_model[:, :current_horizon])

loss = (
    lambda_j * jepa_loss(full_pred.z_k, jepa_target)
    + lambda_k * koopman_structure_loss(full_pred.z_k_base, z_k_future_online)
    + lambda_r * residual_target_loss(...same-coordinate detached targets...)
    + lambda_b * closure_budget(...)
    + lambda_g * grounding_loss(...)
    + lambda_v * variance_loss(...)
)

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
update_ema_after_optimizer_step(target_encoder, online_encoder)
```

联合训练时仍然分别记录 `z_k_base` 和 `z_k_next` 的误差。正式 long-horizon 指标来自 `rollout_closed_loop()`。

## 31. 必做消融实验

为了证明研究贡献来自“结构化 latent”而不是模型参数更多，至少比较：

| ID | Encoder/Training | Dynamics | Residual | Physics grounding |
|---|---|---|---|---|
| B0 | AE | MLP | No | No |
| B1 | AE | Transformer | Full | No |
| B2 | JEPA | Transformer | Full | No |
| B3 | JEPA | Koopman | No | No |
| B4 | JEPA | Koopman | Attention residual | No |
| B5 | JEPA | Koopman | Attention residual | Yes |
| B6 | Physics-JEPA | Koopman | Attention residual | Yes + constraints |

核心不是只报告一步 MSE，而是报告：

\[
\boxed{
\text{short error}
+\text{long rollout}
+\text{spectrum}
+\text{physics}
+\text{control}
}
\]

---

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

这直接支持本框架加入：

\[
\mathcal L_{ground}+\mathcal L_{cf}+\mathcal L_{physics-latent}.
\]

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
+\text{explicit physical identifiability}
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
| physical identifiability collapse | 不同物理状态 latent 不可区分 | physical grounding |
| counterfactual collapse | 不同 action 未来 latent 过近 | action-conditioned Koopman + counterfactual loss |
| reconstruction bias | latent 保留细节而非动力学 | JEPA prediction objective |
| finite Koopman closure | 固定 \(K\) 无法解释复杂瞬态 | residual latent + Attention closure |
| spectral instability | rollout 爆炸/坍缩 | structured generator / bounded spectrum |
| long-horizon error accumulation | 单步准、长期错 | multi-step open-loop curriculum |
| attention takeover | Koopman 变成装饰 | gated residual + residual budget + staged training |
| latent/physics metric mismatch | latent 近不代表物理近 | physical probe/readout |
| generalization across parameters | 新 Re/Ma/geometry 失效 | parameter-conditioned spectral core |
| full-field decoding expensive | 控制效率差 | selective decoder |
| partial observation | 传感器不构成 Markov state | context encoder / latent memory（后续） |

---

## 36. 第一版超参数建议

仅作为启动值：

```yaml
latent:
  d_total: 128
  d_phys: 16
  d_koopman: 64
  d_residual: 48

context:
  history: 16
  future_horizon_start: 1
  future_horizon_max: 16

koopman:
  continuous_time: true
  oscillator_blocks: 32
  parameter_conditioned: false
  bilinear_action: false

attention:
  d_model: 256
  layers: 3
  heads: 4
  dropout: 0.05
  residual_gate_init_bias: -3.0

jepa:
  ema_tau_start: 0.99
  ema_tau_end: 0.9999

loss:
  lambda_jepa: 1.0
  lambda_koopman: 1.0
  lambda_multistep: 0.5
  lambda_ground: 0.2
  lambda_residual_budget: 1e-3
  lambda_disentangle: 1e-3
  lambda_physics: 0.0   # first MVP
  lambda_counterfactual: 0.0  # enable for control phase
  lambda_field: 0.0     # enable when decoder is introduced
```

这些权重不是最终答案，应以 gradient scale 和 validation behavior 调整，而不是机械沿用。

---

## 37. 必须记录的训练诊断

每个 epoch/validation 至少记录：

```text
loss/jepa
loss/koopman
loss/multistep
loss/ground
loss/residual_budget

latent/variance_min
latent/variance_mean
latent/cov_K_R

koopman/alpha_min_mean_max
koopman/omega_min_mean_max
koopman/spectral_radius_discrete

residual/norm_ratio_mean
residual/gate_mean
residual/gate_p95

forecast/error_1
forecast/error_4
forecast/error_8
forecast/error_16

physics/frequency_error
physics/energy_error
physics/observable_error
```

否则很容易得到一个“预测 MSE 看起来很好”但结构完全退化的模型。

---

## 38. 关键失败模式与调试顺序

### 失败 A：latent variance 接近 0

说明 representation collapse。

优先检查：

1. EMA tau；
2. target stop-gradient；
3. batch normalization/latent normalization；
4. variance regularization；
5. grounding loss 是否太弱。

### 失败 B：Koopman-only 很差，Attention 打开后突然很好

可能 Attention 完全接管。

检查：

\[
R_{burden},\quad g_t,
\]

并增加 residual penalty、降低 Transformer 容量、延长 Koopman warm-up。

### 失败 C：一步预测很好，16 步后爆炸

检查：

1. \(\alpha_i\)/谱半径；
2. non-normal action matrices；
3. residual feedback amplification；
4. 是否真正做 open-loop training；
5. rollout 中是否误用了 teacher forcing。

### 失败 D：latent error 小但物理量错

说明 latent metric 与 physics 不一致。

增加 physical grounding/readout，并以 physical observables 作为 validation 选择标准之一。

### 失败 E：稳定态好，模态切换错

这并不一定意味着架构失败，可能恰好说明有限 Koopman closure 不足。

检查 residual gate 是否在 transition 前后升高。如果会升高但预测仍错，可：

- 增大 history；
- 引入 regime-conditioned/mixed Koopman；
- 增加 residual latent；
- 使用 event-aware loss。

---

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
\textbf{A physically grounded predictive latent-state architecture with}
\textbf{spectrally structured dynamics and attention-based closure.}
}
\]

对应中文：

> **一种具有物理锚定、预测性潜在表征、谱结构动力学和 Attention 闭合修正的物理世界模型。**

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
V0.2  数据窗口 + PhysicalAnchor
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

- `raw`：有真实物理单位，用于 `PhysicalAnchor`、物理损失、最终指标；
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

`PhysicalAnchor` 根据 channel name/ProblemSpec 查找物理变量。缺少必需变量时应明确报错或跳过该 observable，禁止悄悄用错误 channel。

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

当前模型的 closure 输入包含 `z_phys`，但未来真实 `U_{t+k}` 在推理时不可用。因此必须定义两种 rollout：

**Teacher-forced rollout（只用于诊断）**：允许使用真实未来 encoded state / physical anchor；不得作为正式 forecasting 指标。

**Closed-loop rollout（正式评估）**：context 结束后只能使用：

- 自己预测得到的 `z_k`；
- 已知 future action / `dt` / static parameters；
- 由 `PhysicalReadout` 从预测 `z_k` 得到的

\[
\boxed{
\hat z_{t+k}^{phys}=H_{phys}(\hat z_{t+k}^{K})
}
\]

然后把 `\hat z_phys` 和 `\hat z_k` 重新压入历史队列，供下一步 closure 使用。

正式 rollout 禁止访问：

```text
future_states_raw
future_states_model
true future z_phys
true future encoded z_K
```

除非只是计算 loss/metric，且该 tensor 不进入下一步预测路径。

必须增加 `test_no_future_leakage.py`：将未来真值随机打乱后，closed-loop prediction 必须保持不变。

### 42.15 Attention 必须是因果的，并显式条件化真实时间间隔

ResidualMemory 不是普通 bidirectional encoder。第一版强制 causal mask。

若 `dt` 可变，token 必须包含与状态到下一状态对应的 `dt`：

\[
\boxed{
\xi_i=P([z_i^K,z_i^{phys},a_i,\mu,\Delta t_i])
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
physical_anchor_spec
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

数学记号仍使用 `z^K, z^{phys}, z^R`。Python 代码统一 snake_case：

```text
z_k
z_phys
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
- 跨模型正式比较优先使用 raw-unit physical observables、field error、频率/谱、控制性能；
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
test_raw_vs_model_anchor_semantics.py
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
    z_phys: Tensor | None
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

## 44. V0.2 — Data Windows & Physical Anchors

### 目标

建立时间窗数据协议和确定性 `z_phys`。

### 输入协议

严格遵守第 42.6 节时间语义。模型接收标准 `ProblemBatch`：

```text
context_states_raw/model : [B,H,C,Nx,Ny]
future_states_raw/model  : [B,K,C,Nx,Ny]
history_actions          : [B,H-1,d_a]   # optional
future_actions           : [B,K,d_a]     # optional; element 0 drives U_t -> U_{t+1}
history_dts              : [B,H-1]
future_dts               : [B,K]
mu_static                : [B,d_mu]       # optional
```

底层 trajectory 使用 `[T+1] states + [T] transitions` 语义，不允许另建含糊的 action 对齐方式。

### 新增模块

```text
src/data/datasets.py
src/data/windows.py
src/data/splits.py
src/data/normalization.py
src/physics/anchors.py
src/physics/observables.py
```

### 第一版 PhysicalAnchor

先支持通用统计量与简单二维流体 observables：

- channel mean/std/RMS；
- total/mean kinetic energy（若变量定义允许）；
- vorticity RMS / enstrophy（规则网格）；
- 用户可注册 custom observable。

### 关键要求

`PhysicalAnchor` 无 trainable parameter：

\[
z^{phys}=G_{phys}(U_{raw}).
\]

特别强调：`PhysicalAnchor` 只能读取 raw-unit state；neural encoder 使用 normalized `U_model`。normalizer 单独 fit 在 train split，val/test 禁止重新 fit。积分型 observable 必须使用坐标/网格权重或明确声明等距规则网格假设。

V0.2 必须同时输出并保存 `split_manifest`、`data_fingerprint`、`normalizer_state`。

### 测试

- window 不跨 trajectory；
- train/val/test 无 trajectory leakage；
- constant field 的 gradient observable 为 0；
- normalizer inverse transform；
- shape tests。

### 验收

生成一个 toy dataset，打印：

```text
U_context.shape
U_future.shape
z_phys.shape
train/val/test trajectory ids
```

结果必须可复现。

---

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

## 46. V0.4 — Learned KoopmanEncoder on Low-Dimensional/Synthetic Data

### 目标

引入 `E_K`，验证“learned lifting + Koopman”本身可训练且不塌缩。

### 新增模块

```text
src/models/koopman_encoder.py
src/losses/koopman.py
src/losses/collapse.py
src/models/physical_readout.py
src/train/train_koopman.py
```

### 训练目标

\[
\mathcal L
=\lambda_K\mathcal L_K
+\lambda_M\mathcal L_{K,multi}
+\lambda_V\mathcal L_{var}
+\lambda_G\mathcal L_{ground}.
\]

第一版可增加一个小 reconstruction warm-up head，但必须是可选配置。

### 必须记录

- `z_K` 每维 mean/std；
- covariance condition number；
- one-step error；
- 16/32/64-step rollout error；
- learned spectrum。

### 验收

- `z_K` 不塌缩；
- Koopman-only rollout 显著优于 persistence；
- 关闭 encoder 后 V0.3 测试仍全部通过。

---

## 47. V0.5 — 2D/PDE Koopman-Only Baseline

### 目标

将结构扩展到规则网格物理场，但**仍然不使用 JEPA 和 Attention**。

### Encoder

第一版默认小型 CNN encoder；FNO encoder 作为可选后端，不要同时实现两个复杂版本。

\[
U_t\xrightarrow{E_K}z_t^K.
\]

### 新增/修改

```text
src/models/koopman_encoder.py   # CNN backend
configs/data/pde2d.yaml
configs/model/koopman_only.yaml
scripts/evaluate_rollout.py
```

### 推荐测试问题

分成两层：

1. **仓库内置 smoke dataset（必须）**：二维周期 advection-diffusion 的小型确定性场序列，用于 CI 与接口验证，不需要联网或外部 CFD；
2. **research dataset（研究者显式提供）**：优先 cylinder wake、2D Navier–Stokes 或 reaction-diffusion。

Codex 不得自动联网下载数据，也不得为了 smoke test 临时实现一个复杂 CFD solver。不要第一版直接进入爆轰。

### 训练

Residual branch 不存在：

\[
\hat z_{t+k}^K=K^kz_t^K.
\]

使用 physical readout：

\[
H(z_K)\rightarrow \hat q.
\]

### 验收

至少比较：

- persistence；
- DMD/POD（若可实现）；
- learned Koopman。

必须报告：

- latent rollout；
- observable error；
- frequency/spectrum；
- `z_K` variance。

---

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

## 49. V0.7 — Residual Target Infrastructure + Tiny Closure Baseline

### 目标

先验证 residual target 和 stop-gradient 机制，不急着上 Transformer。

### 新增模块

```text
src/losses/residual.py
src/metrics/residual_burden.py
src/models/residual_head.py
src/train/train_residual.py
```

### 核心 target

Residual target 必须与 `z_k_base` 处于同一个 online latent 坐标系：

\[
\boxed{
r_{t+1}^{tar}
=\operatorname{sg}\left[E_\theta(U_{t+1})\right]
-\operatorname{sg}\left[z_{t+1}^{K,base}\right]
}
\]

不要用 EMA teacher latent 直接做差。进入 V0.7 时先 hard-sync `target <- online`，随后 online/target/Koopman 全部冻结并暂停 EMA，直到 V0.9。

### Closure baseline

只使用一个极小 MLP：

```text
[zK_t, zphys_t, a_t] -> delta_zK
```

不使用历史 Attention。

### 冻结策略

```text
online/target encoder : frozen
Koopman core          : frozen
MLP residual          : trainable
```

### 测试

- residual target 无 gradient；
- 冻结参数 optimizer step 后不变；
- `delta_zK=0` 时结果严格等于 Koopman-only；
- residual burden metric 正确。

### 验收

要求证明：

1. MLP correction 可以降低部分 one-step residual；
2. 必须记录 `C_closure/R_closure`，并检查 correction 是否在大部分稳定区异常主导；
3. Koopman-only 指标没有被修改。

---

## 50. V0.8 — Attention Closure / History-Dependent `z_R`

### 目标

正式引入第三个 latent：

\[
\boxed{
z_t^R=M_\psi(\mathcal H_t)}.
\]

### 新增模块

```text
src/models/residual_memory.py
src/models/gate.py
configs/model/full_closure.yaml
```

### 第一版 Transformer 约束

只允许小模型，例如：

```text
d_model: 64 or 128
layers : 2-3
heads  : 4
history: 8-16
```

不要做大型 Transformer。

### 输入 token

每个时间步拼接/投影：

\[
\boxed{
\xi_t=P([z_t^K,z_t^{phys},a_t,\mu,\Delta t_t])
}
\]

对于无控制系统去掉 `a_t`；固定 `dt` 仍保留接口但可输入常数。ResidualMemory 必须使用 causal mask。

历史：

\[
[\xi_{t-H+1},\ldots,\xi_t]
\xrightarrow{Attention}
z_t^R.
\]

### Closed-loop 历史更新

一步训练可以使用真实 context 的 `z_phys`。但 multi-step rollout 进入未来后，禁止从 `U_future` 重新计算 `z_phys`。使用已在 V0.4/V0.5 训练的 `PhysicalReadout`：

\[
\hat z_{t+k}^{phys}=H_{phys}(\hat z_{t+k}^K),
\]

并把 `\hat z_k, \hat z_phys, future_action, future_dt` 压回历史队列。正式 forecasting 指标只允许这种 closed-loop 模式。

### 输出

\[
\Delta z_{t+1}^K=W_Rz_t^R,
\qquad
g_t=\sigma(w_g^Tz_t^R+b_g).
\]

第一版 shape 固定：

```text
z_r        : [B,d_r]
delta_z_k  : [B,d_k]
gate       : [B,1]     # scalar gate
```

初始化要求：

- `W_R` 使用 near-zero small initialization（优先于严格全零，避免第一步完全阻断 memory gradient）；
- `b_g<0`，建议使初始 gate 落在约 0.05--0.15；
- vector gate 留到 V2.x。

### 训练

仍然冻结 Koopman branch：

\[
\mathcal L=\mathcal L_R+\lambda_B\mathcal L_{budget}.
\]

### 验收

必须画出/保存：

- `C_closure(t)` 与 `R_closure(t)`；
- gate 随时间；
- Koopman base error；
- corrected error；
- residual 与瞬态/模态变化的对应关系；
- teacher-forced 与 closed-loop rollout 的差距。

必须通过 causal-mask 与 no-future-leakage 测试。`C_closure` 应被报告为分布/时间序列，不设置跨所有物理系统通用的硬阈值；如果 correction 长期主导全部增量，则标记为结构风险并进入人工科学审查。

---

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

每一步未来历史都由模型自己的 `z_k_next` 与 `PhysicalReadout(z_k_next)` 更新，禁止 teacher forcing 作为正式 rollout loss 的唯一来源。

### loss

开启：

- JEPA；
- Koopman consistency；
- multi-step；
- residual target；
- residual budget；
- non-collapse；
- physical grounding。

暂不强制加入复杂 PDE residual。

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
- physical observables 更准确或至少不恶化；
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
use_physical_grounding
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

## 53. V1.1 — Selective Physical Projection / Field Decoder

### 目标

解决 latent 到物理空间的映射，但不改变 latent dynamics 本体。

两种 readout：

### A. 低维 observables

\[
H_{phys}(z_K,z_R)\rightarrow q.
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
\ell(H_{phys}(z_{t+k}),q_{target})
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
v0.2-data-anchors
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



## 60. Codex 实施前最终风险审计清单（v2.1 新增）

在把任何一个版本交给 Codex 前，先逐项确认：

### A. 数据是否闭合

- [ ] `states[T+1] / actions[T] / dts[T]` 对齐唯一；
- [ ] context/future window 不跨 trajectory；
- [ ] future action 第 0 个元素明确对应当前状态到下一状态；
- [ ] raw/model 两套状态没有混用；
- [ ] `z_phys` 使用 raw units、坐标和 quadrature weights；
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
- [ ] future `z_phys` 由 `PhysicalReadout(z_k_pred)` 产生；
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
- [ ] 跨模型比较使用 raw-unit physical observables/field metrics；
- [ ] closure 使用 `C_closure` + `R_closure`，不是单一不稳定比例；
- [ ] base Koopman 和 full model 的 error 同时报告；
- [ ] 至少 3 seeds 的阶段报告不只保留 best run；
- [ ] Codex 只输出 scientific evidence，不自行宣布论文结论成立。

### G. 代码是否仍保持模块化

- [ ] `world_model.py` 仅编排；
- [ ] `physics/` 无 trainable NN；
- [ ] `koopman_core.py` 不依赖 JEPA/Attention；
- [ ] `residual_memory.py` 不调用 Koopman 内部实现；
- [ ] `PhysicalReadout` 是独立模块；
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

JEPA 决定“学什么表示”，Koopman 决定“主要动力学怎样组织”，Attention 决定“剩余复杂关系怎样闭合”，Physics grounding 决定“这个 latent 是否仍然属于物理世界”。

