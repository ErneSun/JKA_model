# V0.3 Code Walkthrough — Direct-State Continuous-Time KoopmanCore

本文对应仓库中的最终 V0.3 代码。软件版本为 `0.3.0`，架构修订仍为 `2.2`。本版本只回答：
给定低维 direct state `z`，continuous-time linear generator 能否被正确传播、学习和分析？

## 1. V0.3 到底增加了什么

```text
V0.1: engineering contracts / config / checkpoint / reproducibility
V0.2: real trajectories / windows / normalization / executable physics contracts
V0.3: first dynamical model — ContinuousKoopmanCore
```

| 能力 | V0.2 | V0.3 |
|---|---|---|
| dynamical model | 无 | direct-state `dz/dt=A z` |
| time propagation | 数据 contract | exact `matrix_exp(A*dt)` |
| learnable object | 无 | 仅矩阵 `A[d,d]` |
| rollout | 无 learned rollout | closed-loop constant/variable-dt rollout |
| spectrum | 无 | detached continuous eigenspectrum |
| strict reference | advection-diffusion PDE | damped oscillator closed form |
| nonlinear diagnostic | 无 | unforced Duffing limitation |

V0.3 仍然没有 learned latent encoder。这里写作 `z` 的 state 就是 `[x,v]`，不是未来
`z_k=E_K(U)`。

## 2. 核心数学：`A` 与 `K(dt)`

autonomous continuous system：

\[
\frac{dz}{dt}=Az.
\]

当 `A` 固定时，线性 ODE 的精确解为

\[
z(t+\Delta t)=e^{A\Delta t}z(t).
\]

定义

\[
K_{\Delta t}=e^{A\Delta t}.
\]

- `A`：continuous-time generator，单位为 `1/time`；
- `K(dt)`：某一个具体时间跨度的 dimensionless discrete propagator；
- autonomous system 中 `A` 不随 step 改变；
- `dt` 改变时，`K(dt)` 必然改变。

因此必须牢记：

\[
\boxed{A\ne K_{\Delta t}}.
\]

### Column-vector convention

数学唯一方向为

\[
z_{next}=Kz.
\]

Python 单状态 `z[d]` 直接计算 `K @ z`。batch `z[B,d]` 的每一行只是数学 column state 的
存储形式，所以 shared matrix 使用等价运算 `z @ K.T`；per-sample matrices 使用
`einsum("bij,bj->bi",K,z)`。代码没有混用 `z@K`。

## 3. 为什么不用 Euler

Euler 使用

\[
z_{next}\approx(I+\Delta tA)z,
\]

它只是 exponential 的一阶近似，step 较大或 long rollout 时会积累误差，也会模糊
continuous spectrum 与 variable-`dt` 语义。

`ContinuousKoopmanCore.transition_matrix()` 真正调用 `torch.matrix_exp(A*dt)`。通用 numerical
reference test 使用独立 rotation-decay closed form

\[
e^{-\alpha dt}
\begin{bmatrix}
\cos(\omega dt)&-\sin(\omega dt)\\
\sin(\omega dt)&\cos(\omega dt)
\end{bmatrix}
\]

验证它，不用同一个 core 自证。实际 `[x,v]` oscillator smoke 另用其物理生成元
`[[0,1],[-omega0^2,-2*gamma]]` 和独立 underdamped analytical transition 交叉验证。
`matrix_exp` 位于 CPU-safe autocast-disabled precision island；正式 core 可用 float32，数学
reference/identification smoke 使用 float64。

## 4. 真实代码结构图

```text
scripts/smoke_v0_3.py
  ├── data/toy_oscillators.py
  │    ├── analytical damped oscillator
  │    └── reference-only Duffing RK4
  ├── models/koopman_core.py
  │    ├── generator_matrix
  │    ├── transition_matrix
  │    ├── step
  │    ├── rollout
  │    └── spectrum
  ├── training/direct_koopman.py
  ├── metrics/spectral.py
  ├── evaluation/dynamics.py
  ├── rollout/koopman_rollout.py
  ├── data/splits.py
  ├── utils/logging.py
  └── utils/checkpoint.py
```

其它入口：

- `generate_oscillator.py`：保存解析 oscillator trajectories；
- `explain_v0_3.py`：13 STEP 教学流程；
- `analyze_spectrum.py`：读取 config true generator 或 learned checkpoint；
- `configs/v0_3_smoke.yaml`：所有实验参数，不把训练设置硬编码在脚本中。

`koopman_rollout()` 只是调用 `core.rollout()`，没有第二份 propagation 实现。

## 5. 数据流与训练流

单步 propagation：

```text
z_t + dt_t
    │
    ▼
A * dt_t
    │
    ▼
torch.matrix_exp
    │
    ▼
K(dt_t)
    │
    ▼
K(dt_t) @ z_t
    │
    ▼
z_(t+1)
```

训练：

```text
z_t ──> ContinuousKoopmanCore ──> predicted z_(t+1)
                                      │
true z_(t+1) ─────────────────────────┤
                                      ▼
                                     MSE
                                      │
                                      ▼
                            gradient through matrix_exp
                                      │
                                      ▼
                                      A
```

完整 trajectory 先由 V0.2 `make_split_manifest()` 做 deterministic、trajectory-level split；只用
train trajectories 构造 one-step pairs，100-step 指标在 held-out test trajectory 上计算。唯一
loss 为

\[
\mathcal L_{step}=\frac1N\sum_i\|e^{A\Delta t_i}z_i-z_{i+1}\|_2^2.
\]

没有 reconstruction、JEPA、physics、residual 或 spectrum training loss。Spectrum 只读
`A.detach()`。

## 6. Tensor Shape Table 与 API

| Object | Shape | Meaning |
|---|---:|---|
| `A` | `[d,d]` | continuous generator |
| single `z` | `[d]` | one mathematical column state |
| batch `z` | `[B,d]` | B column states stored as rows |
| scalar `dt` | `[]`/Python scalar | shared interval |
| batch `dt` | `[B]` | per-sample interval |
| scalar rollout schedule | scalar + `horizon` | repeated interval |
| shared schedule | `[H]` | all batch samples use same schedule |
| per-sample schedule | `[B,H]` | each sample has its own intervals |
| single rollout | `[H+1,d]` | includes `z0` |
| batch rollout | `[B,H+1,d]` | includes every initial state |

```python
core.generator_matrix()
core.transition_matrix(dt)
core.step(z, dt)
core.rollout(z0, dts, horizon=None)
core.spectrum()
```

`dt=0` 合法并返回 identity；negative/non-finite/wrong-shape `dt` 失败。Batch `dt` 一次形成
`[B,d,d]` 并调用 batched `torch.matrix_exp`，不是逐 sample Python loop。Rollout 因为 closed-loop
时间依赖而按 step 递推，每个 next prediction 成为下一步输入，绝不偷用 ground truth。

## 7. Spectrum Table

连续 eigenvalue：

\[
\lambda=\sigma+i\omega.
\]

| Quantity | Code field | Meaning |
|---|---|---|
| eigenvalue | `eigenvalues` | continuous mode |
| `sigma` | `growth_rates` | 正值增长，负值衰减 |
| `abs(omega)` | `angular_frequencies` | rad/time |
| `abs(omega)/(2*pi)` | `frequencies_hz` | cycles/time |

本次 learned oscillator：

```text
lambda = -0.1500000001 ± 1.9943670680 i
growth/decay rate = -0.1500000001
frequency = 0.3174133772 Hz
```

`SpectrumDiagnostics` 的所有 tensor 均 detached；训练不经 `eigvals` 反传。测试比较 real part
与 `abs(imag)`，不依赖 PyTorch 的 arbitrary eigenvalue ordering。

## 8. Damped Oscillator：Fixed-A 与 learned-A 结果

系统：

\[
\dot x=v,\qquad
\dot v=-\omega_0^2x-2\gamma v,
\]

\[
A_{true}=\begin{bmatrix}0&1\\-\omega_0^2&-2\gamma\end{bmatrix},
\quad
\lambda_\pm=-\gamma\pm i\sqrt{\omega_0^2-\gamma^2}.
\]

config 使用 `omega0=2.0`、`gamma=0.15`、deterministic variable `dt`。trajectory generator 用
独立 underdamped analytical formula，不调用 `KoopmanCore`。

| Quantity | True | Learned | Relative error |
|---|---:|---:|---:|
| angular frequency | 1.9943670680 | 1.9943670680 | `2.29e-10` |
| ordinary frequency | 0.3174133773 | 0.3174133772 | `2.29e-10` |
| damping | 0.1500000000 | 0.1500000001 | `8.02e-10` |

```text
initial one-step MSE: 3.281851e-3
final one-step MSE:   5.290348e-22
100-step Koopman MSE: 6.373837e-19
100-step persistence: 1.202215e+0
```

learned growth rate 为负，100-step closed-loop rollout finite，并显著优于 persistence。

## 9. Fixed-A vs Trainable-A

### Fixed-A numerical verification

已知物理状态生成元
`A_true=[[0,1],[-omega0^2,-2*gamma]]`，`trainable=False`，并用独立解析 transition 验证
implementation：

```text
closed-form matrix-exp max error: 1.11e-16
semigroup max error:              3.33e-16
zero-dt max error:                0
```

这不是机器学习。

### Trainable-A system identification

不知道 `A_true`，从 deterministic random small matrix 初始化；不能用 near-truth initialization。
Adam 只更新 `A`，每个 pair 使用自己的 variable `dt_i`。频率 `<1%` gate 验证学到的是连续
动力学，而不只是某个 fixed-step transition。

## 10. Variable `dt` 的真实语义

例如教学脚本使用：

```text
dt0=0.02, dt1=0.05, dt2=0.03
K0=exp(A*0.02)
K1=exp(A*0.05)
K2=exp(A*0.03)
z1=K0 z0
z2=K1 z1
z3=K2 z2
```

`K0/K1/K2` 不同，但 generator `A` 相同。Damped trajectories 也随机但 deterministic 地 jitter
每一步 `dt`，training 将完整 `[N]` dt tensor 送入 batched matrix exponential，不 resample。

Autonomous semigroup property

\[
e^{A(dt_1+dt_2)}=e^{A dt_2}e^{A dt_1}
\]

既验证时间语义，也验证 successive rollout 与单次累计时间 propagation 一致到数值容差。

## 11. Duffing 为什么不可能被固定 `2×2 A` 完全闭合

unforced Duffing：

\[
\dot x=v,\qquad
\dot v=-\delta v-\alpha x-\beta x^3.
\]

`x^3` 使 vector field 对 direct `[x,v]` 非线性。一般不存在一个固定 `2×2 A` 能在整个状态
空间满足 `dz/dt=A z`。V0.3 使用 small deterministic RK4 只生成 reference data；RK4 不进入
KoopmanCore。

真实 diagnostic：

```text
Duffing one-step MSE: 7.698247e-7
Duffing rollout MSE:  8.309227e-5
finite: YES
```

这些误差大于线性 oscillator 是预期 closure limitation，不是程序 bug。V0.3 不增加 polynomial
lifting、encoder、额外 state dimension 或 residual 来美化结果。

## 12. Tests 查阅表

| Test | 数学意义 | 防止的 bug |
|---|---|---|
| `test_matrix_exp_against_closed_form` | 独立解析 exponential | 用 Euler 或 circular verification |
| `test_zero_dt_is_identity` | `exp(0)=I` | zero interval 改变 state |
| `test_semigroup_property` | autonomous flow composition | transition 方向/dt 错误 |
| single/batch step tests | `[d]`/`[B,d]` contract | transpose 或 broadcasting 错误 |
| per-sample dt / batch-vs-loop | batched `matrix_exp` | 所有样本误用同一 dt |
| invalid dt/state rejection | forward-time/finite/shape policy | silent correction、NaN/Inf |
| `test_matrix_exp_gradient_wrt_A` | differentiable identification | matrix exponential 断图 |
| autocast precision island | core precision | bfloat16 污染 matrix exponential |
| rollout tests | closed-loop/include initial | teacher forcing 或 off-by-one |
| spectrum tests | continuous rates/frequency | 分析 discrete K 或依赖 eig 顺序 |
| analytical damped reference | independent true trajectory | model自生成、模型自验证 |
| identification frequency | `<1%` scientific gate | 只看 training loss |
| 100-step/persistence | long-horizon dynamics | one-step 好但 rollout 崩溃 |
| checkpoint round-trip | persistence identity | reload 后 A/config 漂移 |
| Duffing pipeline | finite limitation experiment | 为非线性系统偷偷扩模型 |
| V0.1/V0.2 full regression | earlier contracts stable | 新模型破坏数据/physics 基线 |

## 13. Recommended Reading Order

1. `scripts/explain_v0_3.py`：先看具体矩阵和 13 STEP；
2. `models/koopman_core.py`：核心 `matrix_exp`、shape、rollout；
3. `data/toy_oscillators.py`：独立解析/RK4 references；
4. `metrics/spectral.py`：continuous detached spectrum；
5. `training/direct_koopman.py`：唯一 loss 与优化循环；
6. `evaluation/dynamics.py`：rollout/persistence metrics；
7. `scripts/smoke_v0_3.py`：全部组件与 checkpoint；
8. `tests/test_koopman_*.py`、`test_direct_koopman_identification.py`：数学 gates。

## 14. Top Symbols

| Symbol | File | Input | Output/Purpose |
|---|---|---|---|
| `ContinuousKoopmanCore` | `models/koopman_core.py` | state dim、A、trainable | continuous generator module |
| `transition_matrix` | same | scalar/`[B] dt` | exact `exp(A*dt)` |
| `step` | same | `[d]`/`[B,d] z`、dt | one propagated state/batch |
| `rollout` | same | z0、scalar/`[H]`/`[B,H] dts` | closed-loop states including z0 |
| `spectrum` | same | live A | detached `SpectrumDiagnostics` |
| `generate_damped_oscillator_trajectories` | `data/toy_oscillators.py` | config、seed | analytical records/spec |
| `generate_duffing_trajectories` | same | config、seed | RK4 records/spec |
| `trajectory_transition_tensors` | same | records | aligned z/target/dt pairs |
| `initialize_direct_koopman` | `training/direct_koopman.py` | seed/init scale | random trainable core |
| `one_step_mse` | same | core、pairs | sole V0.3 loss |
| `train_direct_koopman` | same | core/data/config | result + optimizer state |
| `continuous_spectrum` | `metrics/spectral.py` | A | detached eigen diagnostics |
| `relative_frequency_error` | same | estimated/reference frequency | scientific acceptance metric |
| `spectral_growth_rate` | same | spectrum | maximum continuous growth diagnostic |
| `evaluate_rollout` | `evaluation/dynamics.py` | prediction/truth | Koopman/persistence metrics |
| `koopman_rollout` | `rollout/koopman_rollout.py` | core/z0/dts | delegates to core source of truth |

## 15. V0.3 还没有什么

```text
No learned encoder
No z_k lifting network
No TrainingDecoder
No JEPA / EMA target encoder
No z_r / residual closure
No Attention / Transformer / GRU
No action-conditioned B
No parameter-conditioned A(mu)
No PDE encoder or neural operator
No MPC / RL
```

虽然 direct state 在数学上写作 `z`，它不是未来 learned `z_k=E_K(U)`。

## 16. V0.4 接口

V0.4 将实现（仅说明，不在当前代码中）：

```text
physical/model state U
        │
        ▼
learned KoopmanEncoder E_K
        │
        ▼
learned z_k
        │
        ▼
V0.3 ContinuousKoopmanCore.step/rollout/spectrum
```

也就是说，V0.4 将 learned `z_k` 输入已经验证过的 V0.3 core；当前 core 不依赖 oscillator
specific code，可以原样复用。本版本不创建 `KoopmanEncoder` 或任何 lifting network。
