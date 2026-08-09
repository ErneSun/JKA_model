# V0.4 Code Walkthrough：学习连续时间 Koopman 坐标

本文对应项目版本 `0.4.0`、架构修订 `2.2` 的实际实现。Mandatory validation 全部在 CPU、
`float64` 上完成。V0.4 只加入 learned lifting 与 training decoder，不进入 V0.5。

## 1. 从 V0.3 到 V0.4

V0.3 直接令 `z=U`；V0.4 改为

\[
z_K=E_K(U^{model}).
\]

时间传播仍由原来的 `ContinuousKoopmanCore` 完成，没有第二套 dynamics implementation。

## 2. 完整数学链

\[
U_t^{model}\xrightarrow{E_K}z_t^K,
\qquad
z_t^K\xrightarrow{\exp(A\Delta t_t)}\hat z_{t+1}^K,
\qquad
\hat z_{t+1}^K\xrightarrow{D_{train}}\hat U_{t+1}^{model}.
\]

`D_train` 的 target 是 normalized/model-space state；raw-unit evaluation 只通过已有
`normalizer.inverse_transform()` 获得。

## 3. 为什么要学习坐标

primary system 的真实状态是 `s=[q,p]`，观测却是

\[
U=[q,p,q^2,qp,p^2].
\]

五个 nonlinear observation channel 并不表示动力学本征维度是五。Encoder 寻找的是“让时间
演化尽量线性且可闭合”的二维坐标，而不只是一般意义的压缩表示。

## 4. 为什么 learned latent 不必等于 true state

若 `z=T s` 且 `T` 可逆，则

\[
\dot z=T A_{true}T^{-1}z.
\]

`z` 可以旋转、缩放或线性混合；`A_z` 与 `A_true` 仍为 similarity transform，因此 eigenvalue、
frequency 和 decay rate 相同。评估采用 train-fit/test-apply affine alignment，而不是直接比较
两个任意坐标系的 component MSE。

## 5. KoopmanEncoder

文件：`src/jka_model/models/koopman_encoder.py`。

primary acceptance 的实际结构为 `Linear(5,2)`，参数量 `5×2+2=12`，输入 `[...,5]`，输出
`[...,2]`。它是随机初始化并通过 V0.4 objective 学到的，不使用 true latent 或 oracle inverse
map。实现也支持 1/2 hidden-layer Tanh/SiLU MLP，但复验的小型 Tanh MLP 没有达到严格数值门槛；
因此正式 config 选择零 hidden-layer 的退化小型网络。这是相对规范“建议 2–3 Linear MLP”的
明确偏差，不是隐藏实现细节。

## 6. TrainingDecoder

文件：`src/jka_model/models/training_decoder.py`。实际结构为
`2 → 48 → 48 → 5`，两个 hidden activation 均为 Tanh，参数量
`(2×48+48)+(48×48+48)+(48×5+5)=2741`。

它只实现 `forward(z_k) -> U_model_hat`，不接收 `dt`，没有 `step()`、`rollout()` 或 generator。

## 7. Loss table

| Loss | 公式 | 实际权重 | 作用 |
|---|---|---:|---|
| `L_K` | `MSE(exp(A dt) z_t, E(U_t+1))` | 10 | one-step consistency |
| `L_K_multi` | `MSE(core.rollout(z_t), E(U_future))` | 50 | closed-loop long-horizon consistency |
| `L_rec` | `MSE(D(E(U_model)), U_model)` | 10 | 保留 observation information |
| `L_var` | `mean(ReLU(0.2-std(z))²)` | 10 | 防止 latent collapse |
| `L_spec` | symmetric-part logarithmic-norm penalty | 0.01 | 小权重 stability regularizer |

`std` 使用 `unbiased=False`。`L_spec` 是可微稳定性 proxy，不对 detached eigenspectrum 反传；类的
默认权重为 0，smoke 明确启用 0.01。

训练每 100 epoch 保存一次 diagnostic snapshot（含五项 loss、latent mean/std/min/max），实际
1000 epoch run 得到 11 个 snapshots，total loss 从 `10.18071159` 降到 `0.002310492028`。

## 8. 为什么只有 Koopman loss 会 collapse

若 `E_K(U)=0`，则 `exp(A dt)0=0`，one-step 和 multi-step consistency 都可以是零，但 representation
完全没有信息。`L_rec` 要求 decoder 能恢复输入，`L_var` 直接惩罚小标准差，两者共同排除这一
平凡解。实际 held-out minimum std 为 `0.2700218965`。

## 9. One-step 与 multi-step

```text
one-step:   z_t -> z_hat_t+1

multi-step: z_t -> z_hat_t+1 -> z_hat_t+2 -> ... -> z_hat_t+8
```

`koopman_multi_step_loss()` 一次调用 V0.3 `core.rollout()`；预测 latent 被连续传播，真实 future
latent 只构成最终 target tensor，不会成为下一步输入。去掉 multi-step loss 后 60-step decoded
rollout MSE 从 `8.3861e-4` 上升到 `2.9238`。

held-out test prediction diagnostics 为：one-step latent MSE `8.652510580e-9`，60-step
closed-loop multi-step latent MSE `3.048477090e-7`，decoded model/raw MSE 分别为
`8.386056823e-4` 与 `9.356564258e-5`。

## 10. Data flow

```text
ProblemBatch
├── context_states_model [B,2,5]
│   └── last state [B,5] ──┐
└── future_states_model [B,8,5] ─┤ concatenate [B,9,5]
                                  ▼
                           KoopmanEncoder
                                  │ z [B,9,2]
                 ┌────────────────┴───────────────┐
                 ▼                                ▼
          TrainingDecoder                 ContinuousKoopmanCore
          reconstruction                  closed-loop prediction
```

raw states、true latent、normalizer statistics 和 metrics 都不进入 optimizer。

## 11. Training call graph

```text
scripts/smoke_v0_4.py
  -> run_known_latent_experiment()
  -> TrajectoryWindowDataset / collate_problem_batches()
  -> train_koopman_representation()
  -> compute_representation_loss()
  -> encoder -> core.rollout -> decoder
  -> total.backward() -> Adam.step()
```

`TrainStage.KOOPMAN` 通过 `train_stage_modules()` 只拥有三个 group：`koopman_encoder`、
`koopman_core`、`training_decoder`。primary 总参数量为 `12+4+2741=2757`。

## 12. Shape table

| Object | Shape |
|---|---|
| `U_context_model` | `[B,2,5]` |
| `U_future_model` | `[B,8,5]` |
| current + future sequence | `[B,9,5]` |
| `z_sequence` | `[B,9,2]` |
| `z_current` | `[B,2]` |
| `z_future_target` | `[B,8,2]` |
| `A` | `[2,2]` |
| decoded sequence | `[B,9,5]` |
| rollout including initial | `[B,9,2]` |

## 13. Known-latent evaluation

数据先按 trajectory ID 切为 `20/5/5`，normalizer 只用 20 条 train trajectories 拟合，再建
window。true latent 保存在 `KnownLatentDataset.true_latents`，不属于 `TrajectoryRecord` 或
`ProblemBatch`。

V0.4 config 会在训练前交叉检查：`observation_dim=5`、trainable core、standard train-only
normalization、trajectory/window 长度、rollout horizon，以及 constant-dt 与 generator config
的一致性；不做 silent clipping 或 correction。

训练后使用 train `(z,s)` 拟合 affine map，仅在 test 上应用。结果：alignment
`R²=0.999998859958`、MSE `4.406735934e-7`；test model-space reconstruction MSE
`9.578686862e-4`。

## 14. Spectrum

| Quantity | True | Learned |
|---|---:|---:|
| eigenvalues | `-0.05 ± 1.2i` | `-0.0502484443 ± 1.2000623596i` |
| frequency | `0.190985932 Hz` | `0.190995857 Hz` |
| decay | `0.05` | `0.0502484443` |

relative frequency error 为 `5.196631094e-5`。这是 coordinate-invariant diagnostic；测试另以
`A'=TAT^-1` 验证 spectrum invariance。

## 15. Duffing secondary diagnostic

同一 held-out 轨迹设置下，V0.3 direct-state model 的 raw rollout MSE 为 `0.0138441780`，V0.4
四维 learned lifting 为 `0.0038979942`，两者均 finite。这个结果说明当前数据上 lifting 有帮助，
但 Duffing 一般不具有该小型网络能够精确表达的有限维线性 closure，不能外推为普遍结论。

## 16. Tests

| Test focus | 数学意义 | 防止的 bug |
|---|---|---|
| exact one/multi-step | `exp(A dt)` consistency | 错误传播或 teacher forcing |
| collapse/non-collapse | population variance | 零 representation 假解 |
| affine alignment | coordinate equivalence | 强迫 latent component 等于 true state |
| spectrum similarity | eigenvalue invariance | 用坐标依赖指标下结论 |
| optimizer ownership | 只训练 `E,D,A` | true latent/metric 意外入 optimizer |
| checkpoint round-trip | 完整可复现状态 | 只保存 A 或漏 normalizer |
| variable-dt rollout | 每步真实 `dt` | silent constant-dt substitution |

严格 acceptance 放在 smoke，避免将所有阈值塞入脆弱的 unit test。

## 17. Recommended reading order

1. `scripts/explain_v0_4.py`
2. `src/jka_model/data/known_latent.py`
3. `src/jka_model/models/koopman_encoder.py`
4. `src/jka_model/models/training_decoder.py`
5. `src/jka_model/models/koopman_autoencoder.py`
6. `src/jka_model/losses/koopman.py`
7. `src/jka_model/training/koopman_representation.py`
8. `src/jka_model/evaluation/representation.py`
9. `src/jka_model/evaluation/known_latent_experiment.py`
10. `scripts/smoke_v0_4.py`

## 18. Top Symbols

| Symbol | File | Input | Output | Role |
|---|---|---|---|---|
| `KoopmanEncoder` | `models/koopman_encoder.py` | `U_model [...,5]` | `z_k [...,2]` | learned coordinates |
| `TrainingDecoder` | `models/training_decoder.py` | `z_k [...,2]` | `U_model_hat [...,5]` | reconstruction only |
| `ContinuousKoopmanCore` | `models/koopman_core.py` | `z,dt` | `exp(A dt)z` | unchanged V0.3 propagation |
| `KoopmanAutoencoder.encode` | `models/koopman_autoencoder.py` | model state | latent | thin encoder call |
| `KoopmanAutoencoder.decode` | `models/koopman_autoencoder.py` | latent | model state | thin decoder call |
| `koopman_one_step_loss` | `losses/koopman.py` | current/target latent, dt | scalar MSE | one-step consistency |
| `koopman_multi_step_loss` | `losses/koopman.py` | initial/future latent, dts | scalar MSE | closed-loop consistency |
| `variance_loss` | `losses/koopman.py` | latent sequence | scalar penalty | anti-collapse |
| `train_koopman_representation` | `training/koopman_representation.py` | model/windows/config | states + diagnostics | train exactly `E+D+A` |
| `fit_affine_latent_alignment` | `metrics/representation.py` | train `(z,s)` | affine coefficients | coordinate alignment |
| `evaluate_affine_latent_alignment` | `metrics/representation.py` | fixed map + test `(z,s)` | R²/MSE | held-out evaluation |
| `generate_known_latent_trajectories` | `data/known_latent.py` | config/seed | trajectories + hidden eval state | synthetic reference |

`analyze_latent_v0_4.py` 会按修改时间检查 checkpoint，但只选择包含 V0.4 known-latent、
autoencoder 和 representation-training sections 的兼容产物，因此目录中最新文件即使是 V0.3
checkpoint 也不会被误加载。

## 19. V0.4 还没有什么

```text
No JEPA
No target encoder
No EMA
No residual z_r / closure
No Attention
No physical-field PDE training
No physics loss
No action-conditioned dynamics
```

## 20. V0.5 interface（仅说明，不实现）

V0.5 可以复用 `KoopmanEncoder` 的设计原则、`TrainingDecoder`、`ContinuousKoopmanCore` 与现有
training infrastructure，把输入扩展到 2D/regular-grid PDE state。届时才第一次对 decoded
model state 做 `inverse_transform()`，并让 `PhysicsConstraint` 作用于 decoded raw state。
