# V0.5 Koopman 调整与最终验证研究记录

> 审阅日期：2026-08-16
> 正式训练代码 commit：`976de084c540e28a21411614ed0c854291ed491c`
> 结果归档 commit：`b4719a3`
> GPU：NVIDIA GeForce RTX 5080，FP32
> 最终证据：[`v05-final-20260814`](../gpu_validation/v0_5/results/v05-final-20260814/final_validation.md)

## 1. 结论

**当前 V0.5 Koopman 验证通过。**

这里的“通过”严格限定为当前项目合同中的二维周期、常系数、单 Fourier 模式
advection-diffusion 问题。自动报告保持
`scientific_status=PENDING_REVIEW` 和
`overall_acceptance=PENDING_RESEARCHER_REVIEW`，是因为验证脚本有意不自动授予最终科学
PASS。经过本次人工审阅：

- seeds `47/53/59` 的频率、衰减、稳定性和重建门槛全部通过；
- 三个 horizon 的 closed-loop forecast、相对质量漂移和谱一步算子门槛全部通过；
- physics/no-physics 配对实验通过预测及约束非劣性门槛；
- 六个最终 run 均为当前代码重新训练，旧 checkpoint 因 config hash 不兼容而没有复用。

因此，本次人工审阅决定为：

```text
V0.5 Koopman contract: PASS
GPU workflow: PASS
Scientific gates: PASS after researcher review
Claim boundary: reduced analytical single-mode PDE validation
```

这不等于模型已经对一般多模态 PDE、外部 CFD 或实验数据获得验证。

## 2. 当前 Koopman 要求逐项核对

最终报告统计三个 physics seeds；表中的“最差值”是最不利种子的结果。

| 项目 | 验收条件 | 三种子结果 | 判定 |
|---|---:|---:|---:|
| 角频率相对误差 | `<= 5%` | mean `0.2455%`；worst `0.2528%` | PASS |
| 衰减率相对误差 | `<= 20%` | mean `15.8926%`；worst `16.5942%` | PASS |
| 谱横坐标 | `<= 1e-3` | mean `-7.858e-5`；worst `-2.335e-5` | PASS |
| 重建 | 小于 short persistence RMSE | worst reconstruction `0.003801`；最小 short persistence `0.030444` | PASS |
| short forecast | RMSE 小于 persistence | `0.002841` vs `0.031909`（mean） | PASS |
| medium forecast | RMSE 小于 persistence | `0.002170` vs `0.306101`（mean） | PASS |
| long forecast | RMSE 小于 persistence | `0.013767` vs `0.958274`（mean） | PASS |
| 相对质量漂移 | 每个 horizon `<= 0.01` | worst `0.000776` | PASS |
| 谱一步 operator MSE | 每个 horizon `<= 1e-4` | worst `1.180e-5` | PASS |
| physics 预测非劣性 | added RMSE / persistence `<= 5%` | median worst `1.413%` | PASS |
| physics 约束非劣性 | added constraint / hard limit `<= 10%` | median worst `4.028%` | PASS |

三个 horizon 的 physics 平均结果为：

| Horizon | RMSE | Relative L2 | Persistence RMSE | Relative mass drift | Spectral-step MSE |
|---|---:|---:|---:|---:|---:|
| short (`1`) | `0.00284069` | `0.00406544` | `0.0319093` | `0.000541403` | `8.69440e-6` |
| medium (`16`) | `0.00217049` | `0.00316417` | `0.306101` | `0.000590532` | `7.23457e-7` |
| long (`80`) | `0.0137667` | `0.0226102` | `0.958274` | `0.000736435` | `2.53258e-7` |

相对 persistence，平均 RMSE 分别下降 `91.10%`、`99.29%` 和 `98.56%`。

## 3. 初始版本的问题

这里的“初始 Koopman”指第一次正式 GPU 验证
[`v05-final-20260812`](../gpu_validation/v0_5/results/v05-final-20260812/final_validation.md)
所对应的 V0.5 field Koopman，而不是更早版本中已经固定的连续时间公式。

连续时间动力学始终是：

\[
\dot z=Az,\qquad z(t+\Delta t)=\exp(A\Delta t)z(t).
\]

第一次正式验证中：

- frequency relative error 为 `96.31%`，远高于 `5%` 门槛；
- short/medium/long RMSE 分别为 `0.7514/0.7218/0.6295`；
- 虽然 long RMSE 优于 persistence，但 latent generator 没有识别到正确旋转频率；
- 场预测、质量和 operator 数值较大，无法支持 Koopman 动力学已经学对的结论。

主要原因不是 `matrix_exp` 实现错误，而是编码器、损失和优化没有向生成元提供足够且正确的
相位/频率监督。

## 4. 调整了哪些模块

### 4.1 Encoder：保留传播相位

初始 `KoopmanEncoder2D` 使用 `AdaptiveAvgPool2d(1)`。全局空间平均会使编码近似平移不变，
而 travelling Fourier wave 的时间演化本质上表现为相位平移；因此编码器会丢掉 Koopman
旋转模态最关键的信息。

调整后：

- 去除 global average pooling；
- 保留两次 stride-downsample 后的粗空间 feature map；
- flatten 后投影到 8 维 latent；
- 显式检查输入网格大小。

结果是不同 wave phase 能映射到不同 latent 坐标，生成元 `A` 才可能学习对应的共轭复
特征值。

代码：[`field_koopman_autoencoder.py`](../src/jka_model/models/field_koopman_autoencoder.py)

### 4.2 Koopman loss：从 latent 对齐扩展到动力学和场预测共同约束

初始损失主要依赖 one-step/multi-step latent consistency、reconstruction 和 variance。
调整后新增：

1. **Generator consistency**

   \[
   L_{generator}=\left\|
   \frac{z_{n+1}-z_n}{\Delta t}-z_nA^T
   \right\|_2^2.
   \]

   它直接监督连续时间生成元，而不是仅通过多个 matrix exponential 间接学习 `A`。

2. **Decoded forecast loss**

   \[
   L_{forecast}=\|D(\hat z_{n+1:n+H})-u_{n+1:n+H}\|_2^2.
   \]

   防止 latent 距离很小、但 decoder 后物理场预测仍然不准确。

3. **Stability regularizer**

   对 `A+A^T` 的不稳定对称部分施加惩罚，配合谱横坐标硬门槛限制增长模态。

4. **质量累计方式**

   所有未来预测都相对 rollout 初始质量比较，避免只约束相邻预测而允许误差逐步累积。

代码：[`field_koopman.py`](../src/jka_model/losses/field_koopman.py)

### 4.3 Generator 初始化与优化

初始生成元是小随机矩阵。调整后初始化为：

\[
A_0=s(R-R^T-I),
\]

其中反对称部分允许旋转模态，`-I` 提供稳定衰减起点。训练时 encoder/decoder 使用基础
learning rate，generator 使用 `5x` learning rate，以解决 `A` 学习明显慢于 CNN 的问题。

这不是把真实频率写入模型；初始化没有使用 reference `omega` 或 `gamma`。

代码：[`train_v0_5.py`](../src/train/train_v0_5.py)

### 4.4 PhysicsConstraint：从不匹配的离散残差改为一致的谱一步约束

第二轮验证已经解决 Koopman 频率问题，但 physics 消融失败。旧 operator 使用二阶有限差分
和梯形时间残差，而训练数据由周期单 Fourier 模式的解析解生成。真实解析轨迹代入旧约束
本身就有非零离散残差，因此物理梯度会对正确轨迹施加偏置。

当前 operator 使用与数据合同一致的精确谱一步传播：

\[
\widehat u^{n+1}_{k_x,k_y}=
\exp\left(
[-\nu(k_x^2+k_y^2)-i(c_xk_x+c_yk_y)]\Delta t
\right)\widehat u^n_{k_x,k_y}.
\]

同时质量损失改为无量纲相对形式：

\[
L_{mass}=\operatorname{mean}\left[
\frac{M(\hat u)-M(u_{ref})}
{\sum |u_{ref}|w+\epsilon}
\right]^2.
\]

这样训练和评估的物理尺度一致，不再随区域面积和场幅值任意变化。由于 operator 的量纲和
数值尺度改变，物理权重也重新标定为
`lambda_physics=0.2, lambda_mass=1, lambda_operator=1`。

代码：[`constraints.py`](../src/jka_model/physics/constraints.py)、
[`advection_diffusion_2d.py`](../src/jka_model/problems/advection_diffusion_2d.py)

### 4.5 Checkpoint 与验证策略

训练与评估还进行了以下工程/研究调整：

- 分离 `best_forecast` 和 `best_physics` checkpoint；
- 增加 post-warmup checkpoint，防止物理 warmup 前的早期模型绕过物理目标；
- checkpoint selection 使用与训练一致的加权 physics objective；
- 增加 decay error 和 spectral abscissa 门槛；
- 将单 seed 扩展为 seeds `47/53/59` 的配对 physics/no-physics 实验；
- 增加 short/medium/long 的 forecast、mass 和 operator 硬门槛；
- checkpoint 复用前检查 schema、architecture revision 和 config hash；不兼容时自动补训。

最终六个 run 均为重新训练，排除了旧 checkpoint 配置迁移对结果的影响。

## 5. 三轮研究结果

| 阶段 | 主要状态 | Frequency error | Physics mean RMSE（short/medium/long） | 结论 |
|---|---|---:|---:|---|
| 2026-08-12 初始正式验证 | scientific FAIL | `96.31%` | `0.7514 / 0.7218 / 0.6295` | 未识别正确 Koopman 频率 |
| 2026-08-13 动力学修正 | direct gates PASS；ablation FAIL | `0.2392%` | `0.002994 / 0.002184 / 0.013580` | Koopman 已学对；旧 physics 定义和严格比值门槛仍有问题 |
| 2026-08-14 最终修正 | all gates PASS；等待人工审阅 | `0.2455%` | `0.002841 / 0.002170 / 0.013767` | Koopman、滚动预测、绝对物理门槛和非劣性全部成立 |

从初始版到最终版：

- frequency relative error 降低约 `99.745%`；
- short RMSE 降低约 `99.622%`；
- medium RMSE 降低约 `99.699%`；
- long RMSE 降低约 `97.813%`。

这些改善是 encoder、loss、初始化和优化联合修改后的系统效果。当前证据不能把改善比例严格
分摊给某一个模块，因为没有为每项修改单独进行 factorial ablation。

## 6. 如何解释 physics 与 no-physics

最终 paired median raw relative change 为：

| Horizon | Physics RMSE 相对变化 | Physics mass drift 相对变化 | Physics operator 相对变化 |
|---|---:|---:|---:|
| short | `+17.776%` | `+79.577%` | `+38.592%` |
| medium | `-3.519%` | `+90.041%` | `+27.771%` |
| long | `-0.782%` | `+116.731%` | `+16.036%` |

这些百分比不能单独作为失败依据，因为 no-physics 的分母已经接近误差底限。使用预先定义的
有意义尺度后：

- short 增加的预测误差只占 persistence RMSE 的 `1.413%`，低于 `5%` 非劣门槛；
- medium/long 的 physics RMSE 中位数分别改善 `3.519%/0.782%`；
- mass 的最大中位退化只占绝对质量门槛的 `4.028%`；
- operator 的最大中位退化只占绝对 operator 门槛的 `2.426%`；
- physics 自身的所有 mass/operator 绝对硬门槛均通过。

因此，当前证据支持的准确表述是：

> 在当前解析单模态数据充足的设置中，加入物理损失没有带来统一的 raw-metric 优势，但在
> 三种子实验中没有造成具有实际意义的预测或物理一致性退化，并且满足所有绝对物理门槛。

不能表述为“physics 模型在所有指标上优于 no-physics”。要证明物理约束的正向收益，下一步
应研究低数据、分布外参数、多模态初值或更长外推，而不是继续比较已经接近零的训练内误差。

## 7. 哪些核心数学合同没有改变

调整过程中没有改变以下 Koopman 核心合同：

- latent 仍满足 `dz/dt=Az`；
- variable-dt 仍使用精确 `exp(A*dt)`；
- rollout 仍为 closed loop，不使用 future truth teacher forcing；
- decoder 只负责 latent-to-field 映射，不承担时间推进；
- physics constraint 仍在 inverse normalization 后的 raw physical units 中计算；
- 没有向模型注入真实频率、真实衰减率或测试集信息。

这说明修正针对的是可辨识性、优化和离散一致性，而不是更换 Koopman 数学模型。

## 8. 证据索引

- 最终人工审阅依据：
  [`final_validation.md`](../gpu_validation/v0_5/results/v05-final-20260814/final_validation.md)
- 最终机器可读记录：
  [`final_validation.json`](../gpu_validation/v0_5/results/v05-final-20260814/final_validation.json)
- 初始失败结果：
  [`v05-final-20260812`](../gpu_validation/v0_5/results/v05-final-20260812/final_validation.md)
- 动力学修正后的中间结果：
  [`v05-final-20260813`](../gpu_validation/v0_5/results/v05-final-20260813/final_validation.md)
- 当前完整 GPU 配置：
  [`gpu_full.yaml`](../gpu_validation/v0_5/configs/gpu_full.yaml)
- 当前验证合同：
  [`evaluation.md`](../docs/v0_5/evaluation.md)

## 9. 后续研究建议

当前 V0.5 可以结束“实现是否正确、Koopman 是否识别基本频率、闭环预测是否稳定”的验证。
若继续研究 physics loss 的实际价值，优先顺序建议为：

1. 减少训练轨迹，比较 low-data sample efficiency；
2. 在训练范围外改变 `cx/cy/nu`，测试 parameter OOD；
3. 使用多 Fourier modes，检查 8 维 latent 的多频率辨识能力；
4. 延长超过 80 steps 的 rollout，并独立检查误差随时间增长；
5. 最后再考虑外部数值解、CFD 或实验数据校准。

这些属于后续版本研究，不应回写为当前 V0.5 已经证明的结论。
