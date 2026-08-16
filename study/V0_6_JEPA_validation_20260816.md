# V0.6 JEPA 设计、优化效果与最终验证

> 审阅日期：2026-08-16  
> 正式训练 commit：`c49f0eefe09e9c8e7ecbda4ba97f31f1abe8fa2a`  
> 验证 ID：`v06-final-20260816T030842Z`  
> 最终证据：[`final_review.md`](../gpu_validation/v0_6/results/v06-final-20260816T030842Z/final_review.md)

## 1. 最终结论

V0.6 在当前预注册实验范围内验证通过：

```text
V0.6 implementation: PASS
V0.6 GPU validation: PASS
V0.6 scientific acceptance: PASS_AFTER_REVIEW
Claim boundary: reduced analytical single-mode PDE validation
```

三组 seed `47/53/59` 的 JEPA 长时 rollout RMSE 均低于 matched no-JEPA control，说明
JEPA 确实改善了最终预测，而不是仅降低一个与任务脱节的辅助 loss。所有 Koopman、physics、
non-collapse、EMA 和 online-only inference 门槛均通过。

这一结论不外推到多模态 PDE、参数分布外、CFD 或实验数据，也不支持“JEPA 改善所有物理
指标”的表述。

## 2. JEPA 在数学上优化什么

V0.6 保留完整 V0.5 目标，并增加一阶和多步 JEPA：

\[
L_{V0.6}=L_{V0.5}+\lambda_{J1}L_{J1}+\lambda_{Jm}L_{Jm}.
\]

在线编码器产生当前 latent：

\[
z_t=E_\theta(u_t),
\]

唯一时间预测器仍是连续 Koopman 传播：

\[
\hat z_{t+k}=\exp(A\Delta t_k)\hat z_{t+k-1}.
\]

未来目标由停止梯度的 EMA 编码器给出：

\[
\bar z_{t+k}=E_{\bar\theta}(u_{t+k}),\qquad
\bar\theta\leftarrow\tau\bar\theta+(1-\tau)\theta.
\]

因此 JEPA loss 直接把梯度施加到在线 encoder 和 Koopman generator `A`，要求它们产生的
未来 latent 接近变化较慢的 EMA target。target encoder 不接收梯度、不进入 optimizer，
decoder 也不由 JEPA 项直接优化；decoder 仍由完整 V0.5 reconstruction/forecast/physics 目标
训练。

直观上，V0.5 的 online future target 会与 online encoder 同时移动；JEPA 增加一个缓慢移动
的未来参照，减少表示空间与动力学同时漂移的自由度。它优化的是 latent 时间可预测性和长时
传播稳定性，而不是增加新的推理模块或替代 `exp(A\Delta t)`。

## 3. 实际预测效果

| 指标 | Control 三种子均值 | JEPA 三种子均值 | JEPA 相对变化 |
|---|---:|---:|---:|
| Short RMSE | `0.00253439` | `0.00151723` | `-40.13%` |
| Medium RMSE | `0.00237391` | `0.00170430` | `-28.21%` |
| Long RMSE | `0.01509914` | `0.01057590` | `-29.96%` |
| Long operator MSE | `2.49540e-7` | `1.26902e-7` | `-49.15%` |
| Frequency relative error | `0.2677%` | `0.1800%` | `-32.75%` |
| Decay relative error | `17.0578%` | `11.7913%` | `-30.87%` |

长时 RMSE 的逐 seed 改善为：

| Seed | Control | JEPA | 改善 |
|---:|---:|---:|---:|
| 47 | `0.01524667` | `0.01501253` | `1.54%` |
| 53 | `0.01342211` | `0.01068108` | `20.42%` |
| 59 | `0.01662864` | `0.00603409` | `63.71%` |

三个 seed 的改善幅度差异较大，但方向完全一致。因此当前证据支持“JEPA 对该任务具有预测
优化作用”，同时说明效果大小仍有 seed 敏感性。

## 4. Koopman 与 collapse 合同

三个 JEPA run 均满足：

- frequency relative error `< 0.05`；
- decay relative error `< 0.20`；
- spectral abscissa `< 0.001` 且实际全部为负；
- short/medium/long RMSE 全部优于 persistence；
- online/target minimum latent std 为 `0.213–0.497`，高于 `0.02` collapse 门槛；
- optimizer update 和 EMA update 均为 `8925`；
- EMA target 不在 rollout 推理路径中；
- control/JEPA 除 JEPA 权重和描述标签外完全 matched。

所以改善不是通过 collapse、额外推理容量、target leakage 或改变 Koopman 方程获得的。

## 5. Physics 指标的准确解释

JEPA 的 long operator MSE 在所有 seed 上都改善，但质量漂移不是统一改善：

| 指标 | Control 均值 | JEPA 均值 | 相对变化 |
|---|---:|---:|---:|
| Long operator MSE | `2.49540e-7` | `1.26902e-7` | `-49.15%` |
| Long mass drift | `0.00122698` | `0.00224025` | `+82.58%` |

Seed 47 的 mass drift 改善，seed 53/59 变差；最差 JEPA long mass drift 为 `0.00363909`，
仍低于预注册绝对门槛 `0.01`。因此正确结论是：

> JEPA 显著改善预测和 operator consistency，并保持质量漂移处于当前物理合同允许范围；
> 当前证据不能声称 JEPA 优化了质量守恒。

## 6. 计算代价

- 三种子平均训练时间由 `32.26` 分钟增加到 `38.66` 分钟，约 `+19.81%`；
- 平均峰值 GPU 显存由 `2756.0 MiB` 增加到 `2833.7 MiB`，约 `+2.82%`；
- 推理路径和推理参数量不增加，因为 EMA target 仅在训练期使用。

这表明 JEPA 的主要代价是训练吞吐，而不是部署复杂度。

## 7. 最终判断

当前 JEPA **确实实现了优化**，证据是三个独立 seed 的实际 online-only rollout 均改善，且
平均 short/medium/long RMSE、operator、频率和衰减误差均下降。该优化可以归纳为：

1. 更准确的短、中、长时场预测；
2. 更好的 latent 连续时间动力学识别；
3. 更低的谱一步 operator 误差；
4. 无 collapse，且不增加推理模块。

它没有证明的部分是统一改善 mass conservation、一般多模态泛化和参数 OOD。后续若研究
JEPA 的泛化价值，应优先进行多 Fourier modes、低数据、parameter OOD 和更长 rollout，
而不是继续在当前单模态训练内问题上微调 loss 权重。
