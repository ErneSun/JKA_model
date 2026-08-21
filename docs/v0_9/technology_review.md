# V0.9 technology review

## ADOPT

- **Parameter-varying Koopman idea**：参数变化时不应继续假设一个固定 operator；本项目采用受限低秩连续 generator
  modulation，而不直接插值多个完整 operators。参考 Lee, Park & Kim, 2023,
  <https://arxiv.org/abs/2309.10278>。
- **Koopman with inputs/control distinction**：known operating condition 属于已知外部输入，必须和 latent-inferred
  experiment 分开。参考 Proctor, Brunton & Kutz, 2016, <https://arxiv.org/abs/1602.07647>。
- **Exact batched matrix exponential**：继续使用 PyTorch `torch.linalg.matrix_exp`，支持 batched square matrices，
  <https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_exp.html>。
- **Long-rollout stability evaluation**：稳定约束可能改善长期预测，但本项目不把严格收缩强加给振荡 wake；采用
  burden/proxy diagnostics 加真实 rollout/physics gate。参考 Mamakoukas, Abraham & Murphey, 2020,
  <https://arxiv.org/abs/2005.04291>。

## OPTIONAL

- validation-selected rollout training；
- explicit trust/fallback gate；
- Mixture-of-Koopman for confirmed discrete regimes；
- parametric DMD/operator interpolation baseline，参考 <https://arxiv.org/abs/2204.12006>。

## DEFER

- strict contractive generator by construction；
- joint fine-tuning；
- remaining-residual closure；
- unseen-condition generalization。

## REJECT FOR PRIMARY V0.9

- unrestricted full-matrix hypernetwork；
- additive residual correction；
- large spatial Transformer；
- test-driven rank selection。
