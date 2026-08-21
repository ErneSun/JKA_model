# V0.9 evaluation

顺序固定：V0.8 strict handoff → physical schedule gate → validation-only rank sweep → 3×3 nested formal training →
locked test → teacher-free rollout → physical decoding → aggregation。

每个正式 run 检查：

- one-step adaptive gain；
- `Gamma_op`；
- dynamic-over-static gain；
- R3 real-history over shuffled-history gain；
- horizons 8/16/32/80；
- longest horizon；
- finite/stability；
- operator burden；
- velocity/vorticity/divergence/lift/drag/frequency/no-slip。

Known 与 latent-inferred 分别判定。V0.9 scientific support 使用至少 2/3 backbone joint pass；V1.0 readiness
要求 3/3 backbone 对 known 与 latent 两条链共同通过。Software PASS 与 scientific result 分离。
