# V0.9 code walkthrough

1. `config/schema.py` 定义 condition、adaptive、training 和 evaluation contracts。
2. `data/cylinder_wake_2d.py` 生成 fixed-viscosity smooth/abrupt trajectories。
3. `adaptive/handoff.py` 执行 3/3 V0.8 readiness audit。
4. `adaptive/cache.py` 保存 latent、nominal residual 与 causal condition series。
5. `adaptive/models.py` 实现冻结 context 和低秩 adaptive generator。
6. `train/train_v0_9.py` 只优化 operator adapter，并保存 exact-resume checkpoint。
7. `adaptive/rollout.py` 在 initial context 后完全使用 predicted latent history。
8. `eval/evaluate_v0_9.py` 计算 residual decomposition、controls、rollout 和 physical metrics。
9. `adaptive/reporting.py` 先在 operator init 内、再跨 backbone seeds 聚合。
10. `gpu_validate_all.py` 是唯一正式 GPU orchestration entrypoint。

GPU scripts 只调 canonical APIs，不复制模型、trainer 或 evaluator。
