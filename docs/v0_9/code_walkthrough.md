# V0.9 code walkthrough

1. `config/schema.py` 定义 condition、adaptive、training 和 evaluation contracts。
2. `data/cylinder_wake_2d.py` 生成 fixed-viscosity smooth/abrupt trajectories。
3. `adaptive/handoff.py` 执行 3/3 V0.8 readiness audit。
4. `adaptive/cache.py` 保存 latent、nominal residual 与 causal condition series。
5. `adaptive/models.py` 实现冻结 context 和低秩 adaptive generator。
6. `adaptive/objectives.py` 实现 teacher-free curriculum、relative propagator growth 和 burden gate。
7. `observables/base.py` 定义问题无关的训练、评估和 gate 接口。
8. `problems/cylinder_observables.py` 独立拥有 cylinder 的速度、涡量、散度、壁面、升阻力数学。
9. `adaptive/physics.py` 只负责冻结 decoder、数据 provenance 与梯度桥接。
10. `evaluation/gates.py` 实现方向、绝对/相对阈值、分辨率和三态结果。
11. `train/train_v0_9.py` 只优化 operator adapter，并保存 exact-resume checkpoint 与课程日志。
12. `adaptive/rollout.py` 在 initial context 后完全使用 predicted latent history并记录 trust gate。
13. `eval/evaluate_v0_9.py` 计算 residual decomposition、controls、rollout 和 adapter observables。
14. `adaptive/reporting.py` 独立聚合机制/动态/observable gates，再形成原 joint decision。
15. `gpu_validate_all.py` 执行受约束 rank selection，是唯一正式全流程 GPU entrypoint；
    `gpu_reassess_existing.py` 只重算已有 checkpoint 的 locked-test 与报告。

GPU scripts 只调 canonical APIs，不复制模型、trainer 或 evaluator。
