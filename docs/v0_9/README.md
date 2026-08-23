# V0.9 — Context-Conditioned Low-Rank Adaptive Koopman

V0.9 在相同 cylinder-wake 几何和 PDE 上引入受控的 smooth/abrupt inlet 变化，并使用冻结的
V0.8 dynamic context 驱动低秩时变 Koopman generator；主实验禁止 additive residual correction，
从而保持 operator adaptation 的可辨识性。

核心链路：

\[
z_{t-H:t}\rightarrow c_t\rightarrow\eta_t\rightarrow
A_t=A_0+U\operatorname{diag}(\eta_t)V^\top\rightarrow e^{A_t\Delta t}z_t.
\]

长时稳定性修订将坐标限制为
`eta = sigmoid(gate) * eta_max * tanh(raw_eta)`，并采用 H4/H8/H16/H32 teacher-free
课程、相对 nominal propagator-growth 约束和冻结 decoder 的 problem-owned observable anchor。
阶段 1 的独立门控、多尺度 observable 与复评合同见 `evaluation_and_observable_revision.md`；
V0.9 Added Phase 2 的条件/历史因子化、成对辨识和有效负结论见
`../v0_9_added/phase_2_implementation.md`。

确认性结论只有在 V0.8 严格达到 3/3 backbone/data seed 联合 readiness 时才允许；显式
`supported` handoff 只能形成探索性条件证据。V0.9 同时比较
known-condition 与 latent-inferred-condition；后者不得读取 Re、入口速度、change label 或 transition time。

V0.9 不实现 unseen-transition generalization、persistent `z_R`、additive closure 或 joint fine-tuning。
首次 GPU 结果和 Phase-1 结果均作为证据保留；Phase 2 必须使用新 validation ID 重新训练并形成独立证据。
