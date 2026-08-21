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
课程、相对 nominal propagator-growth 约束和冻结 decoder 的 raw-unit physics anchor。完整数学与模块边界见
`stabilization_revision.md`。

只有 V0.8 严格达到 3/3 backbone/data seed 联合 readiness 时才允许训练。V0.9 同时比较
known-condition 与 latent-inferred-condition；后者不得读取 Re、入口速度、change label 或 transition time。

V0.9 不实现 unseen-transition generalization、persistent `z_R`、additive closure 或 joint fine-tuning。
首次 GPU 结果作为失败基线保留；修订版必须使用新 validation ID 重新形成独立证据。
