# V0.9 — Context-Conditioned Low-Rank Adaptive Koopman

V0.9 在相同 cylinder-wake 几何和 PDE 上引入受控的 smooth/abrupt inlet 变化，并使用冻结的
V0.8 dynamic context 驱动低秩时变 Koopman generator；主实验禁止 additive residual correction，
从而保持 operator adaptation 的可辨识性。

核心链路：

\[
z_{t-H:t}\rightarrow c_t\rightarrow\eta_t\rightarrow
A_t=A_0+U\operatorname{diag}(\eta_t)V^\top\rightarrow e^{A_t\Delta t}z_t.
\]

只有 V0.8 严格达到 3/3 backbone/data seed 联合 readiness 时才允许训练。V0.9 同时比较
known-condition 与 latent-inferred-condition；后者不得读取 Re、入口速度、change label 或 transition time。

V0.9 不实现 unseen-transition generalization、persistent `z_R`、additive closure 或 joint fine-tuning。
