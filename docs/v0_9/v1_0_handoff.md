# V1.0 handoff — not implemented

只有 V0.9 known 与 latent-inferred adaptive Koopman 均通过，且 3/3 backbone seeds 联合满足 all-horizon、
longest-horizon、physics、stability、controls 和 `Gamma_op`，才标记 `V1.0_READY=YES`。

V1.0 才允许测试：未见 Re pairs、transition times、ramp rates、reverse/repeated transitions 与更广 operating
range。V0.9 数据属于训练分布内 controlled changes，不能提前形成这些泛化声明。

V1.0 仍不自动引入 additive closure；remaining residual closure 属于 V1.1。
