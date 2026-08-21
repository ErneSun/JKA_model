# Controlled cylinder-wake extension

V0.9 保持 V0.8 的二维 cylinder geometry、D2Q9-BGK solver、固定黏度和状态 `[u,v,p]`，唯一新控制量是
入口速度。因黏度与直径固定，`Re(t)=U_inf(t)D/nu`。

默认训练分布为 `Re: 80 -> 120`，对应 lattice inlet 始终低于 0.12 的 low-Mach 上限。两类 schedule：

- abrupt：在记录的 transition index 处执行 step；
- smooth：使用 half-cosine ramp，保留 transition 前后完整窗口。

每条 trajectory 在 metadata 中保存 causal `[Re_t,U_inf(t)]` condition series、schedule type、transition
index 和 ramp steps。Split 按完整 trajectory 且按 schedule family 分层，禁止窗口跨 split。

Known-condition 可以使用当前已知控制；latent-inferred 禁止读取 condition series。V0.9 不测试未见 Re pair、
未见 ramp rate 或未见 transition family。
