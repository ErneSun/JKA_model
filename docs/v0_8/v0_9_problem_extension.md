# V0.9 interface preparation — no implementation

V0.9 可继续使用相同圆柱几何和 PDE，引入经过审计的 `Re(t)` 或 inlet condition metadata，并区分 smooth
与 abrupt transition。V0.8 已保留静态 parameter tensor 和统一 `c_t` downstream interface。

只有 V0.8 在多种子下支持 context、closed-loop utility 和 physics non-inferiority 后，V0.9 才允许研究

\[
c_t\rightarrow\eta_t\rightarrow A_t.
\]

这里的“支持”采用联合门槛：context/rank、adequacy、R3 history、全部 rollout horizons、最长 horizon
以及 physics 必须在同一 backbone seed 上同时稳定。V0.8 可以获得 scientific support，但若这些证据只在
不同 seeds 上分别成立，或只有 2/3 backbone 联合成立，V0.9 readiness 仍必须是 `NOT_READY`。进入算子
自适应要求 3/3 backbone seed 联合通过；V0.8 自身 scientific support 仍采用原定的 2/3 门槛。

本版本没有 condition schedule、change label、eta head、低秩 update、adaptive matrix exponential 或
operator regularizer。此文件是 handoff contract，不是已实现功能。
