# V0.9 interface preparation — no implementation

V0.9 可继续使用相同圆柱几何和 PDE，引入经过审计的 `Re(t)` 或 inlet condition metadata，并区分 smooth
与 abrupt transition。V0.8 已保留静态 parameter tensor 和统一 `c_t` downstream interface。

只有 V0.8 在多种子下支持 context、closed-loop utility 和 physics non-inferiority 后，V0.9 才允许研究

\[
c_t\rightarrow\eta_t\rightarrow A_t.
\]

本版本没有 condition schedule、change label、eta head、低秩 update、adaptive matrix exponential 或
operator regularizer。此文件是 handoff contract，不是已实现功能。

