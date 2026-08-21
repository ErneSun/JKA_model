# Stability and identifiability

可辨识性由三条硬边界保证：冻结 `A0`、冻结 V0.8 context、禁止 additive residual。这样 `Gamma_op` 才能回答
fixed-generator mismatch 是否被低秩 operator change 解释。

优化项可包含 operator burden、smooth-schedule eta variation 和 symmetric-part growth proxy：

\[
\alpha_{sym}(A)=\lambda_{max}\left((A+A^\top)/2\right).
\]

该 proxy 只是瞬时增长上界，不是物理因果或完整 non-normal stability 证明。最终稳定性由 teacher-free all-horizon、
longest-horizon 和 physical metrics 决定。禁止 hard clipping latent/physical state。

严格收缩参数化没有作为默认方案，因为 cylinder wake 具有持续振荡吸引子；把所有 latent direction 强制衰减可能
删除真实 shedding dynamics。显式 learned trust gate 也仅保留为后续独立 ablation。
