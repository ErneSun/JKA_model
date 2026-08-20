# Evaluation and decision protocol

顺序固定：physical gate → cylinder backbone acceptance → V0.7 route → validation-only family selection →
locked test confirmation → teacher-free rollout/physical review。

Backbone 至少检查 reconstruction、相对 persistence 的 rollout skill、latent non-collapse、脱涡主频保留、
散度/边界指标有限。Residual 目标和 split/normalizer/backbone 指纹必须一致。

R3 会在相同 3 backbone/data seeds × 3 context-init seeds 上训练 instantaneous、history MLP 和 causal
Attention 候选，仅用 validation standardized residual MSE 在两个 temporal family 间锁定最终 family。
训练函数不实例化 test dataset；只有锁定 checkpoint 后，`evaluate_v0_8` 才读取 test。

正式输出包括五个 evaluation 文件、十张证据图、nested-seed decision JSON 和 scientific report。
Closed-loop 在初始 history 后只回填 predicted latent，不使用 teacher forcing。物理比较同时解码
Koopman-only 与 context-residual probe，检查 velocity、vorticity、divergence、lift、drag、frequency、
boundary no-slip 和 closure burden。

`DYNAMIC CONTEXT: SUPPORTED` 需要：test residual 改善、init-seed 稳定、context ablation、非 collapse、
R3 history control、closed-loop/physics 无灾难退化。软件 PASS 不自动等于这些科学条件成立。

