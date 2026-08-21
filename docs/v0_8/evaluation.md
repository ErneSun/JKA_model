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

正式判定必须实际使用配置的 `min_context_effective_rank`；R3 的 history MLP 与 Attention 都必须通过
real-history versus shuffled-history control。Adequacy 至少要求 test `R2 >= 0` 且 correlation `>= 0.5`，
并单独报告为 calibrated/uncalibrated，不允许被 residual 指标掩盖。

Closed-loop 对每个 horizon 单独计算 context/Koopman-only mean RMSE 与 relative gain。`POSITIVE` 要求所有
配置 horizon 都达到 material gain，最长 horizon 另外形成独立 nested-seed 判定。脱涡频率只在最长窗口
做 context-versus-Koopman 非劣比较，避免短窗口 FFT 分辨率造成伪失败。

`V0.9 READY` 比 V0.8 `SUPPORTED` 更严格：context/rank、adequacy、history、all-horizon rollout、longest
horizon 和 physics 必须在同一 backbone seed 内同时达到 context-init consistency，然后再跨 backbone
要求 3/3 backbone 全部通过。不得把不同 seed 各自通过的指标拼接成 READY。V0.8 scientific support
本身仍使用预先约定的 2/3 seed consistency，二者不可混淆。
