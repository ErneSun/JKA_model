# V0.9 status

Implementation status: **PHASE 2 CLOSED AS NOT_SUPPORTED; PHASE 3 ENTRY AUDIT IMPLEMENTED**.

Local Phase-3 targeted status: **PASS (5 new mathematical/interface tests)**.

Phase-2 Fix-2 session `v09-added-p2-physical-20260824T105209Z` completed with numerical 18/18 and
physics 16/18, but rollout skill 0/18, observer 0/18 and no supported static/dynamic mechanism.
Phase 2 is frozen and will not be optimized further.

Phase 3 now implements raw-field reconstruction/round-trip/tangent audit, a stream-function physical
decoder candidate, and immutable `frozen/joint/from_scratch` route contracts. Trainable representation
routes are forbidden from using the stale frozen-latent cache.

首次完整 GPU session `v09-full-20260821T033247Z` 的 workflow 为 PASS，但 scientific mechanism 为
`NOT_SUPPORTED`：短期改善，H32/H80 与 physics 退化，selected rank 达到旧候选上限 8。该证据不会被覆盖。

已实现：

- strict confirmatory 与 supported exploratory 两类可审计 V0.8 handoff；
- 12 类升阶、降阶、循环、速率和停留时间 controlled inlet data；
- frozen-context condition/history factorized low-rank adaptive generator；
- known/latent condition separation；
- exact resume；
- bounded coordinates、learned trust gate 和 H4/H8/H16/H32 teacher-free curriculum；
- relative nominal propagator-growth 与 frozen-decoder raw-unit physical objective；
- problem-independent metric gates、FFT resolution floor 与 `INCONCLUSIVE` 状态；
- 通用 frozen-decoder observable adapter、H4/H8/H16 multi-horizon objective、lift/drag global anchors；
- 不重训已有 18 checkpoints 的 locked-test reassessment；
- long-horizon/burden constrained rank selection；
- one-step/rollout/physical/operator evaluation；
- nested aggregation、compact report 与单命令 GPU workflow。
- causal `Re/U/dRe_dt` observer、condition-only control、isolated history shuffle 与
  matched-present/different-history audit；
- 条件中心化动态创新、static/dynamic Frobenius orthogonality，且不强制非零动态方差。

尚未声明：修订后的 adaptive Koopman scientifically supported、V1.0 ready。两项只能由新 ID 的正式 RTX 5080
结果决定。
