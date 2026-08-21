# V0.9 status

Implementation status: **PHASE 1/2 REVISION IMPLEMENTED; NEW FORMAL GPU SCIENCE PENDING**.

Local targeted status: **PASS (22 targeted tests)**.

首次完整 GPU session `v09-full-20260821T033247Z` 的 workflow 为 PASS，但 scientific mechanism 为
`NOT_SUPPORTED`：短期改善，H32/H80 与 physics 退化，selected rank 达到旧候选上限 8。该证据不会被覆盖。

已实现：

- strict confirmatory 与 supported exploratory 两类可审计 V0.8 handoff；
- smooth/abrupt controlled inlet data；
- frozen-context low-rank adaptive generator；
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

尚未声明：修订后的 adaptive Koopman scientifically supported、V1.0 ready。两项只能由新 ID 的正式 RTX 5080
结果决定。
