# Technology review

## Adopted

- Small causal multi-head self-attention: PyTorch 原生实现足以检验短 latent history 的科学问题，避免引入
  新依赖。Causal mask 与 future-access tests 是合同的一部分。
- Low-Mach D2Q9 BGK: 作为可重复、固定几何的 minimum offline generator；必须通过内部 grid/sanity gate，
  且明确不是 production CFD。
- Validation-first nested seeds: 3 backbone/data × 3 context init，避免把初始化随机性与物理数据随机性混合。

## Reviewed and deferred

- Mamba-2/structured state-space duality 提供长序列线性复杂度，但当前 H≤16、latent dimension 32；同时引入
  Attention 与 Mamba 会混淆首个 context 假设。只有确认 R3 且历史长度成为瓶颈后再比较。
- PhysicsNeMo/Transolver 面向大规模空间物理 token/operator learning；本版本的假设位于 frozen Koopman
  latent 的短时间 history，不需要替换已经验证的 field backbone。
- GRU 可作未来 temporal baseline，但 V0.7 finite-history MLP 已提供低复杂度 history control，本阶段不再
  增加第二个递归归纳偏置。

Primary references:

- Dao and Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through
  Structured State Space Duality* (Mamba-2), https://arxiv.org/abs/2405.21060
- *Mori–Zwanzig latent Koopman closure*, https://arxiv.org/abs/2310.10745
- NVIDIA PhysicsNeMo Transolver documentation,
  https://docs.nvidia.com/physicsnemo/latest/physicsnemo-sym/user_guide/neural_operators/transolver.html
- Schäfer and Turek cylinder benchmark,
  https://wwwold.mathematik.tu-dortmund.de/lsiii/cms/papers/SchaeferTurek1996.pdf

