# Technology adoption declaration

## Adopted technologies

### JEPA latent prediction

- Technology: future-embedding prediction with stop-gradient targets
- Source paper / official implementation: I-JEPA and V-JEPA papers/repositories
- Year: 2023–2024
- Decision: `ADOPT`
- Why adopted: predictive representation objective without a new inference module
- Where used in V0.6: future target encoding
- Implementation files: `field_jepa.py`, `field_jepa_koopman.py`
- How validated: semantic-separation and gradient-ownership tests
- Observed benefit: not yet measured; pending matched GPU study
- Observed cost: one extra target-encoder forward during JEPA training

### EMA target encoder

- Technology: frozen same-architecture momentum target after optimizer updates
- Source paper / official implementation: official I-JEPA/V-JEPA/V-JEPA 2 code
- Year: 2023–2026
- Decision: `ADOPT`
- Why adopted: mature, auditable slowly moving target mechanism
- Where used in V0.6: model target and `EMATracker`
- Implementation files: `field_jepa_koopman.py`, `ema.py`, `train_v0_6.py`
- How validated: sync/freeze/optimizer/formula/order/count/checkpoint tests
- Observed benefit: mechanism correctness confirmed; science pending GPU
- Observed cost: target parameters, checkpoint storage and EMA overhead

### JEPA–Koopman conceptual connection

- Technology: near-identity Koopman predictor as an inductive bias
- Source paper / official implementation: AAAI 2026 Koopman-invariants/JEPA paper
- Year: 2026
- Decision: `ADOPT AS DIAGNOSTIC MOTIVATION`
- Why adopted: motivates measuring the finite-time Koopman map
- Where used in V0.6: small/median/large `dt` diagnostic
- Implementation files: `v0_6_diagnostics.py`
- How validated: zero-generator test and evaluation artifact
- Observed benefit: theoretical regime becomes measurable
- Observed cost: negligible evaluation computation

## Reviewed but deferred

- V-JEPA/V-JEPA 2 ViT masking, dense/deep supervision and action predictor: changes
  architecture and the scientific variable.
- LeWorldModel Gaussian regularization/end-to-end no-EMA: retain for a future ablation.
- Phys-JEPA physical/residual decomposition: would introduce forbidden residual latent.
- PDE JEPA action-conditioned planning: current problem has no actions.
- PI-JEPA: withdrawn on arXiv; not adopted or used as evidence.
