# Technology adoption declaration

| Technique | Decision | Scope/reason |
|---|---|---|
| Problem-owned CNN padding | ADOPT | Circular for periodic fields, zero for fixed cylinder boundaries |
| Offline D2Q9-BGK cylinder data | ADOPT WITH LIMITS | Minimum reproducible generator; internal benchmark only |
| Instantaneous MLP | ADOPT FOR R2/CONTROL | Current-state learnability and R3 control |
| History MLP | ADOPT FOR R3 CONTROL | Tests whether history matters without Attention |
| Small causal Attention | ADOPT FOR CONFIRMED R3 CANDIDATE | Short latent history; selected only on validation |
| Mamba/Mamba-2 | DEFER | H is short and simultaneous adoption obscures attribution |
| PhysicsNeMo/Transolver | DEFER | Spatial operator replacement is outside the latent-time hypothesis |
| Adaptive `A_t`, `eta_t` | FORBIDDEN IN V0.8 | V0.9 scientific question |
| Persistent `z_R` | FORBIDDEN IN V0.8 | Reassess remaining residual only after operator adaptation |

