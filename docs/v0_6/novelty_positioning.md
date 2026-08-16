# Technical positioning

No priority or “first” claim is made. This records architecture differences and
candidate research gaps only.

| Work | JEPA | Dynamics | PDE | Koopman | Physics constraint | Difference in V0.6 |
|---|---:|---:|---:|---:|---:|---|
| I-JEPA | yes | no | no | no | no | transfers only EMA latent-target mechanics |
| V-JEPA | yes | video | no | no | no | field evolution without ViT masks |
| V-JEPA 2 | yes | video/action | no | no | no | no action predictor/planning stage |
| Koopman invariants + JEPA | idealized | time series | no | theoretical | no | continuous `A`, variable `dt`, decoded fields |
| JEPA PDE control | yes | controlled | yes | no | task-dependent | uncontrolled PDE and fixed Koopman predictor |
| Phys-JEPA | yes | time series | indirect | no | latent decomposition | no physical/residual split |
| LeWorldModel | JEPA-like | world model | no | no | no | retains EMA; no Gaussian prior |

Candidate contributions to test, not established novelty: continuous-time Koopman
generator as JEPA predictor; raw decoded physics constraints; strict online/EMA target
separation alongside Koopman consistency; variable-`dt` JEPA; and spectrum/
near-identity diagnostics inside the JEPA world model.
