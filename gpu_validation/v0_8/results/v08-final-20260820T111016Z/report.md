# V0.8 scientific report

PHYSICAL PROBLEM: cylinder_wake_2d  
BACKBONE STATUS: PASS  
V0.7 ROUTE ON NEW PROBLEM: R3  
CONTEXT FAMILY: HISTORY_MLP  
RESIDUAL PREDICTION: SUPPORTED  
HISTORY VALUE: SUPPORTED  
KOOPMAN ADEQUACY: CALIBRATED  
DYNAMIC CONTEXT: SUPPORTED  
CLOSED LOOP UTILITY: POSITIVE  
LONGEST HORIZON UTILITY: POSITIVE  
PHYSICS STATUS: PASS  
V0.9 OPERATOR-ADAPTATION READINESS: NOT_READY

## Aggregate evidence

- Residual NRMSE: 0.415268 ± 0.244994 (n=9)
- Residual R2: 0.769942 ± 0.338892
- History-over-shuffled gain: 0.818787 ± 0.258955
- Context effective rank: 2.43563 ± 0.716295
- Adequacy R2: -0.479483 ± 2.72796

## Teacher-free rollout by horizon

| Horizon | Mean relative gain | Ratio of mean RMSE | Material-gain fraction | n |
|---:|---:|---:|---:|---:|
| 8 | 0.764179 | 0.240461 | 1 | 9 |
| 16 | 0.631766 | 0.374574 | 0.888889 | 9 |
| 32 | 0.311873 | 0.698588 | 0.888889 | 9 |
| 80 | 0.045255 | 0.969182 | 0.888889 | 9 |

## Nested backbone support

| Seed | Context | Rank | Adequacy | History | Rollout | Longest | Physics | V0.9 joint |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 47 | 0.667 | 0.667 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | True |
| 53 | 0.333 | 0.333 | 0.333 | 1.000 | 0.667 | 0.667 | 0.667 | False |
| 59 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | True |

Formal nested run count: 9. Joint V0.9 support fraction: 0.666667; V0.9 required: 1. Context-init consistency within each backbone: 0.666667.
Compact audit complete: True.

The additive residual is a utility probe. A0 remains frozen; eta_t, adaptive A_t, and persistent z_R are absent. Attention weights, when available, are diagnostics rather than causal explanations.
