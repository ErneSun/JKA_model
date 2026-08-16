# V0.5 incremental scientific GPU validation

- validation id: `v05-final-20260814`
- workflow status: **PASS**
- scientific status: **PENDING_REVIEW**
- overall acceptance: **PENDING_RESEARCHER_REVIEW**
- seeds: `[47, 53, 59]`
- reused prior evidence: CUDA preflight/parity, profiler, and exact-resume validation
- reused no-physics seed checkpoints: `[]`

## Per-seed hard gates

| Seed | Frequency | Decay | Stability | Reconstruction | Forecast | Mass | Operator |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 47 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 53 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 59 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

## Physics aggregate

| Metric | Mean | Median | Std |
|---|---:|---:|---:|
| frequency_relative_error | 0.00245508 | 0.00247842 | 7.06964e-05 |
| decay_relative_error | 0.158926 | 0.159226 | 0.00585472 |
| spectral_abscissa | -7.85794e-05 | -9.62086e-05 | 3.98913e-05 |
| reconstruction_rmse | 0.00327191 | 0.00326726 | 0.000430151 |

## Median physics vs no-physics relative change

| Horizon | RMSE | Mass drift | Operator |
|---|---:|---:|---:|
| short | +17.776% | +79.577% | +38.592% |
| medium | -3.519% | +90.041% | +27.771% |
| long | -0.782% | +116.731% | +16.036% |

## Forecast non-inferiority vs no-physics

Threshold: 5.000% of persistence RMSE.

| Horizon | Median added RMSE / persistence RMSE | Pass |
|---|---:|---:|
| short | +1.413% | PASS |
| medium | -0.026% | PASS |
| long | -0.011% | PASS |

## Constraint non-inferiority vs no-physics

Threshold: 10.000% of each constraint hard limit.

| Horizon | Added mass drift / limit | Added operator MSE / limit | Pass |
|---|---:|---:|---:|
| short | +2.534% | +2.426% | PASS |
| medium | +2.908% | +0.156% | PASS |
| long | +4.028% | +0.034% | PASS |
