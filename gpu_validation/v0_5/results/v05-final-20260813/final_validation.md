# V0.5 incremental scientific GPU validation

- validation id: `v05-final-20260813`
- workflow status: **PASS**
- scientific status: **FAIL**
- overall acceptance: **NOT_ACCEPTED**
- seeds: `[47, 53, 59]`
- reused prior evidence: CUDA preflight/parity, profiler, and exact-resume validation

## Per-seed hard gates

| Seed | Frequency | Decay | Stability | Reconstruction | Short | Medium | Long |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 47 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 53 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 59 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

## Physics aggregate

| Metric | Mean | Median | Std |
|---|---:|---:|---:|
| frequency_relative_error | 0.00239213 | 0.00239567 | 3.09645e-05 |
| decay_relative_error | 0.156451 | 0.155932 | 0.00105147 |
| spectral_abscissa | -7.95601e-05 | -8.04142e-05 | 2.76571e-06 |
| reconstruction_rmse | 0.00341464 | 0.00354618 | 0.000210925 |

## Median physics vs no-physics relative change

| Horizon | RMSE | Mass drift | Operator |
|---|---:|---:|---:|
| short | +26.730% | +37.765% | +60.954% |
| medium | -0.516% | +38.892% | +38.521% |
| long | -3.132% | +54.298% | +24.827% |
