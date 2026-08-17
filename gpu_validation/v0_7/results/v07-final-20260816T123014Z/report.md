# V0.7 residual decision report

## Residual Learnability

**MODERATE**

## Closed-Loop Utility

**POSITIVE**

## Memory Class

**INCONCLUSIVE**

Effective history: `None` steps; physical time `None`. Confidence: `LIMITED`.

## Residual statistics and minimal baselines

Residual signal RMS: `0.000237328`; best held-out R2: `0.630625`.

| model | residual NRMSE | residual R2 | longest field RMSE |
|---|---:|---:|---:|
| zero | 1 | -0.0100628 | 0.0105776 |
| linear | 4.26387 | -25.2691 | 165330 |

## History-length sweep and controls

| H | physical time | ordered NRMSE | ordered R2 | ordered field RMSE | instantaneous NRMSE | shuffled NRMSE | joint gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 1.23857 | -0.765493 | 0.00788777 | 1.69746 | n/a | 0 |
| 2 | 0.0384481 | 1.26581 | -0.951233 | 0.00807788 | 1.23341 | 2.28346 | -0.0230481 |
| 4 | 0.116901 | 1.4914 | -1.76927 | 0.0167878 | 1.14104 | 2.10555 | -0.66623 |
| 8 | 0.275777 | 1.25122 | -0.727589 | 0.00631225 | 1.2709 | 1.66503 | 0.094764 |
| 16 | 0.595722 | 1.25338 | -0.672885 | 0.00760688 | 1.1746 | 1.54959 | 0.0118268 |

## Evidence boundary

H=1 is the operational Markovian baseline: current latent, next dt, and parameters. H>1 is accepted as memory evidence only when it beats parameter-matched instantaneous and shuffled-history controls in teacher-forced and closed-loop tests.

Physics metrics are evaluation-only non-inferiority gates, not trained losses. Autocorrelation is auxiliary and does not determine the memory class. This is a finite-history closure diagnosis, not identification of an exact Mori-Zwanzig kernel.
