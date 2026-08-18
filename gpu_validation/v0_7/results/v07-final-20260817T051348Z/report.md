# V0.7 residual decision report

## Residual Learnability

**STRONG**

## Closed-Loop Utility

**POSITIVE**

## Memory Class

**INCONCLUSIVE**

Effective history: `None` steps; physical time `None`. Confidence: `LIMITED`.

Model/H selection used validation residual NRMSE only. Test metrics were read once after selection; no test-set oracle selection was used.

## Selected-model evidence

| backbone/data seed | selected test R2 | utility | median field gain | physics pass |
|---:|---:|:---:|---:|---:|
| 47 | 0.983472 | POSITIVE | 0.813362 | 1.000 |
| 53 | 0.971695 | POSITIVE | 0.848617 | 1.000 |
| 59 | 0.941242 | POSITIVE | 0.723607 | 1.000 |

## History-length sweep

| H | physical time | ordered NRMSE mean +/- std | ordered R2 | field RMSE mean +/- std | instantaneous NRMSE | shuffled NRMSE | validation gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.205523 +/- 0.0706 | 0.952515 | 0.00190324 +/- 0.000753 | 0.205523 | n/a | 0 |
| 2 | 0.0384481 | 0.214488 +/- 0.0998 | 0.943984 | 0.00187491 +/- 0.000584 | 0.236295 | 0.266969 | 0.110365 |
| 4 | 0.116901 | 0.227984 +/- 0.075 | 0.942094 | 0.00205544 +/- 0.000595 | 0.222926 | 0.249414 | -0.024315 |
| 8 | 0.275777 | 0.205064 +/- 0.062 | 0.953842 | 0.00187287 +/- 0.000695 | 0.30012 | 0.223056 | 0.0857319 |
| 16 | 0.595722 | 0.210697 +/- 0.102 | 0.945267 | 0.00198413 +/- 0.00077 | 0.237155 | 0.256006 | 0.108439 |

Physics acceptance requires both the absolute physical limit and a baseline-relative tolerance. Closure burden must also remain below its configured bound.

H=1 is a paired Markovian control. ACF remains auxiliary; finite-history evidence is not identification of an exact Mori-Zwanzig kernel.
