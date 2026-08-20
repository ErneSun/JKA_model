# V0.7 final scientific decision

**RESIDUAL SIGNIFICANCE:** `NEGLIGIBLE`  
**RESIDUAL LEARNABILITY:** `STRONG`  
**CONDITIONAL HISTORY GAIN:** `PRESENT`  
**CLOSED-LOOP CLOSURE UTILITY:** `NEUTRAL`  
**MEMORY CLASS:** `INCONCLUSIVE`  
**PHYSICS ACCEPTANCE:** `PASS`  
**RESIDUAL ROUTE:** `R0`  
**V0.8 RECOMMENDATION:** `NO_CONTEXT_CORRECTION_REQUIRED`

## Validation-first evidence chain

Validation route: `R0`; locked test confirmation: `True`. Confidence: `HIGH`.

Residual significance threshold: `0.01`. Validation significance by backbone seed: `{'47': 0.00017945407297717588, '53': 0.00011996508212075566, '59': 6.937826505445057e-05}`.

Predictability is `1 - best Markovian validation standardized MSE / zero standardized MSE`, using train-split per-dimension RMS. Closure family, H, history decision, and preliminary route use validation only; test is a locked confirmation and never selects a configuration.

## Hierarchical history evidence

Locked H: `16`. Effective secondary-memory H: `None` steps; physical time `None`.

Closure-init consistency: `{'47': {'2': 0.0, '4': 0.0, '8': 0.0, '16': 0.3333333333333333}, '53': {'2': 0.6666666666666666, '4': 0.3333333333333333, '8': 0.3333333333333333, '16': 0.6666666666666666}, '59': {'2': 0.3333333333333333, '4': 0.6666666666666666, '8': 0.3333333333333333, '16': 0.6666666666666666}}`.

Backbone/data consistency: `{'labels': {'47': 'ABSENT', '53': 'PRESENT', '59': 'PRESENT'}, 'support_fraction_by_history': {'2': 0.3333333333333333, '4': 0.3333333333333333, '8': 0.0, '16': 0.6666666666666666}}`.

## Route-locked closed-loop evidence

| backbone/data seed | selected variants | utility | median field gain |
|---:|:---|:---:|---:|
| 47 | zero(H=1) | NEUTRAL | 0 |
| 53 | zero(H=1) | NEUTRAL | 0 |
| 59 | zero(H=1) | NEUTRAL | 0 |

## History-length sweep

| H | physical time | ordered NRMSE mean +/- std | ordered R2 | field RMSE mean +/- std | instantaneous NRMSE | shuffled NRMSE | validation gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.205523 +/- 0.0706 | 0.952515 | 0.00190324 +/- 0.000753 | 0.205523 | n/a | 0 |
| 2 | 0.0384481 | 0.214488 +/- 0.0998 | 0.943984 | 0.00187491 +/- 0.000584 | 0.236295 | 0.266969 | 0.110365 |
| 4 | 0.116901 | 0.227984 +/- 0.075 | 0.942094 | 0.00205544 +/- 0.000595 | 0.222926 | 0.249414 | -0.024315 |
| 8 | 0.275777 | 0.205064 +/- 0.062 | 0.953842 | 0.00187287 +/- 0.000695 | 0.30012 | 0.223056 | 0.0857319 |
| 16 | 0.595722 | 0.210697 +/- 0.102 | 0.945267 | 0.00198413 +/- 0.00077 | 0.237155 | 0.256006 | 0.108439 |

Physics acceptance is the logical AND of the inherited V0.5 absolute limit, zero-closure non-inferiority, and closure burden <= 0.25.

Absolute pass: `True`; non-inferiority pass: `True`; burden pass: `True`.
A failed locked test confirmation or failed physics acceptance forces the final residual route to INCONCLUSIVE.

H=1 is a paired Markovian control. ACF remains auxiliary; finite-history evidence is not identification of an exact Mori-Zwanzig kernel.
