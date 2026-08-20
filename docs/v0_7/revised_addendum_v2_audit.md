# V0.7 Revised Addendum v2 implementation audit

Date: 2026-08-18

This audit reconciles the existing V0.7 implementation with the revised
Koopman-attention-adaptive roadmap. The frozen V0.6 online JEPA-Koopman model and
the existing V0.7 residual target remain the mathematical baseline.

## Reused without semantic change

- Online-encoder residual definition and exact `exp(A dt)` Koopman step.
- Frozen V0.6 encoder, Koopman operator, decoder, and EMA target.
- Zero, linear, instantaneous, causal-history, and shuffled-history controls.
- H=1 paired Markovian equivalence, predicted-history closed-loop rollout, and
  analytical parameter matching.
- Independent backbone/data and closure-initialization seed hierarchy.
- Formal GPU grid: 3 backbone/data seeds, 3 closure seeds, H in
  `[1, 2, 4, 8, 16]`, exactly 144 run identities.

## Revised primary scientific decision

The primary evidence chain is now:

```text
residual magnitude diagnostic S_R
  + Markovian predictability P_R
  -> conditional causal-history gain G_H
  -> R1 / R2 / R3 / INCONCLUSIVE
```

`S_R` describes scale but never discards a residual. This preserves small
one-step discrepancies whose effect may accumulate during long rollouts.

Primary prediction and gain comparisons use the per-latent-dimension training
RMS standardized MSE. Validation selects the admissible closure and H; the
locked decision is then checked against held-out test evidence. Closure-seed
consistency is aggregated inside each backbone/data seed before backbone-level
consistency is evaluated.

The earlier `MARKOVIAN / SHORT_MEMORY / LONG_MEMORY_CANDIDATE` result is retained
only as a secondary diagnostic. It no longer selects the V0.8 architecture.

## Hard acceptance gates

- The formal identity set must exactly equal the configured 144-item matrix;
  duplicates, missing controls, unexpected controls, broken H=1 pairs, or
  provenance mismatches fail comparison.
- Physical acceptance is the logical AND of inherited absolute V0.5 limits,
  zero-closure non-inferiority, and closure burden `<= 0.25`.
- A failed locked test confirmation or any failed physical gate changes the
  final route to `INCONCLUSIVE`.

## New auditable artifacts

- `evaluation/residual_structure_assessment.json` (primary result)
- `evaluation/memory_classification.json` (secondary diagnostic)
- `evaluation/history_sweep.csv` with raw, standardized, per-run, physics, and
  provenance fields
- `reports/residual_decision_report.md`
- `reports/v0_8_route_recommendation.md`
- Seven formal diagnostic plots with seed variability or error bars
- Strict `completion.json` and `failure.json`

## Explicitly deferred

V0.7 does not introduce Attention, Transformer, recurrent memory, adaptive
Koopman operators, online adaptation, or a new scientific benchmark. Those are
eligible only after the R1-R3 evidence route is confirmed.

## Verification state

Focused local V0.7 tests and the necessary V0.6/config/checkpoint compatibility
tests passed before the R1–R3 route revision. Per explicit request, no tests were
executed for this decision-only change. Existing 144-record evaluation evidence
can be reclassified without retraining; its generated reports remain historical
until that reclassification is requested.
