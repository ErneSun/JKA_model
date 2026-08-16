# V0.6 JEPA GPU final scientific review

- Validation ID: `v06-final-20260816T030842Z`
- Training commit(s): `c49f0eefe09e9c8e7ecbda4ba97f31f1abe8fa2a`
- Review date: `2026-08-16`
- GPU validation: **PASS**
- Scientific acceptance: **PASS_AFTER_REVIEW**
- Scope: 2D periodic constant-coefficient single-Fourier-mode advection-diffusion

## Decision

All matched software, Koopman, physics, non-collapse, EMA, and online-only inference gates pass for seeds 47/53/59. JEPA improves long-rollout RMSE for every seed. The accepted claim is limited to the registered reduced analytical single-mode PDE experiment.

## Forecast effect

| Metric | Control mean | JEPA mean | JEPA relative change |
|---|---:|---:|---:|
| Short RMSE | `0.0025343925` | `0.0015172252` | `-40.13%` |
| Medium RMSE | `0.0023739128` | `0.0017043014` | `-28.21%` |
| Long RMSE | `0.015099141` | `0.010575901` | `-29.96%` |
| Long operator MSE | `2.4954014e-07` | `1.2690192e-07` | `-49.15%` |
| Long mass drift | `0.0012269837` | `0.0022402546` | `+82.58%` |

Long-rollout RMSE improves for all seeds: seed 47 `1.54%`, seed 53 `20.42%`, and seed 59 `63.71%`. JEPA therefore achieved a measured forecast optimization rather than only lowering its own auxiliary loss.

Mass drift is not uniformly improved: it improves for seed 47 and worsens for seeds 53/59. All horizons remain below the preregistered absolute `0.01` contract, so this is an accepted trade-off, not evidence that JEPA optimizes mass conservation.

## Per-seed evidence

| Seed | Control long RMSE | JEPA long RMSE | Ratio | JEPA long mass | JEPA long operator | Min latent std | Result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 47 | `0.015246667` | `0.015012528` | `0.984643` | `0.0010955867` | `2.4176201e-07` | `0.496839` | PASS |
| 53 | `0.013422114` | `0.01068108` | `0.795782` | `0.0019860858` | `1.0522531e-07` | `0.290488` | PASS |
| 59 | `0.016628642` | `0.0060340943` | `0.362874` | `0.0036390913` | `3.3718443e-08` | `0.213088` | PASS |

## Contract review

For both control and JEPA in every seed: training/evaluation are finite; frequency error is at most 5%; decay error is at most 20%; spectral abscissa is at most `1e-3`; every rollout horizon beats persistence; mass and operator errors remain below `0.01` and `1e-4`; latent std remains above `0.02`; optimizer and EMA update counts agree; the EMA target is absent from rollout inference; matched configs differ only in JEPA coefficients and descriptive tags.

## Compute cost

Mean runtime changes from `32.26` to `38.66` minutes (`+19.81%`). Mean peak GPU memory changes from `2756.0` to `2833.7` MiB (`+2.82%`).

## Interpretation boundary

The evidence supports JEPA as a useful latent predictive regularizer for this registered three-seed single-mode problem. It does not establish improvement for multimode PDEs, parameter OOD, CFD, experiments, or every physics metric. No inference-time module or Koopman propagation equation was changed.
