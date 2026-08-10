# Evaluation

Canonical Python evaluation function:

`src.eval.evaluate_v0_5.evaluate_v0_5`

Evaluation reconstructs the exact seeded dataset and uses only checkpointed normalizer,
split manifest, and model state. It reports fixed short, medium, and long held-out rollout
RMSE/relative L2, persistence baseline RMSE, mass drift, trapezoidal operator penalty,
learned/true angular frequency and Hz, selected eigenvalue decay, relative spectral errors,
latent standard-deviation range, finiteness, and device. Field diagnostics measure decoded
raw-unit predictions; rollout diagnostics are closed loop and never teacher forced.

Checkpoint roles are explicit: `best_forecast.pt` minimizes decoded validation forecast
MSE, `best_physics.pt` minimizes validation mass plus operator penalty, and `last.pt` is the
final resumable state. Formal evaluation defaults to `best_forecast.pt`; the other two are
available for diagnostic comparisons. Persistence repeats the initial raw field and is the
minimum baseline. Spectrum diagnostics inspect eigenvalues of the learned continuous
generator, while physics diagnostics remain in raw physical units.

CPU evaluation records `scientific_acceptance=PENDING_GPU`. GPU execution produces
measurements but does not automatically declare scientific PASS; the full/ablation and
acceptance checklist must be reviewed together.
