# V0.6 overview

V0.6 is exactly `PhysicsConstraint + z_K + JEPA`. It starts from a validated V0.5
checkpoint and adds a frozen EMA copy of the field encoder. The online encoder,
continuous Koopman generator, decoder, data split, normalizer, and complete V0.5 loss
remain the baseline. No attention, residual state, action/control input, extra predictor,
masking pipeline, or latent normalization was added.

Current status: local CPU implementation `PASS`; GPU validation `NOT RUN`; scientific
acceptance `PENDING_GPU`. See [status.md](status.md) and the remote workflow under
`gpu_validation/v0_6/`.
