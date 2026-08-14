# Physics constraints

The decoder outputs model-space fields. `ChannelStandardizer.inverse_transform` is then
called inside the autograd graph; mass and PDE residuals therefore act in physical units
and propagate gradients to encoder, generator, and decoder.

Mass uses `sum(u * dx * dy)` over both spatial axes and divides the mass change by the
cell-weighted L1 magnitude of the reference state, so training and evaluation use the same
dimensionless scale. For the constant-coefficient periodic
advection-diffusion reference, operator consistency uses the exact Fourier evolution

`u_hat_next = exp((-i*(cx*kx+cy*ky)-nu*(kx^2+ky^2))*dt) * u_hat_prev`.

This matches the analytical data generator without the truncation bias of a second-order
finite-difference/trapezoidal residual. It remains differentiable and runs on CPU or CUDA through
`torch.fft`. The generic finite-difference constraint is retained for non-spectral regression
tests and future adapters. Periodicity is structural and is not represented by the incorrect
equality `u[...,0] == u[...,-1]`.

Physics weights warm up within the same Koopman training stage. There is no latent
`z_phys` and no V0.6 physical encoder.
