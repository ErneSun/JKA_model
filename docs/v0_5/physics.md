# Physics constraints

The decoder outputs model-space fields. `ChannelStandardizer.inverse_transform` is then
called inside the autograd graph; mass and PDE residuals therefore act in physical units
and propagate gradients to encoder, generator, and decoder.

Mass uses `sum(u * dx * dy)` over both spatial axes. The operator residual is

`(u_next-u_prev)/dt - 0.5*(F(u_prev)+F(u_next))`,

where `F(u)=-cx Dx(u)-cy Dy(u)+nu(Dxx(u)+Dyy(u))`. All spatial differences use periodic
`roll` on endpoint-free grids. Periodicity is structural and is not represented by the
incorrect equality `u[...,0] == u[...,-1]`.

Physics weights warm up within the same Koopman training stage. There is no latent
`z_phys` and no V0.6 physical encoder.
