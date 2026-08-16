# V0.7 problem classification

- `inherited_v0_6_periodic_advection_diffusion_2d`: primary scientific problem. It uses the validated V0.6 checkpoints and decides the V0.7 residual/closure result.
- `v0_7_synthetic_latent_memory`: deterministic mechanism diagnostic. It verifies the history-control machinery but cannot make V0.7 scientifically pass.

Future problem additions must declare `version_owner`, role, mathematical definition, compute budget, and whether they are permitted to contribute to scientific acceptance.
