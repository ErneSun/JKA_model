# Architecture

The field contracts are `[C,Nx,Ny]`, `[B,C,Nx,Ny]`, `[B,H,C,Nx,Ny]`, and
`[B,K,C,Nx,Ny]`. `KoopmanEncoder2D` uses only circularly padded convolutions and maps a
field to `z_k`. `ContinuousKoopmanCore` advances `z_k` with `exp(A*dt)`. The lightweight
`TrainingDecoder2D` maps each predicted latent back to the fixed grid.

Only `E_K`, `A`, and `D_train` are trainable in `TrainStage.KOOPMAN`. The existing stage
registry, optimizer ownership check, batch contract, normalizer, split manifest,
fingerprint, checkpoint schema, and RNG utilities are reused.

The primary deliberate adjustment from the generic recommendation is that `src/train`
and `src/eval` contain the requested canonical entry points while reusable scientific
components stay in `jka_model`. GPU scripts import these functions and contain no loop.
