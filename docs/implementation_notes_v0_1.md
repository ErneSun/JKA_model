# V0.1 implementation notes

- The installable package lives under `src/jka_model/`. This preserves the architecture's
  responsibility boundaries while avoiding ambiguous top-level packages such as `models`
  and `utils`.
- `ProblemBatch` uses the context/future contract from architecture revision 2.2. As of the
  V0.2 compatibility cleanup, only canonical fields remain public; the earlier aggregate and
  `parameters`/`mask` aliases were removed to avoid a second naming contract.
- `LatentState` contains only `z_k` and optional `z_r`. Physics is represented by a separate
  `PhysicsConstraint` Protocol acting on raw-unit states; no physical diagnostic is a required
  latent coordinate.
- V0.1 defines no concrete constraint or probe. `training_decoder` is only a canonical future
  train-stage ownership name; no decoder module is implemented.
- `ProblemSpec.normalization` declares preprocessing semantics. Fitted statistics remain in
  checkpoint `normalizer_state`; V0.1 does not implement a normalizer.
- V0.1 exact-resume scope is epoch-boundary RNG/config/state restoration. Sampler and
  mid-batch state are intentionally outside this version.
- The repository did not contain existing Python code or dependency configuration to reuse.

The previous revision-2.1 architecture file was removed so that revision 2.2 remains the sole
active specification. Checkpoints carrying revision 2.1 are rejected by default.

No mathematical definition from architecture revision 2.2 was changed.
