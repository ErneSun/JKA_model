# V0.1 implementation notes

- The installable package lives under `src/jka_model/`. This preserves the architecture's
  responsibility boundaries while avoiding ambiguous top-level packages such as `models`
  and `utils`.
- `ProblemBatch` uses the context/future contract from architecture revision 2.1. Read-only
  aggregate properties (`states_raw`, `states_model`, `actions`, `dts`, `parameters`, `mask`)
  expose the lower-level trajectory semantics without creating a second data contract.
- `ProblemSpec.normalization` declares preprocessing semantics. Fitted statistics remain in
  checkpoint `normalizer_state`; V0.1 does not implement a normalizer.
- V0.1 exact-resume scope is epoch-boundary RNG/config/state restoration. Sampler and
  mid-batch state are intentionally outside this version.
- The repository did not contain existing Python code or dependency configuration to reuse.

No mathematical definition from architecture revision 2.1 was changed.

