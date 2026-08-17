# GPU validation checklist

- [ ] Command prints resolved ID and CUDA preflight.
- [ ] Every major stage prints START/PASS/FAIL.
- [ ] V0.6 checkpoint SHA-256 appears in each seed’s provenance.
- [ ] Residual cache reports online-only target semantics.
- [ ] Backbone trainable parameter count is zero.
- [ ] H sweep is exactly `[1,2,4,8,16]`; H=1 has Markovian semantics.
- [ ] Every ordered-history model has a parameter-matched instantaneous control.
- [ ] Every H>1 model has a shuffled-history control preserving current state/next dt/target.
- [ ] Each backbone/data seed has all closure initialization seeds `[101,211,307]`.
- [ ] H=1 history and instantaneous controls are training-identical under paired initialization.
- [ ] All comparisons share checkpoint/cache/data/split/normalizer/evaluation trajectories.
- [ ] Model/H selection uses validation residual NRMSE only; test is not used as an oracle.
- [ ] Formal rollout uses predicted history.
- [ ] Physics is evaluation-only.
- [ ] Closure burden curves, physical H, parameter counts, gains, and marginal gains are recorded.
- [ ] `completion.json`, `history_sweep.csv`, `memory_classification.json`, plots, and both reports exist in results.
- [ ] Final report begins with learnability, utility, and memory class.
- [ ] Scientific status is kept separate from implementation status.
