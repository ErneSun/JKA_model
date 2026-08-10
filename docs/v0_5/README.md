# V0.5 — 2-D PDE Koopman and reproducible CPU/GPU validation

V0.5 adds one bounded capability: a learned continuous-time Koopman model for an
analytic two-dimensional periodic advection-diffusion field, with differentiable
raw-unit physics constraints and a separate GPU validation package.

Local status and remote scientific status are intentionally separate. A CPU tiny run
validates correctness and integration; it is not scientific evidence for GPU acceptance.

Core route:

```text
raw [B,C,Nx,Ny]
  -> train-only channel normalization
  -> circular CNN encoder E_K
  -> exact exp(A*dt) closed-loop rollout
  -> training decoder D_train
  -> differentiable inverse normalization
  -> raw mass and PDE-operator constraints
```

No JEPA, EMA target, residual state, recurrent/attention model, control, RL, or V0.6
feature is included.

## Documentation map

- [Architecture](architecture.md)
- [Physics](physics.md)
- [Training](training.md)
- [Testing](testing.md)
- [Evaluation](evaluation.md)
- [Problem adapter contract](problem_contract.md)
- [Latest technology review](latest_tech_review.md)
- [References](references.md)
- [Independent GPU validation package](../../gpu_validation/v0_5/README.md)
