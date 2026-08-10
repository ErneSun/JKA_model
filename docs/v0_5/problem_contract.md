# Problem contract

The reference equation is

`u_t + cx u_x + cy u_y = nu (u_xx + u_yy)`

on `[0,Lx) x [0,Ly)` with an endpoint-free periodic grid. The exact family is a constant
mean plus one Fourier mode. `cx`, `cy`, and `nu` are fixed; trajectory phase, amplitude,
mean, and optional time-step jitter vary. Each record stores `[cx,cy,nu]`, coordinates,
positive cell-area weights, `T+1` fields, and exactly `T` positive intervals.

`AdvectionDiffusion2DProblemAdapter` is the problem/trainer boundary. It builds the
`ProblemSpec`, dataset, physics constraints, reference metrics, and description.

## Replacing the physical problem without changing the trainer

1. Implement a `ProblemAdapter` that builds its dataset and `ProblemSpec`.
2. Return named `mass` and `operator` constraints (or compatible problem roles) from the
   adapter; constraints must consume raw-unit states.
3. Add the problem-specific config and strict validation schema.
4. Register one factory with `register_problem_adapter(problem_name, factory)`.
5. Reuse the same `train_v0_5` and `evaluate_v0_5` entry points. Neither contains a
   concrete advection-diffusion constructor.

Dataset records must still satisfy the canonical transition contract: `T+1` states,
exactly `T` positive `dt` values, aligned action length when actions exist, trajectory-level
splits, and train-only normalization.
