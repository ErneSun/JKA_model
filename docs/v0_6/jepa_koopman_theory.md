# JEPA–Koopman mathematical review

A Koopman-invariant coordinate family satisfies

\[
E(U_{t+\Delta t})=\exp(A\Delta t)E(U_t).
\]

V0.6 asks a continuously propagated online coordinate to match a slowly moving future
coordinate. Existing reconstruction, variance, forecast, stability,
generator-consistency and physics terms restrict trivial or dynamically meaningless
solutions.

The AAAI 2026 Koopman-invariants/JEPA paper proves a connection for an idealized
time-series objective and near-identity linear predictor. V0.6 is not a reproduction:
it uses a learned continuous generator, variable `dt`, decoded PDE constraints, and the
complete V0.5 trajectory objective. The paper is motivation, not proof that V0.6 learns
Koopman eigenfunctions. `||exp(A dt)-I||_F/||I||_F` at small/median/large observed `dt`
makes the relevant regime measurable rather than assumed.
