# V0.9 phase-1/phase-2 revision

This revision separates evidence correction from model retraining.

## Phase 1 — problem-independent evidence gates

`MetricGateSpec` declares only a metric name, direction, optional absolute threshold,
baseline-relative non-inferiority margin and numerical resolution floor. `GateResult` returns
`PASS`, `FAIL` or `INCONCLUSIVE` with the effective limit and provenance. The Koopman core does
not know cylinder-specific metric names.

For a lower-is-better metric, the baseline requirement is

\[
x_{adapt}\le x_0(1+\rho)+\delta_{abs}+\delta_{res}.
\]

If a hard absolute limit also exists, both requirements must hold. Higher-is-better metrics use
the reversed inequality. Non-finite or insufficient evidence is `INCONCLUSIVE`, never silently
converted to `FAIL` or `PASS`.

Shedding frequency is measured on a finite FFT window. Its gate therefore includes the declared
resolution floor

\[
\Delta f_{res}=\frac{n_{bins}}{N\Delta t}.
\]

This permits differences below one resolvable bin; it does not relax velocity, vorticity, force,
divergence or boundary gates.

The compact result now preserves full epoch metrics, scalar validation summaries, every scientific
gate and every per-trajectory observable gate. It separately aggregates:

- operator-explained residual;
- dynamic adaptation over a static operator correction;
- decoded observable non-inferiority;
- the original joint scientific decision.

An existing completed raw session can be reassessed without training:

```bash
.venv/bin/python gpu_validation/v0_9/scripts/gpu_reassess_existing.py --validation-id <resolved-v09-id> --device cuda
```

The reassessment requires all 18 formal checkpoints and records `training_count=0`.

## Phase 2 — generic frozen-decoder observable adapter

The optimization core depends on `ObservableObjective`, not on cylinder equations. A problem
adapter owns three operations: differentiable training components, locked-test metrics and gate
specifications. The frozen decoder converts predicted latent states to raw physical units while
allowing gradients only with respect to the adaptive latent input.

For configured horizons `H_obs`,

\[
\mathcal L_{obs}=\frac{1}{\sum_h\alpha_h}
\sum_{h\in H_{obs}}\alpha_h\sum_k w_k
\ell_k\!\left(D(\hat z_{t+h}),U_{t+h}\right).
\]

A nominal-relative protection term is

\[
\mathcal L_{NI}=\frac{1}{|H_{obs}|K}\sum_{h,k}
\left[
\frac{\operatorname{ReLU}(\ell^{adapt}_{h,k}-(1+\rho)\ell^0_{h,k}-\delta)}
{|\ell^0_{h,k}|+\delta}
\right]^2.
\]

The V0.9 operator objective adds

\[
s_{obs}(e)\lambda_{obs}
(\mathcal L_{obs}+\lambda_{NI}\mathcal L_{NI}).
\]

The current cylinder adapter supplies velocity, vorticity, divergence, no-slip, lift and drag.
Frequency remains an evaluation metric because the finite-window peak selector is not used as a
differentiable training target. The positive floor `delta` prevents a nearly zero nominal loss
from creating a singular relative penalty. The formal curriculum applies observables at H4/H8/H16 with
weights 0.2/0.4/0.4. All inherited frozen-backbone, bounded-coordinate, trust-gate, H4–H32 latent
rollout and H80 locked-test contracts remain unchanged.

## Interpretation boundary

Phase 1 may change the interpretation of an existing run, but cannot claim that the new objective
worked. Phase 2 requires a new formal training ID. A pass means the adaptive operator improves the
declared latent/observable contract on this controlled problem and seed matrix; it does not by
itself establish unseen-problem generalization.
