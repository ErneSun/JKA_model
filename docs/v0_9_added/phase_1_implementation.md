# V0.9 Added Route — Phase 1 Implementation

## Status

- Implementation: complete
- Targeted local verification: pass
- Formal three-seed GPU evidence: pending
- Scientific claim: not yet assigned

Passing software tests establishes implementation consistency only. It does not establish that
observable quality or physical non-inferiority improves on the locked cylinder-wake problem.

## Frozen contract

Phase 1 keeps the V0.8 encoder, decoder, JEPA target/online representation, context encoder and
nominal generator `A0` frozen. Only the existing low-rank adaptive operator is optimized. No
additive residual closure or persistent residual state is introduced.

## Implemented mathematics

### Train-only robust observable scales

For observable `k`, a fixed scale is fitted from training trajectories only:

\[
s_k=\operatorname{MAD}_{\rm train}(O_k)
\quad\text{or}\quad
s_k=\operatorname{RMS}_{\rm train}(O_k-\bar O_k).
\]

The optimized component loss is

\[
\mathcal L_k=\mathbb E\left[\operatorname{Huber}
\left((\hat O_k-O_k)/(s_k+\epsilon)\right)\right].
\]

The scale method, values, sample counts, split fingerprint and epsilon are stored in every
Phase-1 checkpoint. Divergence and boundary scales use dimensionally matched relative floors to
avoid division by a near-zero numerical target.

### Causal force-window objective

Lift and drag are optimized over causal decoded windows:

\[
\mathcal L_F=\mathcal L_{\rm waveform}
+\lambda_{\rm corr}(1-\operatorname{corr}(\hat F,F))
+\lambda_{\rm spec}\mathcal L_{\rm spectrum}.
\]

The waveform uses component-specific train-only scales. Correlation is evaluated only when the
target window has identifiable variation. The spectrum compares normalized differentiable power
spectra. Window decoding is strided to bound GPU memory.

### Inequality augmented Lagrangian

The primary forecast, rollout, velocity, vorticity and force objectives are separated from three
constraints:

\[
g_{\rm div}\le0,\qquad g_{\rm BC}\le0,\qquad g_{\rm burden}\le0.
\]

Optimization uses

\[
\mathcal L_{\rm AL}=\mathcal L_{\rm primary}
+\sum_k\lambda_k[g_k]_+
+\frac{\rho_k}{2}[g_k]_+^2,
\qquad
\lambda_k\leftarrow[\lambda_k+\rho_k g_k]_+.
\]

Multipliers, penalties, violations and their exact-resume state are logged. Penalties grow only
when positive violations fail to improve. Checkpoint selection uses a fixed, multiplier-invariant
score so changing Lagrange multipliers cannot make epochs incomparable.

### Stochastic H4–H80 observable curriculum

Training samples one horizon from

\[
\mathcal H_{obs}=\{4,8,16,32,80\}
\]

and applies the importance-corrected weight `alpha_h / p(h)`. Locked validation evaluates all
five horizons. The rollout dataset therefore owns at least 80 future steps even though the latent
rollout curriculum remains H4–H32.

### Gradient geometry

At the declared interval, the trainer measures

\[
G_{ij}=\frac{\langle\nabla_\omega\mathcal L_i,
\nabla_\omega\mathcal L_j\rangle}
{\|\nabla_\omega\mathcal L_i\|\,\|\nabla_\omega\mathcal L_j\|+\epsilon}
\]

for forecast, rollout and available observable objectives. This is diagnostic only: Phase 1 does
not silently enable PCGrad or CAGrad before measured conflict justifies it.

### Four-level error attribution

Locked evaluation reports each observable at:

1. numerical data `U`;
2. reconstruction `D(E(U))`;
3. nominal prediction `D(exp(A0 dt)E(U))`;
4. adaptive prediction `D(exp(At dt)E(U))`.

The report separates the data floor, representation increment, nominal-dynamics increment and
adaptive-dynamics increment. If reconstruction already fails divergence or boundary gates, the
result is `REPRESENTATION_BLOCKED`, and V1 readiness is prohibited.

## Module boundaries

- `jka_model/observables/scaling.py`: robust scales and standardized Huber loss;
- `jka_model/optimization/augmented_lagrangian.py`: generic inequality optimizer state;
- `jka_model/optimization/gradient_geometry.py`: gradient cosine diagnostics;
- `jka_model/adaptive/error_attribution.py`: four-level attribution;
- `jka_model/problems/cylinder_observables.py`: cylinder-owned observables and force physics;
- `train/train_v0_9.py`: orchestration and checkpoint/report state;
- `eval/evaluate_v0_9.py`: locked attribution and representation-floor decision;
- `jka_model/adaptive/reporting.py`: nested compact Phase-1 evidence.

Generic Koopman and optimization modules contain no cylinder-specific equations.

## Required artifacts

Every formal run produces:

- `logs/epoch_metrics.csv` with constraints, multipliers and penalties;
- `logs/gradient_geometry.jsonl`;
- `evaluation/gradient_geometry_summary.json`;
- checkpointed train-only observable scales and augmented-Lagrangian state;
- `evaluation/error_attribution.csv` and `error_attribution.json`;
- the existing rollout, physical, gate and scientific-decision artifacts.

Nested aggregation copies error attribution and gradient geometry into the compact result and
marks the compact audit incomplete if either is absent.

## Acceptance interpretation

Formal validation may produce any of these valid outcomes:

- observable improvement with physical non-inferiority;
- `REPRESENTATION_BLOCKED`, directing the route to Phase 3 after the Phase-2 identifiability audit;
- operator optimization failure despite an adequate representation;
- inconclusive evidence when nested seeds disagree.

No outcome is converted into support merely because the workflow completed.

## Canonical GPU validation

```bash
python gpu_validation/v0_9/scripts/gpu_validate_all.py --validation-id v09-added-phase1-$(date -u +%Y%m%dT%H%M%SZ) --v0-8-handoff-policy supported --seeds 47 53 59
```

Occupied validation IDs are resolved by the existing `-r1`, `-r2`, ... policy.
