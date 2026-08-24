# V0.9 Added Route — Phase 2 Implementation

## Status and scientific scope

- Implementation: pre-Phase-3 physics-feasible/identifiable revision complete locally
- Targeted verification: 49 relevant tests pass
- Formal three-seed GPU evidence: pending
- Scientific claim: not assigned before the formal report

### Stability revision after the first formal attempt

Session `v09-added-p2-identifiable-20260823T100133Z` completed all eight rank-sweep runs and five
formal train/evaluation pairs, then failed for backbone seed 47, latent mode, operator seed 907.
The first reported non-finite tensor entered the frozen context encoder during validation rollout;
the cache and earlier runs were valid. Seed 47's trajectory split places abrupt up/down schedules
in validation while training contains smooth rates, so the normalized condition-rate observer must
extrapolate. Joint operator gradients could then distort the observer and amplify a closed-loop
state before the next context call.

That first stability revision applied three problem-independent corrections:

1. normalized observer output is smoothly bounded and trained with Huber loss;
2. latent `q_hat` is stop-gradient when consumed by the operator, so only its physical supervision
   trains `Q`;
3. each low-rank branch is scaled by a logarithmic-norm trust region, preserving its dyadic rank
   while bounding the symmetric generator increment.

The last bound uses

\[
\widetilde{\Delta A}_b=s_b\Delta A_b,\qquad
s_b=\min\left(1,\frac{\beta/2}
{\sqrt{\|\operatorname{sym}(\Delta A_b)\|_F^2+\epsilon}}\right),
\quad b\in\{s,d\}.
\]

Because the Frobenius norm upper-bounds the spectral norm,
`lambda_max(sym(Delta A_static + Delta A_dynamic)) <= beta` by the triangle inequality, without
adding a new state residual or altering `A0`. Training also rejects non-finite losses/gradients at
their originating epoch and rollout step rather than surfacing them later as a generic context
error.

Phase 2 addresses the remaining ambiguity in Phase 1: a time-varying operator can improve because
it recognizes the current operating condition, even when older history carries no additional
predictive information. The implementation therefore separates those two mechanisms without
changing the frozen V0.8 representation, the nominal generator `A0`, the cylinder equations, or
the physical constraints.

### Continuous three-stage revision after `v09-added-p2-stable-20260823T110904Z`

The stability-corrected run completed 18/18 formal train/evaluation pairs and all 216 rollouts
remained finite, but scientific support was not obtained. Relative to Phase-1 r1, average operator
burden fell from roughly 10--11% to 0.55--0.67%; known-condition one-step gain fell to 0.64% and
latent-inferred gain to 0.02%. The condition observer passed 0/18 nested runs. This is evidence of
over-constrained adaptation, not numerical divergence.

Phase 2 now uses one continuous run with auditable mechanism states:

1. `static_oracle`: train only `Delta A_s(q)` from true train/validation condition labels;
2. `dynamic_residual_oracle`: detach the identified static branch and fit `Delta A_d(h,q)` to its
   remaining forecast error;
3. `observer_calibration`: continue training `Q(c,o)` alone; latent joint refinement is activated only after
   validation NRMSE and R2 pass their predeclared gates.

Oracle conditions are privileged training information only. Locked latent evaluation remains
teacher-free and rejects condition-label input. A failed observer gate therefore produces valid
negative latent evidence rather than silently falling back to known conditions.

The old branch-wise stability projection is replaced by one projection of the physical total
generator increment:

\[
\Delta A=\Delta A_s+\Delta A_d,\qquad
\widetilde{\Delta A}=s\Delta A,
\quad
s=\min\left(1,\frac{\rho}
{\sqrt{\|\operatorname{sym}(\Delta A)\|_F^2+\epsilon}}\right).
\]

The budget follows `0.05 -> 0.10 -> 0.15`. This bounds the logarithmic-norm contribution of the
actual combined operator while avoiding the unnecessary requirement that each branch consume
exactly half the budget. Dynamic-stage gradients cannot rewrite the static dyads.

Rank selection now uses the known-condition oracle sweep only and is lexicographic: burden
feasibility, material H80 gain, maximum H80 gain up to a narrow equivalence tolerance, validation
objective, then smallest rank. Latent rank runs remain observer diagnostics but cannot choose
operator capacity. If no rank reaches
the 2% H80 gate, the workflow still selects the best burden-feasible diagnostic rank but records
`constraints_satisfied=false`. This prevents a broad loss-relative tolerance from selecting an
underfit rank solely for parsimony.

H80 is activated during the dynamic-operator stage, receives 35% of stochastic physical-horizon
sampling and 45% of physical-horizon weight. PCGrad begins with the physical curriculum. Reports
separate `NUMERICAL STABILITY` (finite and burden-bounded) from `LONG-ROLLOUT SKILL` (material
predictive gain).

### Pre-Phase-3 correction after `v09-added-p2-continuous-20260823T133237Z`

The continuous run established a real known-condition long-horizon signal: mean H80 gain rose to
2.09% and mean operator-explained fraction to 5.54%. It did not establish all-horizon or dynamic-
history support. H8/H16/H32 mean gains remained below 1%, the observer remained below readiness,
and only 7/18 runs passed physical non-inferiority. Rank 8 was selected although its validation
boundary constraint was positive, because the previous selector placed H80 gain before physics.

The corrected selector is

\[
\text{physical feasibility}\;\prec\;\text{burden feasibility}\;\prec\;
\text{H80 gain}\;\prec\;\text{validation objective}\;\prec\;\text{rank}.
\]

Checkpoint selection follows the same lexicographic rule. If no rank is physically feasible, the
workflow first minimizes the maximum positive boundary/divergence violation and only then uses
gain. A checkpoint trained before the physical ramp reaches one cannot outrank a mature checkpoint.
Every mechanism stage restores the preceding stage's selected state and resets optimizer moments;
it does not inherit an arbitrary final SGD iterate. Validation constraint and burden maxima are
aggregated by worst batch/horizon rather than averaged away. No physics threshold is relaxed.

The observer now uses the causal feature map

\[
o_t=\left[z_t,\;\frac1H\sum_{j=0}^{H-1}z_{t-j},\;
\frac{z_t-z_{t-H+1}}{\sum_{j=1}^{H-1}\Delta t_{t-j}}\right],
\qquad \hat q_t=Q(c_t,o_t).
\]

It is supervised during static-oracle and dynamic-residual-oracle training, but its prediction is
not consumed by those oracle operator branches. Observer gradients therefore cannot change the
oracle condition or contaminate mechanism identification, while all supervised epochs are used.

Dynamic history is residualized before operator coordinates are produced:

\[
\hat h(q_t)=M_\theta(q_t),\qquad h_t^\perp=h_t-\hat h(q_t),\qquad
\xi_t=H_\psi(h_t^\perp).
\]

`M_theta` is trained by a smooth conditional-context reconstruction loss, while the existing
kernel penalty still enforces `E[xi|q] approximately 0`. The dynamic head has no direct `q` input;
condition-only effects must remain in the static branch.

Finally, every rollout endpoint uses a dimensionless nominal-relative training error

\[
\mathcal L_h=
\frac{\|z_{t+h}^{adapt}-z_{t+h}^{true}\|^2}
{\operatorname{stopgrad}(\|z_{t+h}^{nom}-z_{t+h}^{true}\|^2)+\epsilon}.
\]

This aligns optimization with the declared relative-gain gates and prevents the largest absolute-
error horizon from dominating merely because of scale. The known oracle report excludes the
observer gate; the latent route and overall deployable claim still require it.

## Mathematical contract

The adaptive generator is

\[
A_t=A_0+\Delta A_s(\hat q_t)+\Delta A_d(h_t,\hat q_t),
\]

with dyadic low-rank bases

\[
\Delta A_s=\sum_{j=1}^{r_s}\phi_j(\hat q_t)u_j^s(v_j^s)^T,
\qquad
\Delta A_d=\sum_{k=1}^{r_d}\xi_{t,k}u_k^d(v_k^d)^T.
\]

The condition vector is

\[
q_t=(Re_t,U_{\infty,t},\dot{Re}_t),
\]

where the rate uses a causal backward difference. In the known-condition control, `q_t` is
supplied. In the latent route, `Q(c_t,o_t)` estimates it from frozen context plus causal latent
state/mean/trend features;
condition labels are used as held-out targets but are never passed into latent inference.

Identifiability is enforced by

\[
\mathbb E[\xi_t\mid q_t]\approx0,
\qquad
\langle u_j^s(v_j^s)^T,u_k^d(v_k^d)^T\rangle_F\approx0.
\]

The first condition uses a differentiable kernel conditional-mean penalty. The second uses the
exact dyadic Frobenius inner product. Each branch also retains within-basis orthogonality. No
variance floor is imposed on `xi`: the mathematically valid answer may be that memory is not
required.

All coordinate heads are zero-initialized, hence `A_t=A0` exactly at initialization. Static and
dynamic trust gates are separate: the static gate reads only `q`, while the dynamic gate reads only
the condition-residualized history context. Training first identifies the static branch and then
fits the dynamic residual with the static branch detached. The observer is supervised in parallel
but cannot control either oracle branch; its dedicated calibration stage follows. Latent joint
refinement is unavailable until the validation observer gate passes.

## Problem-independent controls

Each locked run now measures:

1. frozen nominal `A0`;
2. the legacy global mean operator correction;
3. `A0 + condition branch`;
4. `A0 + condition branch + dynamic innovation`;
5. the same model with only the dynamic history shuffled while condition and present state remain
   fixed;
6. condition-observer RMSE and R2 for `Re`, `U_infinity`, and `dRe/dt`;
7. matched-present pairs with close condition/current latent, separated older histories, and
   separated futures.

The isolated shuffle is important: shuffling history is not allowed to change the condition branch.
Otherwise a failed shuffle control could merely show that the operating condition was corrupted.

## Excitation schedules

The physical PDE and geometry remain unchanged. Twelve controlled schedules provide the
identifiability excitation:

- upward and downward transitions;
- slow, medium, fast, and abrupt changes;
- cyclic up/down changes;
- short and long dwell variants.

Splits are stratified by `up`, `down`, and `cyclic` families. Normalization is fitted on training
trajectories only. This is a richer experiment design, not a problem-specific correction to the
model.

## Decision semantics

Phase 2 distinguishes three scientifically different outcomes:

- `DYNAMIC_ADAPTIVE_KOOPMAN_SUPPORTED`: condition observation, paired history gain, real-over-
  shuffled history, physics, and long rollout jointly pass;
- `PARAMETERIZED_KOOPMAN_SUPPORTED; HISTORY_ADAPTATION_NOT_REQUIRED`: the condition branch is
  useful and observable, but history innovation supplies no material held-out gain;
- `LATENT_CONDITION_NOT_IDENTIFIABLE`: the frozen history representation cannot reliably identify
  the operating condition, so failure is not assigned to Koopman rank.

Insufficient matched pairs are `INCONCLUSIVE`, never silently converted to failure or support.

## Implementation boundaries and artifacts

- `adaptive/models.py`: factorized operator and condition observer;
- `adaptive/identifiability.py`: causal targets, train-only scaling, conditional centering,
  observer metrics, and matched-pair selection;
- `adaptive/objectives.py`: Phase-2 supervised and identifiability losses;
- `data/cylinder_wake_2d.py`: extended controlled schedules only;
- `train/train_v0_9.py`: continuous static/dynamic/observer state orchestration and observer gate;
- `eval/evaluate_v0_9.py`: isolated controls and paired locked-test audit;
- `adaptive/reporting.py`: nested seed aggregation and valid-negative classification.

Formal compact results include `condition_observer_metrics.csv`, `matched_history_pairs.csv`, the
per-run JSON evidence, scientific gate rows, training logs, physics metrics, and the final Markdown
report. The GPU workflow retains visible stage `START` and `PASS`/`FAIL` output while formal model
training prints only its start and compact final result.

## Acceptance boundary

Local tests establish equation/interface consistency, causal condition construction, exact `A0`
initialization, no label leakage, schedule coverage, differentiability, and artifact contracts.
They do not establish a Phase-2 scientific improvement. That conclusion is reserved for the new
three-seed formal GPU run.
