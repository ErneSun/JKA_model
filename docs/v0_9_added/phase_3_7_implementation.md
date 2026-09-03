# V0.9 Added Phase 3.7 — Physics-Aligned Representation and Observer Admission

## Status

- Implementation: complete locally.
- Targeted CPU tests: passed locally (69 tests; warnings are upstream PyTorch deprecations).
- Formal RTX-5080 validation: pending.
- Scientific status before the new GPU run: **not evaluated**.
- Frozen matched reference: `v09-added-p3-routes-20260829T025754Z`.
- Phase-2 and from-scratch training are not repeated.

Phase 3.7 addresses the three remaining Phase-3.6 bottlenecks without changing the frozen
nominal generator, the matched data/seeds/budgets, or the predeclared 2% decoded-field gate:

1. latent error is measured in the local physical metric induced by the frozen decoder;
2. coordinate gauge drift is separated from incompatibility with `A0`;
3. the latent condition observer must pass an independent history-control audit before it can
   control the condition-dependent operator branch.

## 1. Frozen scientific contract

The following remain unchanged:

\[
z_t=E_\theta(U_t),\qquad
z_{t+1}=\exp(A_t\Delta t)z_t,
\]

\[
A_t=A_0+\Delta A_s(\hat q_t)+\Delta A_d(h_t^\perp,\hat q_t).
\]

- `A0` is frozen for the joint route.
- Every trainable representation route re-encodes raw fields online.
- Physics and decoded supervision use raw physical units.
- The locked test remains evaluation-only.
- The frozen and completed from-scratch routes remain immutable controls.
- Decoded field must improve by at least 2% at H8/H16/H32/H80; velocity and vorticity must be
  non-inferior.

## 2. Frozen-decoder pullback metric

Let

\[
\delta z_h=\hat z_h-z_h^\star,
\]

where `z*` is the immutable JEPA target.  The original latent objective uses Euclidean distance,
which treats every latent direction equally even though the decoder can be almost insensitive to
some directions.  Phase 3.7 adds

\[
G_{\rm phys}(z_h^\star)
=J_{D_0}(z_h^\star)^T W_{\rm phys}J_{D_0}(z_h^\star),
\]

\[
\mathcal L_{\rm pullback}
=\sum_h\omega_h\,
\delta z_h^T G_{\rm phys}(z_h^\star)\delta z_h.
\]

`D0` is a frozen copy of the inherited decoder.  Therefore the model cannot reduce this loss by
shrinking the metric itself.  The implementation evaluates the product

\[
J_{D_0}(z_h^\star)\delta z_h
\]

with a Jacobian-vector product; it never materializes a full decoder Jacobian.  Its field,
velocity and vorticity energies are normalized by the corresponding target physical energies and
combined with the existing `2.0 / 1.0 / 0.2` physical weights.

Because normalization is affine, a tangent is mapped to raw units only by channel scale:

\[
\delta U_{\rm raw}=\sigma_U\odot\delta U_{\rm model}.
\]

The channel mean must not be added to a perturbation.  This is implemented by
`ChannelStandardizer.inverse_transform_tangent()`.

## 3. Dynamically compatible latent gauge

CKA and ordinary Procrustes detect geometric similarity but do not determine whether a coordinate
rotation remains compatible with the frozen generator.  On the validation representation bank,
Phase 3.7 solves

\[
T^\star=\arg\min_{T^TT=I}
\|\widetilde Z_{\rm new}T-\widetilde Z_{\rm ref}\|_F,
\]

and records

\[
d_{\rm gauge}=\|\widetilde Z_{\rm new}T^\star-\widetilde Z_{\rm ref}\|_F,
\]

\[
d_{A}=\frac{\|A_0T^\star-T^\star A_0\|_F}
{\|A_0\|_F+\epsilon}.
\]

The first value distinguishes semantic drift from orthogonal gauge freedom.  The commutator tests
whether the aligned coordinates preserve the same nominal dynamics.  Both are included in mature
checkpoint feasibility and the final representation gate.  Pre-GPU limits are fixed at 0.10 for
both values.  The original representation-drift limit of 0.10 is retained; it is not relaxed after
the Phase-3.6 result.

## 4. Independent observer admission

The observer is trained before joint representation/operator optimization while `E`, `D`, `A0`,
the context encoder and both operator branches are frozen.  Three observers use the same
architecture, initialization, training split, epoch budget and optimizer.  Each control is selected
by its own validation NRMSE/R2 checkpoint, so the real-history route receives no checkpoint-selection
advantage:

- `history`: the real causal latent history;
- `instantaneous`: the current latent is repeated across the history window;
- `shuffled_history`: pre-current history is deterministically reversed while the current latent
  is preserved.

A fourth control predicts the training-condition mean.  The target is

\[
q_t=(Re_t,U_{\infty,t},\dot{Re}_t),
\]

normalized only with Phase-2 training-split statistics.  Admission requires

\[
\operatorname{NRMSE}_{H}\le0.50,
\qquad
\min R_H^2\ge0.20,
\]

\[
\operatorname{NRMSE}_{I}-\operatorname{NRMSE}_{H}\ge0.02,
\qquad
\operatorname{NRMSE}_{S}-\operatorname{NRMSE}_{H}\ge0.02,
\]

and the history observer must beat the mean predictor.  The selected history observer is then
frozen; operator gradients cannot change its physical meaning.  Because the online encoder is
still refined, the same frozen controls are reevaluated on every validation epoch and on the
locked test.  A checkpoint is feasible only if the relative history advantage remains present
after representation refinement.  The operator route itself is fixed by the initial validation
admission and is never switched using locked-test outcomes.

If admission fails, `latent_inferred` does **not** use `q_hat`.  The static condition branch is
zeroed and the run falls back to the R3 history-only dynamic branch:

\[
A_t=A_0+\Delta A_d(h_t^\perp,0).
\]

The run still completes to measure whether history alone is useful, but its observer gate is false
and it cannot establish latent-condition support.

## 5. Checkpoint and acceptance order

Mature checkpoints are ranked lexicographically by:

1. raw physical-manifold feasibility;
2. original representation drift;
3. round-trip consistency;
4. dynamical-gauge NRMSE and generator commutator;
5. observer admission plus observer NRMSE/R2 for `latent_inferred`;
6. decoded field/velocity/vorticity error;
7. latent predictive shortfall and total objective.

Locked-test scientific support still additionally requires matched decoded-field gain, velocity and
vorticity non-inferiority, both condition modes, operator-seed robustness and backbone-seed
robustness.  No new diagnostic can pass the version by itself.

## 6. Artifacts

Every run writes:

- `evaluation/observer_admission.json` with initial, post-refinement validation and locked-test
  controls and admission decisions;
- validation and locked-test pullback losses;
- `dynamical_gauge_nrmse` and `generator_commutator`;
- `observer_admitted` and the selected operator condition route;
- checkpoint provenance, frozen reference decoder state and complete resolved config.

The compact aggregate additionally reports dynamical-gauge pass fraction, observer-admission
fraction and history-only fallback fraction.

## 7. Canonical one-line RTX-5080 validation

```bash
.venv/bin/python gpu_validation/v0_9/scripts/gpu_validate_phase3_joint.py --phase37 --validation-id v09-added-p37-aligned-$(date -u +%Y%m%dT%H%M%SZ) --phase2-id v09-added-p2-physical-20260824T105209Z --audit-id v09-added-p3-audit-20260826T043840Z --frozen-reference-id v09-added-p3-routes-20260829T025754Z --seeds 47 53 59 --operator-seeds 701 809 907 --condition-modes known latent_inferred
```

The workflow is non-silent at stage level, does not print every training epoch, preserves failures,
and automatically allocates `-r1`, `-r2`, ... when a requested ID is already occupied.
