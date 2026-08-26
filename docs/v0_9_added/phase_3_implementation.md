# V0.9 Added Route — Phase 3 Implementation

## Status

- Phase 2 is frozen; no Phase-2 retraining or further hyperparameter repair is allowed.
- Phase 3.0 evidence freeze: implemented.
- Phase 3.1 raw-field reconstruction/round-trip/tangent audit: implemented locally.
- Phase 3.2 physical decoder interface: stream-function candidate implemented and unit-tested.
- Matched `frozen/joint/from_scratch` training: pending the Phase-3 entry audit result.
- Local Phase-3 targeted tests: 5 passed.
- Formal RTX-5080 audit: pending.

The current entry audit is deliberately not called scientific support. Its output status is
`AUDIT_COMPLETE_TRAINING_PENDING` and it selects the mathematically appropriate Phase-3 training
candidate.

## Why Phase 3 begins with an audit

Phase-2 Fix 2 completed all formal work but showed a stable negative mechanism result: physical
passing improved to 16/18 while predictive skill, observer readiness and independent history gain
remained unsupported. Phase 3 is therefore not another Phase-2 optimization attempt. It asks
whether the frozen V0.8 representation is physically and Markov-sufficient for the variable-
condition problem.

The existing V0.9 adaptive cache contains latents encoded by the frozen V0.8 encoder. If encoder or
representation parameters are changed, those latents and their nominal residuals become stale.
Consequently:

\[
\boxed{
\texttt{joint/from\_scratch} \Longrightarrow
\text{raw-field online re-encoding}
}
\]

Using a frozen latent cache for a trainable representation route is rejected in code. This prevents
an apparently convenient but scientifically invalid joint-training workflow.

## Phase 3.0 — frozen evidence handoff

The entry workflow requires both:

```text
runs/v0_9/<phase2-id>/
gpu_validation/v0_9/results/<phase2-id>/
```

It verifies Phase-2 workflow completion, freezes its selected rank and decision fields, and records
`phase2_retraining_performed=false`. The current default source is
`v09-added-p2-physical-20260824T105209Z`.

## Phase 3.1 — raw-field representation audit

For each of the three inherited backbone seeds, locked V0.9 test trajectories are selected from the
Phase-2 adaptive-cache split manifest. The audit loads the original fields and computes:

1. reconstruction relative L2 error;
2. latent round-trip error

   \[
   \frac{\|E(D(z))-z\|_{RMS}}{\|z\|_{RMS}+\epsilon};
   \]

3. data and reconstruction divergence;
4. data and reconstruction no-slip error;
5. a finite-difference tangent diagnostic along the inherited nominal generator

   \[
   z_\epsilon=z+\epsilon A_0z,
   \qquad
   \frac{g(D(z_\epsilon))-g(D(z))}{\epsilon}.
   \]

The tangent diagnostic tests whether a nominal latent direction immediately leaves the decoded
physical manifold. It is not a time integrator and cannot by itself establish long-rollout skill.

The audit uses V0.9's variable-condition split, not the V0.8 backbone split. Reusing the backbone
manifest would mix trajectory identities from different datasets.

## Phase 3.2 — physical decoder candidate

The primary implemented candidate is a two-dimensional stream-function decoder:

\[
u=\partial_y\psi,\qquad v=-\partial_x\psi.
\]

The same discrete derivative pair is used for both components, so the curl contribution is
divergence-free in the rectangular-grid interior. Cylinder solid cells are set to no-slip and the
benchmark inlet/far-field velocities are imposed explicitly. Pressure is decoded separately
because incompressibility does not determine its gauge.

The module is only a candidate. It must not be inserted post hoc into evaluation and called an
improvement. If selected, it is trained under the matched route study with reconstruction, JEPA,
round-trip and physical constraints visible to optimization.

## Route decision

The Phase-3 entry audit produces one of four next-candidate classifications:

- `PHYSICAL_MANIFOLD_DECODER`: reconstruction physics itself fails;
- `JOINT_MARKOV_REPRESENTATION`: reconstruction passes but round-trip, tangent or observer evidence
  shows that the frozen representation is not sufficient;
- `HISTORY_NOT_REQUIRED_CONTROL`: representation is adequate but the Phase-2 dynamic branch remains
  unsupported;
- `FROZEN_REPRESENTATION_ADEQUATE`: frozen representation and dynamic mechanism both pass.

The classification chooses the next experiment. It does not change acceptance thresholds.

## Matched route contract

The following route names are immutable:

- `frozen`: preserve the V0.8 encoder/decoder/JEPA/context/`A0` and current Phase-2 diagnostic as the
  primary reference;
- `joint`: initialize from the same handoff, train a declared representation allow-list and adaptive
  modules with online raw-field re-encoding;
- `from_scratch`: fully reinitialize representation, context and operator on the same variable-
  condition data and forfeit inherited V0.8 validation claims.

All routes must match split fingerprint, trajectory IDs, backbone/data seed, operator seed, epochs,
evaluation-gate hash and reported compute. Code rejects stale latent-cache use for `joint` and
`from_scratch`.

The initial joint backbone allow-list is restricted to `online_encoder.projection` and
`training_decoder.refine.2`; it does not silently unfreeze the complete inherited backbone.

## Files

- `jka_model/config/schema.py`: `V09Phase3Config`;
- `jka_model/manifold/physical.py`: stream-function decoder and manifold metrics;
- `jka_model/manifold/audit.py`: locked raw-field representation/tangent audit;
- `jka_model/manifold/routes.py`: route ownership and matched-contract enforcement;
- `gpu_validation/v0_9/scripts/gpu_validate_phase3.py`: non-silent Phase-3 entry workflow;
- `tests/test_v0_9_phase3.py`: new Phase-3 mathematical/interface tests.

## One-line RTX-5080 entry audit

```bash
.venv/bin/python gpu_validation/v0_9/scripts/gpu_validate_phase3.py --validation-id v09-added-p3-audit-$(date -u +%Y%m%dT%H%M%SZ) --phase2-id v09-added-p2-physical-20260824T105209Z --seeds 47 53 59
```

This command does not retrain Phase 2. It produces raw artifacts under
`runs/v0_9/<resolved-id>/` and a compact report under
`gpu_validation/v0_9/results/<resolved-id>/`.
