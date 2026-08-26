# V0.9 Added Phase 3 — returned audit review and continuation

## Evidence reviewed

- Audit ID: `v09-added-p3-audit-20260826T043840Z`.
- Workflow status: `PASS`.
- Three backbone seeds audited: `47`, `53`, `59`.
- Phase-2 retraining performed by the audit: `NO`.
- Raw audit artifacts: complete on the GPU server; compact seed metrics and report returned.

## Corrected mathematical interpretation

The audit outputs divergence RMS,

\[
r_{div}=\sqrt{\operatorname{mean}[(\nabla\cdot u)^2]},
\]

while `max_divergence_mse` stores the threshold for the squared quantity. Therefore the legal
absolute RMS gate is

\[
r_{div}\le \sqrt{\texttt{max\_divergence\_mse}},
\]

not `r_div <= max_divergence_mse`. This is now identical to the existing official cylinder
observable gate.

| seed | raw divergence RMS | reconstruction RMS | relative change | corrected status |
|---:|---:|---:|---:|:---:|
| 47 | 0.083323 | 0.049457 | -40.64% | PASS |
| 53 | 0.072488 | 0.033476 | -53.82% | PASS |
| 59 | 0.072496 | 0.033918 | -53.21% | PASS |

The physical reconstruction is therefore feasible. The unresolved problem is the latent
round-trip error (`0.3677`–`0.5310` versus the `0.25` gate), together with the unsupported Phase-2
observer. The corrected candidate is `JOINT_MARKOV_REPRESENTATION`; the stream-function decoder is
retained only as a modular fallback and is not activated in the primary experiment.

## Phase 3.3 implementation after review

The joint route now has a separate raw-field trainer and workflow:

1. every trainable history is encoded online from the original physical field;
2. only immutable JEPA teacher targets may be cached;
3. the inherited nominal generator `A0` is frozen and checked bitwise after every epoch;
4. the trainable representation is restricted to the declared encoder/decoder tail allow-list;
5. context and factorized adaptive operator are optimized jointly;
6. the operator begins from the same declared initialization seed as the frozen route—no trained
   Phase-2 operator checkpoint is loaded, avoiding an unmatched extra training budget;
7. reconstruction, latent round-trip and frozen-target JEPA consistency are optimized together;
8. divergence/no-slip/outer-boundary physics enter as inequality violations, so an already-passing
   physical metric is not driven lower at the expense of forecast skill;
9. normalized representation drift is both penalized and used as a checkpoint feasibility gate;
10. early stopping cannot occur before the long-horizon/physics curriculum is mature;
11. the formal workflow covers `3 backbone seeds x 3 operator seeds x 2 condition modes` and runs a
    locked-test evaluation for every trained model;
12. the scientific result remains Phase-3-pending until the matched from-scratch control and final
    route aggregation are completed.

## Canonical RTX-5080 command

```bash
.venv/bin/python gpu_validation/v0_9/scripts/gpu_validate_phase3_joint.py --validation-id v09-added-p3-joint-$(date -u +%Y%m%dT%H%M%SZ) --phase2-id v09-added-p2-physical-20260824T105209Z --audit-id v09-added-p3-audit-20260826T043840Z --seeds 47 53 59 --operator-seeds 701 809 907 --condition-modes known latent_inferred
```

This command does not rerun Phase 2. It visibly reports stage `START/PASS/FAIL`, stores full raw
artifacts under `runs/v0_9/<resolved-id>/`, and writes the compact report under
`gpu_validation/v0_9/results/<resolved-id>/`.
