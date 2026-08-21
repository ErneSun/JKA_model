# V0.9 scientific report index

## Preserved baseline evidence

Session: `v09-full-20260821T033247Z`

- Workflow: PASS
- Evidence tier: EXPLORATORY_CONDITIONAL
- Low-rank operator adaptation: NOT_SUPPORTED
- Long-rollout stability: FAIL
- Physics: FAIL
- Selected rank: 8 (old maximum candidate)
V1.0 readiness: NOT_READY

The authoritative immutable report is
`gpu_validation/v0_9/results/v09-full-20260821T033247Z/report.md`.

## Stabilization revision

Session: `v09-stabilized-20260821T114835Z`

- Workflow: PASS
- Long-rollout stability: PASS
- Physics: FAIL
- Low-rank operator adaptation: NOT_SUPPORTED
- Selected rank: 12
- V1.0 readiness: NOT_READY

The stabilization objective corrected the earlier H32/H80 failure, but did not establish decoded
observable non-inferiority or the full adaptive mechanism.

## Phase-1/phase-2 revision

Software status: IMPLEMENTED; 22 targeted local tests PASS.

Phase 1 can reassess either completed raw session without training and will report operator,
dynamic and observable evidence separately. Phase 2 adds a generic multi-horizon observable
objective and therefore requires a new formal training ID.

Formal GPU science: PENDING A NEW VALIDATION ID.
Scientific support and V1.0 readiness remain unclaimed until that result is returned.
