# Review addendum — `v09-added-p3-joint-20260826T053347Z`

## Verdict

- Workflow completion: **PASS**, with 18/18 formal train and locked-test runs.
- Numerical/physical execution: finite and physically feasible under the reported manifold metric.
- Scientific joint-route support: **NOT SUPPORTED** under the complete declared contract.
- The original compact files are preserved as returned evidence. This addendum supersedes only
  their scientific interpretation.

## Corrected gate accounting

The reported `joint_feasibility_pass_fraction=0.444` counted only physical-manifold compliance and
representation drift. Reassessment using all declared gates gives:

| Gate | Passing runs |
|---|---:|
| Physical manifold | 18/18 |
| Representation drift | 8/18 |
| Round trip | 12/18 |
| Physical + drift + round trip | 4/18 |
| Material gain at every declared horizon | 0/18 |
| Complete joint contract | 0/18 |

Mean relative gains over the frozen reference were positive but sub-material at H4-H32 and became
negative at H80. The run therefore shows limited short-horizon representation adjustment, not a
validated long-horizon Koopman improvement.

## Training-contract defect

All 18 runs stopped at epoch 49. The old implementation accumulated early-stopping staleness from
epoch 1 while forbidding stopping until the curriculum became mature. At epoch 49, stopping became
legal and the already-exhausted patience terminated training immediately. This left only about five
epochs of exposure to H80 and selected checkpoints without requiring round-trip, observer, or
predictive feasibility.

## Corrective action

The corrected Phase-3.3 workflow:

1. starts checkpoint selection and patience only after the full curriculum is active;
2. ranks mature checkpoints by physical, drift, round-trip, observer, and all-horizon predictive
   feasibility before total loss;
3. reports each gate separately and reserves `strict_joint` for their intersection;
4. records selected best epochs and decoded field, velocity, vorticity, divergence, and boundary
   metrics on validation and locked test.

The matched `from_scratch` control is deferred until the corrected joint result is reviewed. This
avoids spending the matched-control budget against a defective joint-training contract.
