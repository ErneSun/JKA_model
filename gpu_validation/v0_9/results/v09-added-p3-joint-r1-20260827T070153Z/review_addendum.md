# Review addendum — `v09-added-p3-joint-r1-20260827T070153Z`

## Verdict

- Workflow: **PASS**, 18/18 train and locked-test runs complete.
- Early-stopping correction: **PASS**, completed epochs span 65--80 rather than all stopping at 49.
- Route-internal predictive gate: **PASS**, 18/18 runs exceed 2% at H8, H16, H32 and H80.
- Scoped-joint representation gate: **FAIL**, 0/18 remain below the frozen-coordinate drift limit.
- Complete scoped-joint contract: **NOT SUPPORTED**.

## Mechanism evidence

Relative latent gains changed from approximately 0.14%, 0.12%, 0.01%, -0.16% in the initial
joint run to 21.67%, 20.64%, 15.52%, 8.80% at H8, H16, H32, H80. Round trip passes 18/18 and the
reported physical-manifold penalty is zero in 18/18. The improvement is therefore real within the
new latent route, not an isolated operator seed.

However, normalized coordinate drift is 0.254--0.527 against the predeclared joint limit 0.10.
The run supports the representation-bottleneck hypothesis but does not support a scoped
continuation of the inherited Koopman coordinates.

## Reporting correction

The aggregate `observer_pass_fraction=0.667` includes all nine known-condition runs, for which the
observer gate is intentionally not required. The deployable latent-inferred observer passes only
3/9 runs, all on backbone seed 47. Future reports therefore expose a separate
`latent_observer_pass_fraction`.

## Remaining evidence gap

Latent gain is measured against the nominal operator inside the changed representation. The
decoded H80 errors (field 0.281, velocity 0.134, vorticity 0.412 on average) have no exactly matched
frozen baseline in this result. Consequently this session cannot establish that physical-field
forecasting improved over the inherited route.

The next workflow evaluates every inherited frozen checkpoint with the same decoded metrics,
trains a true from-scratch EMA-JEPA/context/nominal-plus-adaptive operator control, and performs
pairwise frozen/joint/from-scratch aggregation without relaxing the joint drift threshold.
