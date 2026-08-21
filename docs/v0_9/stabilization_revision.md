# V0.9 long-horizon stabilization revision

## Why this revision exists

The first complete V0.9 GPU study (`v09-full-20260821T033247Z`) verified the software
workflow but did not support the scientific mechanism.  The adaptive operator improved short
rollouts but degraded H32/H80 and decoded physics, while the validation sweep selected its
largest available rank.  That is evidence of an under-constrained one-step objective, not
evidence that time-varying Koopman adaptation is impossible.

This revision preserves the frozen V0.8 backbone, decoder, context encoder and nominal generator
`A0`.  It changes only the optimization and restricted adapter interface.  Additive residuals,
persistent `z_R`, latent clipping and test-set model selection remain forbidden.

## Restricted adaptive generator

Let `q_t=G_omega(c_t,u_t)` be the raw coordinate head output.  The effective coordinates are

\[
g_t=\sigma(h_omega(c_t,u_t)),\qquad
\eta_t=g_t\,\eta_{max}\tanh(q_t),
\]

and

\[
A_t=A_0+U\operatorname{diag}(\eta_t)V^\top,
\qquad \hat z_{t+1}=\exp(A_t\Delta t_t)\hat z_t.
\]

Both heads have zero-weight initialization; `q_t=0` therefore gives `A_t=A0` exactly.  The gate
starts near 0.2 and can learn when adaptation is trustworthy.  `eta_max` is an auditable bound on
each effective coordinate; it is not state clipping.

## Teacher-free curriculum objective

After the initial measured history, every future context is computed from predicted latent
states.  Gradients pass through the frozen context encoder with respect to its inputs, while its
parameters remain frozen.  For active curriculum horizons `H_e`, the operator-only objective is

\[
\begin{aligned}
\mathcal L_e={}&\mathcal L_{1}
+\lambda_{roll}\sum_{h\in H_e}w_h
 \|S^{-1}(\hat z_{t+h}-z_{t+h})\|_2^2\\
&+\lambda_A\left[\mathbb E B_A^2+
 \mathbb E\operatorname{ReLU}(B_A-B_{max})^2+\mathcal L_{UV}\right]\\
&+\lambda_{sym}\mathcal L_{sym}
+\lambda_{prop}\mathbb E\operatorname{ReLU}
 \left(\|e^{A_t\Delta t}\|_2-\|e^{A_0\Delta t}\|_2-m\right)^2\\
&+\lambda_{smooth}\mathcal L_{smooth}
+s_{phys}(e)\lambda_{phys}\mathcal L_{phys},
\end{aligned}
\]

where `S` is the train-only residual scale and
`B_A=||A_t-A0||_F/(||A0||_F+epsilon)`.  The propagator term is relative to the inherited nominal
dynamics, so it does not incorrectly force an oscillatory wake to be strictly contractive.

The default curriculum activates H4, H8, H16 and H32 at training fractions 0, 0.2, 0.45 and 0.7.
The physical weight begins at fraction 0.35, ramps linearly for 0.25 of training, and then remains
fully active.  Validation always evaluates the full curriculum.  A stride of four reduces
redundant overlapping windows without changing the trajectory split or causal ordering.  Early
stopping receives a fresh patience budget only after the final rollout/physics curriculum stage is
active.

## Frozen-decoder physical anchor

Only a small subset of each batch is decoded at H8.  In raw units,

\[
\mathcal L_{phys}=w_u\mathcal L_u+w_\omega\mathcal L_\omega
+w_{div}\mathcal L_{div}+w_{wall}\mathcal L_{wall}.
\]

The terms are normalized relative velocity error, relative vorticity error, divergence energy
relative to target gradient energy, and cylinder no-slip energy relative to target fluid velocity
energy.  Decoder weights never receive gradients or optimizer state; the loss changes only the
adaptive operator through the predicted latent state.  Lift, drag and frequency remain locked-test
acceptance metrics rather than training targets.

## Rank selection and reproducibility

The validation sweep tests ranks 2/4/8/12 for both condition modes.  A rank is eligible only when
its mean longest-curriculum gain is non-negative and its maximum burden is within the configured
limit.  Among eligible ranks, the smallest rank within 2% of the best composite validation
objective is selected.  If no rank is eligible, the workflow records that constraint failure and
uses the same parsimonious fallback over all ranks; it does not silently declare the sweep valid.

Each module has an independent contract:

- `adaptive/dataset.py`: causal rollout windows and stride;
- `adaptive/models.py`: bounded coordinates and trust gate;
- `adaptive/objectives.py`: curriculum, differentiable rollout and stability terms;
- `adaptive/physics.py`: frozen decoder and raw-unit physical losses;
- `train/train_v0_9.py`: stage composition, exact resume and compact logging;
- `gpu_validate_all.py`: constrained rank selection and nested formal validation.

The existing frozen-handoff route remains the primary V0.9 experiment.  Matched joint-fine-tuning
and from-scratch routes remain separately deferred to “V0.9 added”; they must not be mixed into
this stabilization result.
