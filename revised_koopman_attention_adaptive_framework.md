# Revised Mathematical Framework and Version Roadmap
## Koopman + Residual + Attention Context + Adaptive Koopman + Optional Residual Closure

## 1. Core architectural revision

\[
\boxed{
\text{Residual exposes Koopman mismatch; Attention learns dynamic context from that mismatch; the context then determines whether and how Koopman should adapt.}
}
\]

The final residual closure is postponed until after adaptive Koopman has been tested.

## 2. Physical state and Koopman latent

Let

\[
U_t \in \mathcal M_{\rm phys}.
\]

The Koopman encoder gives

\[
\boxed{
z_t^K = E_\theta(U_t)
}
\]

with

\[
z_t^K \in \mathbb R^{d_K}.
\]

Interpretation:

\[
\boxed{
z_t^K = \text{current dynamical state coordinate}
}
\]

It answers where the system currently is in the learned dynamical state space.

## 3. JEPA role

JEPA remains a representation-learning principle used to shape \(E_\theta\) so that \(z_K\) is stable and predictive.

\[
\boxed{
\text{JEPA = predictive representation learning}
}
\]

After V0.6, later residual and adaptive-operator studies should first keep \(E_\theta\) fixed so that all analyses use the same latent coordinate system.

## 4. Nominal Koopman dynamics

Define the nominal Koopman generator

\[
\boxed{
A_0
}
\]

and propagator

\[
K_0(\Delta t)=e^{A_0\Delta t}.
\]

Then

\[
\boxed{
z_{t+1}^{0}
=
e^{A_0\Delta t_t}z_t^K.
}
\]

The preferred interpretation is **nominal/persistent dynamics**, not “steady dynamics”. \(A_0\) may describe oscillation, advection, damping, rotation, or other non-steady evolution; what is fixed is the dynamical law.

## 5. Nominal Koopman residual

The true future latent coordinate is

\[
z_{t+1}^{\rm true}=E_\theta(U_{t+1}).
\]

Define

\[
\boxed{
r_{t+1}^{0}
=
z_{t+1}^{\rm true}
-
e^{A_0\Delta t_t}z_t^K.
}
\]

The superscript \(0\) distinguishes the nominal residual from later \(r^{op}\) and \(r^{rem}\).

Interpretation:

\[
\boxed{
r^0 = \text{dynamics not explained by the nominal Koopman model}
}
\]

Possible sources include regime change, parameter variation, inadequate Koopman coordinates, unresolved nonlinear dynamics, memory effects, external forcing, stochasticity, noise, or numerical/model error.

Therefore V0.7 should not assume in advance what type of residual \(r^0\) is.

## 6. Koopman adequacy / mismatch measure

Introduce

\[
\boxed{
m_t = \text{Koopman inadequacy measure}
}
\]

with the simplest form

\[
m_t=\|r_{t+1}^{0}\|.
\]

A normalized variant can be

\[
\tilde m_t
=
\frac{
\|r_{t+1}^{0}\|
}{
\|z_{t+1}^{\rm true}-z_t^K\|+\epsilon
}.
\]

A robust denominator should be used when the true latent increment is very small.

Small \(m_t\) suggests the nominal operator is locally adequate; large or rapidly increasing \(m_t\) may indicate regime transition, parameter change, transient behavior, unmodeled forcing, or degradation of the current Koopman approximation.
## 7. Residual Structure Assessment: the new routing layer

The revised architecture should not assume that every physical problem requires the same residual model.

The nominal residual remains

\[
\boxed{
r_{t+1}^{0}
=
z_{t+1}^{\rm true}
-
e^{A_0\Delta t_t}z_t^K.
}
\]

However, the structure of \(r^0\) can depend strongly on:

- the physical problem;
- the selected Koopman latent coordinates;
- the latent dimension;
- the sampling interval;
- the operating regime;
- the amount of unresolved physics;
- external forcing;
- stochasticity and measurement noise.

Therefore, before selecting Attention, a Markovian MLP, or any closure family, the model should perform a **Residual Structure Assessment**.

The assessment answers three sequential questions:

\[
\boxed{
\text{Is the residual significant?}
\rightarrow
\text{Is it learnable?}
\rightarrow
\text{Does history add predictive information?}
}
\]

This assessment becomes the main routing mechanism between V0.7 and V0.8.

---

## 8. Residual significance

A residual that is technically nonzero may still be too small to justify an additional model.

Define a residual significance score such as

\[
\boxed{
S_R
=
\frac{
\mathbb E\|r_{t+1}^{0}\|^2
}{
\mathbb E\|z_{t+1}^{\rm true}-z_t^K\|^2+\epsilon
}.
}
\]

Alternative robust normalizations may be used if the true latent increment is close to zero.

Interpretation:

\[
S_R \ll 1
\]

means the nominal Koopman model already explains most of the relevant latent evolution.

This case should not trigger a correction model merely because a residual can be measured.

The preferred decision is

\[
\boxed{
\text{RESIDUAL ACTION = NONE}
}
\]

unless later evidence shows that the small residual is scientifically important.

---

## 9. Residual learnability

The residual can be decomposed conceptually into predictable and innovation components.

Let \(\mathcal I_t\) denote the information available at prediction time. Then

\[
\boxed{
r_{t+1}^{0}
=
\mathbb E[r_{t+1}^{0}\mid\mathcal I_t]
+
\varepsilon_{t+1},
}
\]

with

\[
\mathbb E[\varepsilon_{t+1}\mid\mathcal I_t]=0.
\]

The learnable component is

\[
\boxed{
r_{t+1}^{\rm pred}
=
\mathbb E[r_{t+1}^{0}\mid\mathcal I_t].
}
\]

The key question is whether the predictable component is large enough and stable enough to outperform simple baselines on held-out trajectories.

A practical residual predictability score can be defined as

\[
\boxed{
P_R
=
1-
\frac{E_{\rm best}}{E_0+\epsilon},
}
\]

where

- \(E_0\) is the error of a simple baseline such as zero residual or mean residual;
- \(E_{\rm best}\) is the held-out error of the best admissible probe model.

A high \(P_R\) indicates meaningful predictable structure.

A low \(P_R\) means only:

\[
\boxed{
\text{the residual is not predictably learnable from the currently available information.}
}
\]

It must **not** automatically be interpreted as pure data noise.

Possible causes include:

1. measurement/data noise;
2. missing exogenous variables;
3. insufficient Koopman latent coordinates;
4. inadequate encoder or Koopman training;
5. unresolved stochastic forcing;
6. hidden physical variables;
7. numerical/modeling error.

Therefore the unlearnable branch is a **diagnostic branch**, not automatically a denoising branch.

---

## 10. What “history dependence” means

Two states can have nearly identical present latent coordinates,

\[
z_t^{(A)}\approx z_t^{(B)},
\]

while arriving there through different histories.

If

\[
r_{t+1}^{(A)}\neq r_{t+1}^{(B)},
\]

then the present state \(z_t\) is not sufficient to determine the residual.

In operational terms, history dependence means:

\[
\boxed{
\text{past latent states provide predictive information about }r_{t+1}^{0}
\text{ beyond the information already contained in }z_t.
}
\]

This is stronger than saying that the residual time series has autocorrelation.

The real test is whether

\[
F_H(z_{t-H:t},\ldots)
\]

predicts the residual better than

\[
F_M(z_t,\ldots)
\]

under matched conditions.

---

## 11. Markovian and history-dependent probe models

Define the instantaneous information set

\[
\boxed{
\mathcal I_t^{M}
=
(z_t^K,\Delta t_t,\mu_t,a_t,o_t^{exo},\ldots)
}
\]

and the historical information set

\[
\boxed{
\mathcal I_t^{H}
=
(z_{t-H:t}^K,
\Delta t_{t-H:t},
\mu_{t-H:t},
a_{t-H:t},
o_{t-H:t}^{exo},
\ldots).
}
\]

Use a Markovian probe

\[
\boxed{
\hat r_{t+1}^{M}
=
F_M(\mathcal I_t^{M})
}
\]

and a history-aware probe

\[
\boxed{
\hat r_{t+1}^{H}
=
F_H(\mathcal I_t^{H}).
}
\]

Let their held-out errors be

\[
E_M,\qquad E_H.
\]

Define the conditional history gain

\[
\boxed{
G_H
=
\frac{
E_M-E_H
}{
E_M+\epsilon
}.
}
\]

A positive and statistically stable \(G_H\) means history contains useful information beyond the current state.

---

## 12. Controls required before claiming history dependence

History dependence should not be inferred merely because a larger temporal model has lower error.

At minimum, the comparison should consider:

- held-out trajectories;
- matched data split;
- identical Koopman backbone;
- identical normalization;
- comparable optimization budget;
- approximately parameter-matched Markovian and historical probes;
- multiple random seeds for strong claims;
- shuffled-history control.

A shuffled-history control should preserve the current state \(z_t\) while destroying useful ordering or correspondence in older states.

If

\[
F_H
\]

outperforms both

\[
F_M
\]

and

\[
F_{H,\rm shuffled},
\]

the evidence for genuine temporal information is much stronger.

History length \(H\) may still be swept as a secondary diagnostic, but V0.7 no longer needs to force every problem into a universal Markovian/short-memory/long-memory category.

---

## 13. The three residual routes

The revised system should distinguish three structural routes. The magnitude

\[
S_R
\]

is always recorded, but it is not a discard gate: a small one-step residual can
accumulate into a material long-horizon error.

### R1 — Residual not predictably learnable

\[
P_R \text{ weak}.
\]

Interpretation:

\[
\boxed{
\text{The current information set does not contain a stable deterministic predictor of the residual.}
}
\]

This does not imply pure noise.

The diagnosis should test, where relevant:

- denoising/data quality;
- missing parameters;
- missing external forcing;
- latent dimension and encoder adequacy;
- stochastic closure;
- uncertainty modeling;
- problem formulation.

Action:

\[
\boxed{
\text{Diagnose before introducing a deterministic residual learner.}
}
\]

---

### R2 — Residual learnable without meaningful history gain

\[
P_R \text{ strong},
\qquad
G_H \approx 0
\]

within matched experimental uncertainty.

Interpretation:

\[
\boxed{
r_{t+1}^{0}
\approx
F(z_t^K,\Delta t_t,\mu_t,\ldots).
}
\]

The residual is approximately Markovian with respect to the current resolved representation.

Action:

\[
\boxed{
\text{Use an instantaneous context encoder, not Attention by default.}
}
\]

---

### R3 — Residual learnable with meaningful history gain

\[
P_R \text{ strong},
\qquad
G_H>0
\]

with stable held-out evidence.

Interpretation:

\[
\boxed{
r_{t+1}^{0}
\approx
F(z_{t-H:t}^K,\ldots)
}
\]

and the history contains useful information not recoverable from \(z_t^K\) alone.

Action:

\[
\boxed{
\text{Use a temporal context encoder such as causal Attention.}
}
\]

---

## 14. Residual Structure Router

The residual assessment can therefore be summarized as

\[
\boxed{
\Phi_{\rm context}
=
\begin{cases}
\text{diagnostic / stochastic branch},
&
R1:\ \text{not stably learnable},
\\[2mm]
\Phi_M(z_t^K,\Delta t,\mu,\ldots),
&
R2:\ \text{learnable, no history gain},
\\[2mm]
\Phi_H(z_{t-H:t}^K,\Delta t,\mu,\ldots),
&
R3:\ \text{learnable, history-dependent}.
\end{cases}
}
\]

This is **one architecture with a residual-structure router**, not three unrelated world models.

---

## 15. Unified context interface for R2 and R3

The main architectural rule is:

\[
\boxed{
\text{R2 and R3 use different context encoders but the same downstream interface.}
}
\]

### R2: instantaneous context

For a learnable residual with no history gain,

\[
\boxed{
c_t
=
\Phi_M(z_t^K,\Delta t_t,\mu_t,\ldots).
}
\]

The preferred initial implementation is a small MLP.

Example:

\[
d_K
\rightarrow
d_{\rm hidden}
\rightarrow
d_c,
\qquad
d_c\ll d_K.
\]

Here \(c_t\) means:

\[
\boxed{
\text{the compact part of the current state that is relevant to Koopman mismatch and adaptation.}
}
\]

No temporal Attention is required.

---

### R3: historical context

For a learnable residual with history gain,

\[
\boxed{
c_t
=
\Phi_H(z_{t-H:t}^K,\Delta t,\mu,\ldots).
}
\]

The preferred initial implementation is a small causal Attention encoder.

Here \(c_t\) means:

\[
\boxed{
\text{a compact temporal summary relevant to Koopman mismatch and adaptation.}
}
\]

Thus the semantic meaning of \(c_t\) remains unified even though the encoder family changes.

---

## 16. Meaning of \(c_t\)

Define

\[
\boxed{
c_t\in\mathbb R^{d_c},
\qquad
d_c\ll d_K.
}
\]

The fundamental distinction is

\[
\boxed{
z_t^K
=
\text{where the system currently is}
}
\]

versus

\[
\boxed{
c_t
=
\text{what information about the current dynamical context is relevant to Koopman mismatch/adaptation}.
}
\]

For R2, \(c_t\) is extracted from the current state.

For R3, \(c_t\) is extracted from the causal state history.

No explicit ground-truth \(c_t^{true}\) is required.

It is a learned bottleneck trained through downstream objectives.

---

## 17. Residual supervision of the context representation

Regardless of whether \(c_t\) comes from an MLP or Attention, it should first be validated by residual prediction.

Define

\[
\boxed{
\hat r_{t+1}^{0}
=
R_\phi(c_t,z_t^K,\Delta t_t).
}
\]

The residual supervision loss is

\[
\boxed{
L_{\rm residual}
=
\left\|
\hat r_{t+1}^{0}
-
r_{t+1}^{0}
\right\|^2.
}
\]

Residual is therefore the main teacher signal for context learning.

The key principle is:

\[
\boxed{
\text{Residual supervises }c_t,
\text{ but future ground-truth residual is not an inference-time input.}
}
\]

---

## 18. Koopman adequacy head

The same context can predict the expected magnitude of nominal Koopman mismatch:

\[
\boxed{
\hat m_{t+1}
=
Q_\chi(c_t).
}
\]

A simple target is

\[
m_{t+1}
=
\|r_{t+1}^{0}\|.
\]

This head answers:

\[
\boxed{
\text{How inadequate is the nominal Koopman model likely to be next?}
}
\]

A later change-probability head may be introduced if a scientifically justified definition of regime change exists, but V0.8 should prefer continuous mismatch targets before hard binary labels.

---

## 19. Meaning of \(\eta_t\)

After \(c_t\) has been validated, a later operator-adaptation head produces

\[
\boxed{
\eta_t
=
G_\omega(c_t).
}
\]

The role of \(\eta_t\) is fundamentally different from \(c_t\).

\[
\boxed{
c_t = \text{context description}
}
\]

whereas

\[
\boxed{
\eta_t = \text{operator-adjustment coordinates}.
}
\]

A useful intuitive form is

\[
\boxed{
A_t
=
A_0
+
\sum_{i=1}^{r}
\eta_{t,i}B_i.
}
\]

The learned matrices

\[
B_1,\ldots,B_r
\]

represent admissible directions in generator space, while

\[
\eta_{t,i}
\]

determines how strongly each direction is activated at time \(t\).

Thus

\[
\boxed{
z_t
\rightarrow
c_t
\rightarrow
\eta_t
\rightarrow
A_t
}
\]

means:

1. observe the current resolved state/context;
2. summarize the relevant dynamical context;
3. decide how the nominal operator should change;
4. construct the adapted Koopman generator.

---

## 20. Low-rank adaptive Koopman operator

The preferred primary implementation remains

\[
\boxed{
A_t
=
A_0
+
U\operatorname{diag}(\eta_t)V^\top
}
\]

with

\[
U,V\in\mathbb R^{d_K\times r},
\qquad
r\ll d_K.
\]

Equivalently,

\[
A_t
=
A_0
+
\sum_{i=1}^{r}
\eta_{t,i}B_i.
\]

The adaptive propagation is

\[
\boxed{
z_{t+1}^{A}
=
e^{A_t\Delta t_t}z_t^K.
}
\]

The low-rank restriction prevents the context encoder from freely rewriting the full Koopman dynamics.

---

## 21. Alternative: Mixture of Koopman generators

For problems with clearly discrete dynamical regimes, a secondary route is

\[
\alpha_t
=
\operatorname{softmax}(Wc_t),
\qquad
\sum_{m=1}^{M}\alpha_{t,m}=1
\]

and

\[
\boxed{
A_t
=
\sum_{m=1}^{M}
\alpha_{t,m}A_m.
}
\]

This route should be treated as an alternative architecture for sufficiently discrete regime switching, not the universal default.

---

## 22. Residual decomposition after operator adaptation

The nominal residual is

\[
r_{t+1}^{0}
=
z_{t+1}^{\rm true}
-
e^{A_0\Delta t_t}z_t^K.
\]

The part explained by operator adaptation is

\[
\boxed{
r_{t+1}^{op}
=
\left(
e^{A_t\Delta t_t}
-
e^{A_0\Delta t_t}
\right)z_t^K.
}
\]

The remaining residual is

\[
\boxed{
r_{t+1}^{rem}
=
z_{t+1}^{\rm true}
-
e^{A_t\Delta t_t}z_t^K.
}
\]

Therefore

\[
\boxed{
r_{t+1}^{0}
=
r_{t+1}^{op}
+
r_{t+1}^{rem}.
}
\]

This remains one of the central mathematical decompositions of the architecture.

---

## 23. Operator-explained fraction

Define

\[
\boxed{
\Gamma_{op}
=
1-
\frac{
\mathbb E\|r^{rem}\|^2
}{
\mathbb E\|r^{0}\|^2+\epsilon
}.
}
\]

Interpretation:

- \(\Gamma_{op}\approx 1\): operator adaptation explains most nominal residual;
- \(\Gamma_{op}\approx 0\): adaptation explains little;
- \(\Gamma_{op}<0\): adaptation worsens the residual.

This metric directly tests whether the residual was caused mainly by an inappropriate fixed Koopman operator.

---

## 24. Identifiability constraint

The operator adaptation and additive residual correction must not be freely trained together from the beginning.

Otherwise both can explain the same prediction error:

\[
r^0
\approx
r^{op}
\]

or

\[
r^0
\approx
\hat r^{closure},
\]

making the decomposition scientifically ambiguous.

Therefore the development order must remain:

\[
\boxed{
\text{Residual assessment}
\rightarrow
\text{Context learning}
\rightarrow
\text{Operator adaptation}
\rightarrow
\text{Residual reassessment}
\rightarrow
\text{Closure only if necessary}.
}
\]

---

## 25. Optional remaining residual closure

Only if

\[
r^{rem}
\]

is still:

1. non-negligible;
2. stable;
3. predictably learnable;

should an explicit closure state be introduced.

For example,

\[
\boxed{
z_t^R
=
M_\rho(c_t,z_{t-H:t}^K)
}
\]

and

\[
\boxed{
\Delta z_{t+1}^{R}
=
C_R(z_t^R).
}
\]

The final prediction becomes

\[
\boxed{
\hat z_{t+1}
=
e^{A_t\Delta t_t}z_t^K
+
\Delta z_{t+1}^{R}.
}
\]

The semantic role of \(z_R\) is

\[
\boxed{
z_R
=
\text{unresolved closure state remaining after adaptive Koopman dynamics}.
}
\]

---

## 26. PhysicsConstraint

Decode

\[
\hat U_{t+1}
=
D(\hat z_{t+1})
\]

and apply

\[
\boxed{
PhysicsConstraint(\hat U_{t+1}).
}
\]

The long-term architecture therefore becomes

\[
\boxed{
\text{JEPA representation}
+
\text{Residual Structure Assessment}
+
\text{Adaptive Context Family}
+
\text{Adaptive Koopman}
+
\text{Optional Residual Closure}
+
\text{PhysicsConstraint}.
}
\]

---

# 27. Updated Version Roadmap

## V0.7 — Residual Structure Assessment & Koopman Adequacy

### Single core scientific question

\[
\boxed{
\text{What structure, if any, exists in the nominal Koopman residual }r^0?
}
\]

V0.7 should no longer be framed primarily as a memory-classification version.

Its goal is to determine what predictable structure exists in the residual and
what information is required to represent it.

### V0.7-A — Residual magnitude diagnostic

Measure

\[
S_R
\]

as a reference-scale diagnostic. The result must not be used to discard the
residual or terminate routing, because temporal accumulation can amplify a
small one-step discrepancy.

Output:

\[
\boxed{
\text{RESIDUAL MAGNITUDE:
LOW / MATERIAL / INCONCLUSIVE}
}
\]

### V0.7-B — Residual learnability

Use zero, linear, small nonlinear, and other appropriately controlled probes to determine whether \(r^0\) contains predictable structure.

Output:

\[
\boxed{
\text{RESIDUAL LEARNABILITY:
STRONG / MODERATE / WEAK / NONE}
}
\]

The label NONE/WEAK must not automatically be interpreted as data noise.

### V0.7-C — Conditional history gain

Compare

\[
F_M(z_t,\ldots)
\]

against

\[
F_H(z_{t-H:t},\ldots).
\]

Use

\[
G_H
=
\frac{E_M-E_H}{E_M+\epsilon}
\]

together with matched controls.

Output:

\[
\boxed{
\text{HISTORY GAIN:
PRESENT / ABSENT / INCONCLUSIVE}
}
\]

### V0.7-D — Residual routing decision

The final result should classify the tested problem/backbone/regime as:

\[
\boxed{
R1,\ R2,\ \text{or }R3.
}
\]

where

- R1 = residual not predictably learnable from the tested information;
- R2 = residual learnable, no meaningful history gain;
- R3 = residual learnable, meaningful history gain.

### V0.7-E — Koopman adequacy

Retain

\[
m_t=\|r_{t+1}^{0}\|
\]

and robust normalized variants.

Analyze whether residual magnitude is structured in time or state space and whether it can serve as a useful Koopman inadequacy signal.

### V0.7-F — Existing memory analysis

Existing history-length sweeps, ACF diagnostics, Mori–Zwanzig-inspired interpretation, and closed-loop probes remain useful.

However, they become supporting evidence rather than the main universal classification target.

### V0.7 completion result

V0.7 must end with a machine-readable decision conceptually equivalent to:

```text
RESIDUAL_MAGNITUDE: LOW_MAGNITUDE / MATERIAL_MAGNITUDE / INCONCLUSIVE
RESIDUAL_LEARNABILITY: STRONG / MODERATE / WEAK / NONE
HISTORY_GAIN: PRESENT / ABSENT / INCONCLUSIVE
RESIDUAL_ROUTE: R1 / R2 / R3 / INCONCLUSIVE
```

This result determines whether and how V0.8 proceeds.

---

## V0.8 — Residual-Supervised Dynamic Context Learning

### Core question

\[
\boxed{
\text{Given the residual structure selected by V0.7, can we learn a compact }c_t
\text{ that predicts nominal Koopman mismatch?}
}
\]

V0.8 is no longer synonymous with Attention.

It is a **context-family version**.

### R1 path

If V0.7 returns R1:

\[
\boxed{
\text{Do not blindly train deterministic Attention/MLP closure.}
}
\]

Enter a diagnostic branch:

- data quality;
- missing variables;
- latent adequacy;
- external forcing;
- stochastic modeling.

### R2 path — Instantaneous context

Use

\[
\boxed{
c_t
=
\Phi_M(z_t^K,\Delta t,\mu,\ldots)
}
\]

with a small MLP as the default.

No Attention is required.

### R3 path — Historical context

Use

\[
\boxed{
c_t
=
\Phi_H(z_{t-H:t}^K,\Delta t,\mu,\ldots)
}
\]

with small causal Attention as the primary temporal encoder.

### Unified V0.8 heads

For both R2 and R3:

\[
\boxed{
c_t
\rightarrow
\hat r_{t+1}^{0}
}
\]

and

\[
\boxed{
c_t
\rightarrow
\hat m_{t+1}.
}
\]

The main losses are

\[
L_{\rm residual}
=
\|\hat r_{t+1}^{0}-r_{t+1}^{0}\|^2
\]

and an adequacy-prediction loss such as

\[
L_{\rm adequacy}
=
|\hat m_{t+1}-m_{t+1}|^2.
\]

### Critical V0.8 restriction

Keep

\[
\boxed{
A_t=A_0.
}
\]

V0.8 must validate context learning **before** the context is allowed to modify Koopman dynamics.

### Main V0.8 scientific outputs

1. Was the V0.7 route correct?
2. Does the chosen context encoder outperform appropriate controls?
3. Does \(c_t\) predict residual structure?
4. Does \(c_t\) predict Koopman inadequacy?
5. For R3, does causal Attention outperform parameter-matched instantaneous and shuffled-history controls?

---

## V0.9 — Context-Conditioned Adaptive Koopman

### Core question

\[
\boxed{
\text{Can the validated context }c_t
\text{ modify the Koopman generator and explain nominal residual?}
}
\]

At this point R2 and R3 **converge to the same downstream interface**.

The source of \(c_t\) may differ:

\[
c_t
=
\begin{cases}
\Phi_M(z_t), & R2,\\
\Phi_H(z_{t-H:t}), & R3,
\end{cases}
\]

but the downstream operator adaptation is identical.

### Operator adaptation

\[
\boxed{
\eta_t
=
G_\omega(c_t)
}
\]

followed by

\[
\boxed{
A_t
=
A_0
+
U\operatorname{diag}(\eta_t)V^\top.
}
\]

Equivalent basis form:

\[
A_t
=
A_0
+
\sum_i\eta_{t,i}B_i.
\]

### Critical V0.9 restriction

During the primary experiment, disable additive residual correction.

Prediction must be

\[
\boxed{
\hat z_{t+1}
=
e^{A_t\Delta t_t}z_t.
}
\]

This isolates the explanatory power of operator adaptation.

### Primary loss

\[
\boxed{
L_A
=
\left\|
z_{t+1}^{\rm true}
-
e^{A_t\Delta t_t}z_t
\right\|^2.
}
\]

Include multi-step rollout evaluation.

### Alternative operator family

Mixture-of-Koopman may be evaluated for sufficiently discrete regime-switching problems, but is not the default.

---

## V1.0 — Adaptive Koopman Physical World Model Baseline

### Architecture

\[
\boxed{
PhysicsConstraint
+
JEPA\ z_K
+
Selected\ Context\ Encoder
+
Adaptive\ Koopman
}
\]

### Main task

Recompute

\[
\boxed{
r_{t+1}^{rem}
=
z_{t+1}^{\rm true}
-
e^{A_t\Delta t_t}z_t.
}
\]

Compare it against

\[
r_{t+1}^{0}.
\]

### Main metric

\[
\boxed{
\Gamma_{op}
=
1-
\frac{
\mathbb E\|r^{rem}\|^2
}{
\mathbb E\|r^{0}\|^2+\epsilon
}.
}
\]

### Scientific question

\[
\boxed{
\text{How much of the nominal residual was actually due to an inadequate fixed Koopman operator?}
}
\]

If

\[
r^{rem}
\]

is negligible, the architecture should stop here rather than introduce \(z_R\) unnecessarily.

---

## V1.1 — Remaining Residual Reassessment and Optional Closure

Before creating a full residual closure, reassess

\[
r^{rem}
\]

using the same philosophy as V0.7.

Ask:

1. Is \(r^{rem}\) significant?
2. Is \(r^{rem}\) learnable?
3. Does \(r^{rem}\) still require history?

Only if the answer justifies deterministic closure should \(z_R\) be introduced.

Then

\[
\boxed{
z_t^R
=
M_\rho(c_t,z_{t-H:t}^K)
}
\]

and

\[
\boxed{
\hat z_{t+1}
=
e^{A_t\Delta t_t}z_t
+
C_R(z_t^R).
}
\]

Thus \(z_R\) is reserved for genuinely unresolved dynamics after operator adaptation.

---

## V1.2 — Controlled Joint Fine-Tuning

Only after the previous components are independently validated should joint training be attempted.

Possible trainable modules include

\[
E_\theta,
\quad
\Phi_{\rm context},
\quad
G_\omega,
\quad
A_t,
\quad
z_R,
\quad
D.
\]

Strict controls are required to prevent the context encoder or residual closure from absorbing the full dynamics and making Koopman structure meaningless.

---

# 28. Updated architecture summary

The revised logic is

\[
\boxed{
\text{Koopman Backbone}
\rightarrow
\text{Nominal Residual}
\rightarrow
\text{Residual Structure Assessment}
\rightarrow
\text{Context Family Selection}
\rightarrow
\text{Adaptive Koopman}
\rightarrow
\text{Residual Reassessment}
\rightarrow
\text{Closure if necessary}.
}
\]

The context family is

\[
\boxed{
\Phi_{\rm context}
=
\begin{cases}
\text{diagnostic/stochastic}, & R1,\\
\Phi_M, & R2,\\
\Phi_H, & R3.
\end{cases}
}
\]

For the two deterministic learnable cases,

\[
\boxed{
R2:
\quad
z_t
\rightarrow
\text{small MLP}
\rightarrow
c_t
}
\]

and

\[
\boxed{
R3:
\quad
z_{t-H:t}
\rightarrow
\text{causal Attention}
\rightarrow
c_t.
}
\]

After \(c_t\), both branches share the same interface:

\[
\boxed{
c_t
\rightarrow
\left(
\hat r_{t+1}^{0},
\hat m_{t+1},
\eta_t
\right).
}
\]

In V0.8 only \(\hat r^0\) and \(\hat m\) are active as context-validation heads.

In V0.9, \(\eta_t\) becomes active and modifies Koopman dynamics:

\[
\boxed{
A_t
=
A_0
+
\sum_i\eta_{t,i}B_i.
}
\]

---

# 29. Final mathematical core

The latest framework can be condensed to the following sequence.

### Nominal Koopman model

\[
\boxed{
z_t^K=E_\theta(U_t)
}
\]

\[
\boxed{
z_{t+1}^{0}
=
e^{A_0\Delta t_t}z_t^K
}
\]

\[
\boxed{
r_{t+1}^{0}
=
z_{t+1}^{\rm true}
-
z_{t+1}^{0}
}
\]

### Residual structure assessment

\[
\boxed{
(P_R,G_H)\quad\text{with diagnostic }S_R
\rightarrow
R1/R2/R3
}
\]

### Context selection

\[
\boxed{
c_t
=
\begin{cases}
\Phi_M(z_t,\ldots), & R2,\\
\Phi_H(z_{t-H:t},\ldots), & R3.
\end{cases}
}
\]

### Residual-supervised context learning

\[
\boxed{
c_t
\rightarrow
(\hat r_{t+1}^{0},\hat m_{t+1})
}
\]

### Operator adaptation

\[
\boxed{
\eta_t
=
G_\omega(c_t)
}
\]

\[
\boxed{
A_t
=
A_0
+
U\operatorname{diag}(\eta_t)V^\top
}
\]

### Residual decomposition

\[
\boxed{
r^{0}
=
\underbrace{
\left(
e^{A_t\Delta t}
-
e^{A_0\Delta t}
\right)z_t
}_{r^{op}}
+
\underbrace{
\left(
z_{t+1}^{\rm true}
-
e^{A_t\Delta t}z_t
\right)
}_{r^{rem}}
}
\]

### Optional closure

Only if \(r^{rem}\) remains significant and learnable,

\[
\boxed{
\hat z_{t+1}
=
e^{A_t\Delta t_t}z_t
+
C_R(z_t^R).
}
\]

---

# 30. Final interpretation

The architecture should no longer be described simply as

\[
\text{Koopman + Attention}.
\]

The more accurate description is

\[
\boxed{
\text{JEPA Representation}
+
\text{Koopman Backbone}
+
\text{Residual Structure Assessment}
+
\text{Adaptive Context Family}
+
\text{Adaptive Koopman}
+
\text{Optional Residual Closure}
+
\text{PhysicsConstraint}.
}
\]

Attention remains an important component, but it is selected specifically when the tested residual contains useful history-dependent information.

For learnable residuals that do not benefit from history, a smaller instantaneous context model should be preferred.

The final scientific principle is therefore:

\[
\boxed{
\text{Measure residual structure first; choose model complexity only when the residual structure justifies it.}
}

---

# 31. V0.8–V1.0 Physical Benchmark Program

The post-V0.7 versions should no longer continue indefinitely on the simplest periodic advection–diffusion toy problem.

The reason is structural: V0.8 and later versions now test

\[
\boxed{
\text{history-dependent context}
\rightarrow
\text{dynamic-context representation}
\rightarrow
\text{adaptive Koopman dynamics},
}
\]

so the benchmark itself must contain sufficiently rich transient dynamics and, from V0.9 onward, controlled changes in operating conditions.

The preferred strategy is to use **one continuous physical problem family from V0.8 through V1.0**, rather than changing to an unrelated PDE at every version.

This allows the scientific progression

\[
\boxed{
\text{fixed-condition transient dynamics}
\rightarrow
\text{controlled condition change}
\rightarrow
\text{unseen condition transitions}
}
\]

to correspond directly to the architectural progression

\[
\boxed{
c_t
\rightarrow
\eta_t
\rightarrow
A_t.
}
\]

---

## 31.1 Benchmark selection principle

V0.8 should use a problem that is expected to contain rich temporal structure, but it must still pass the V0.7 Residual Structure Assessment.

A fluid problem should **not** be declared R3 merely because it is a fluid problem.

The intended workflow is:

\[
\boxed{
\text{candidate transient flow problem}
\rightarrow
\text{V0.7 assessment}
\rightarrow
R3\ \text{confirmation}
\rightarrow
\text{formal V0.8 benchmark}.
}
\]

The formal V0.8 benchmark should therefore satisfy

\[
S_R > 0,
\qquad
P_R > 0,
\qquad
G_H > 0
\]

with statistically credible held-out evidence.

If the chosen problem is unexpectedly classified as R2, the architecture should respect that result and use the instantaneous-context branch instead of forcing Attention.

---

## 31.2 Preferred main benchmark: 2D cylinder wake

The preferred V0.8–V1.0 main physical benchmark is

\[
\boxed{
\text{two-dimensional incompressible cylinder wake}.
}
\]

This problem is attractive because a compact low-dimensional Koopman representation can experience:

- initial transient development;
- vortex-formation history;
- amplitude growth toward vortex shedding;
- oscillatory limit-cycle behavior;
- phase-dependent dynamics;
- controllable Reynolds-number changes;
- controllable inflow boundary-condition changes.

The same geometry and PDE can therefore support several consecutive versions without changing the basic problem definition.

---

## 31.3 V0.8 physical problem: fixed-condition transient cylinder wake

The V0.8 benchmark should first keep the physical operating condition fixed.

For example,

\[
U_\infty(t)=U_0,
\qquad
Re(t)=Re_0.
\]

Different trajectories should be generated from different initial perturbations or physically admissible initial states, while the governing PDE, geometry, and boundary conditions remain fixed.

The trajectory should contain the transition from initial transient flow toward developed vortex shedding.

The V0.8 physical question becomes

\[
\boxed{
\text{Under a fixed physical condition, does causal history contain information that helps predict }r^0?
}
\]

The model remains

\[
A_t=A_0.
\]

For an R3 problem,

\[
\boxed{
z_{t-H:t}
\rightarrow
\Phi_H
\rightarrow
c_t
\rightarrow
(\hat r_{t+1}^{0},\hat m_{t+1}).
}
\]

No operator adaptation is activated yet.

This isolates the scientific role of history and dynamic-context learning.

---

## 31.4 Why fixed boundary conditions are important in V0.8

If V0.8 simultaneously introduces changing inlet conditions and temporal Attention, then two effects become mixed:

1. history dependence under a single dynamical law;
2. changing dynamical law due to changing operating conditions.

V0.8 should separate these effects.

Therefore the preferred sequence is

\[
\boxed{
\text{V0.8: fixed operating condition}
}
\]

followed by

\[
\boxed{
\text{V0.9: controlled operating-condition variation}.
}
\]

This keeps each version associated with one main scientific question.

---

# 32. V0.9 Physical Benchmark: Controlled Boundary/Operating-Condition Change

V0.9 activates context-conditioned Koopman adaptation.

Therefore the physical problem should contain a controlled situation in which the nominal generator

\[
A_0
\]

is expected to become insufficient.

The simplest and preferred mechanism is a controlled change of inflow velocity and therefore Reynolds number.

---

## 32.1 Abrupt change

A step change may be defined as

\[
U_\infty(t)
=
\begin{cases}
U_1, & t<t_c,\\
U_2, & t\ge t_c.
\end{cases}
\]

Correspondingly,

\[
Re_1
\rightarrow
Re_2.
\]

The model must determine whether the previously learned context representation can produce

\[
\eta_t
\]

that modifies

\[
A_0
\]

into a suitable

\[
A_t.
\]

The primary pipeline is

\[
\boxed{
z_{t-H:t}
\rightarrow
c_t
\rightarrow
\eta_t
\rightarrow
A_t
\rightarrow
e^{A_t\Delta t}z_t.
}
\]

The additive residual correction remains disabled during the primary V0.9 experiment.

---

## 32.2 Smooth change

A second experiment should use a smooth ramp

\[
U_\infty(t)
=
U_1
+
(U_2-U_1)s(t),
\]

where \(s(t)\) is a smooth transition function.

This tests continuous operator modulation rather than abrupt switching.

The low-rank operator model

\[
\boxed{
A_t
=
A_0
+
U\operatorname{diag}(\eta_t)V^\top
}
\]

is particularly suitable for this experiment.

---

## 32.3 Abrupt versus smooth operator adaptation

The two V0.9 transition types have different scientific interpretations.

### Smooth transition

Tests whether

\[
\eta_t
\]

behaves as a continuous modulation coordinate.

### Abrupt transition

Tests whether the context representation can react quickly to a change in the active dynamical regime.

A Mixture-of-Koopman alternative may be considered for clearly discrete switching, while low-rank continuous modulation remains the primary route.

---

# 33. Known-condition and latent-inferred-condition experiments

The V0.9 benchmark should preferably contain two levels.

## 33.1 Known-condition experiment

Provide the current operating parameter explicitly:

\[
\boxed{
c_t
=
\Phi_H(z_{t-H:t},Re_t)
}
\]

or the corresponding R2 instantaneous form.

This tests whether a parameter-conditioned adaptive Koopman architecture can function correctly.

It is the easier architectural sanity check.

---

## 33.2 Latent-inferred-condition experiment

Do not provide the true \(Re_t\) or inlet-change label.

Use only the dynamical history:

\[
\boxed{
c_t
=
\Phi_H(z_{t-H:t}).
}
\]

Then evaluate whether \(c_t\) can infer enough information from the observed dynamics to drive a useful

\[
\eta_t
\]

and adaptive generator.

This experiment directly tests the intended idea that

\[
\boxed{
\text{dynamic context can be inferred from the evolution itself}.
}
\]

If the latent-inferred model approaches the known-condition model, that provides strong evidence that \(c_t\) captures meaningful dynamical context.

---

# 34. V1.0 Physical Benchmark: Unseen Condition Transitions

V1.0 should continue using the same cylinder-wake family.

The goal is no longer merely to reproduce one known transition.

The model should be evaluated on transitions that differ from training conditions.

Training may include several examples such as

\[
Re_a\rightarrow Re_b,
\qquad
Re_b\rightarrow Re_c,
\qquad
Re_c\rightarrow Re_b.
\]

Testing should include held-out combinations such as

\[
Re_d\rightarrow Re_e
\]

with one or more of the following differences:

- unseen initial Reynolds number;
- unseen final Reynolds number;
- unseen transition time;
- unseen ramp rate;
- reverse transition;
- repeated condition changes.

The V1.0 scientific question becomes

\[
\boxed{
\text{Does }c_t\rightarrow\eta_t\rightarrow A_t
\text{ generalize as an operator-adaptation mechanism rather than memorizing one transition?}
}
\]

The main residual comparison remains

\[
r^0
=
r^{op}
+
r^{rem}
\]

and

\[
\Gamma_{op}
=
1-
\frac{
\mathbb E\|r^{rem}\|^2
}{
\mathbb E\|r^0\|^2+\epsilon
}.
\]

---

# 35. Benchmark continuity across versions

The preferred progression is therefore

\[
\boxed{
\text{V0.8:
fixed-condition transient cylinder wake}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{V0.9:
smooth and abrupt inflow/Re changes}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{V1.0:
unseen operating-condition transitions}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{V1.1:
reassess }r^{rem}\text{ and add closure only if justified}.
}
\]

Using the same PDE and geometry makes the scientific progression more interpretable and allows V0.8 context learning to serve as a real foundation for V0.9 adaptive dynamics.

---

# 36. Single-GPU Compute Constraint

All post-V0.7 benchmark design should satisfy the practical requirement

\[
\boxed{
\text{formal training and validation must be feasible on one NVIDIA RTX 5080}.
}
\]

Multi-GPU training must not be required for the baseline research program.

The user has ample system RAM and storage, so dataset caching, trajectory storage, offline preprocessing, and residual-cache generation may use CPU memory and disk aggressively when useful.

The main bottleneck to control is GPU VRAM and training-time complexity.

---

## 36.1 Compute-design principle

The project should prefer

\[
\boxed{
\text{scientifically sufficient resolution}
}
\]

over unnecessarily large CFD or neural-network scale.

The purpose of V0.8–V1.0 is to test the architecture:

- residual structure;
- context learning;
- operator adaptation;
- generalization across controlled operating changes.

It is not yet intended to reproduce high-Reynolds-number production CFD.

---

## 36.2 Recommended 2D CFD scale

The first formal cylinder-wake dataset should remain moderate.

A practical initial target is approximately

\[
N_x\times N_y
\sim
\mathcal O(10^4-10^5)
\]

grid/cell degrees of freedom rather than multi-million-cell CFD.

For structured or Cartesian formulations, a starting resolution in the broad range

\[
\boxed{
128\times64
\ \text{to}\
256\times128
}
\]

is suitable for architecture development, subject to the actual immersed-boundary/body representation and numerical stability.

Higher resolution should only be introduced after the architecture passes at smaller scale.

---

## 36.3 Dataset storage strategy

Because system RAM and storage are not the primary limitation, trajectories should preferably be:

1. generated or collected offline;
2. stored on disk;
3. normalized using train-only statistics;
4. optionally memory-mapped;
5. converted into latent/residual caches where scientifically appropriate.

In particular, once the V0.6/V0.7 backbone is frozen, it is reasonable to precompute:

\[
z_t^K,
\qquad
r_{t+1}^{0},
\qquad
m_t
\]

for V0.7/V0.8 context experiments.

This can greatly reduce repeated GPU decoding/encoding cost.

The cache must preserve checkpoint, split, normalization, data, and residual-definition fingerprints.

---

## 36.4 Latent dimension

The initial post-V0.7 flow benchmark should keep

\[
d_K
\]

moderate.

A preferred exploratory range is

\[
\boxed{
d_K\sim 16-64.
}
\]

The value should be chosen from representation-quality and residual-significance studies rather than made large by default.

If a larger \(d_K\) dramatically removes the residual, that is itself an important scientific result and should be reported rather than hidden by adding a larger closure model.

---

## 36.5 Context dimension

The context bottleneck should remain smaller than the Koopman latent:

\[
\boxed{
d_c\ll d_K.
}
\]

A practical initial range is

\[
\boxed{
d_c\sim4-16.
}
\]

This prevents \(c_t\) from becoming a second unrestricted latent state.

---

## 36.6 Attention scale for R3

The R3 Attention encoder should remain intentionally small.

A suitable first search space is approximately:

- 2–4 causal Attention blocks;
- model width 64–128;
- 2–4 attention heads;
- moderate history length;
- no giant Transformer backbone;
- no full spatial-grid Transformer.

Attention operates on the **Koopman latent history**, not directly on every CFD grid point.

Therefore its complexity is approximately governed by the history length and latent embedding dimension, not the full physical grid size.

This is essential for single-5080 feasibility.

---

## 36.7 History length

The history horizon should be discovered empirically rather than made extremely long.

A staged sweep such as

\[
H\in\{1,2,4,8,16,32\}
\]

may be used depending on sampling rate.

Only the smallest history length that captures most stable history gain should be retained for formal training.

The physical memory time

\[
T_H
=
\sum_{j=t-H+1}^{t}\Delta t_j
\]

should be reported along with step count.

---

## 36.8 Batch size and sequence strategy

Formal experiments should use the largest batch size that fits comfortably on one RTX 5080, but architecture decisions must not depend on very large batches.

Preferred strategies include:

- latent-cache training for V0.8;
- mixed precision where numerically safe;
- gradient accumulation if required;
- short-to-moderate rollout windows during training;
- longer closed-loop rollouts mainly during evaluation;
- checkpointing/resume;
- CPU-side data caching/prefetching.

---

## 36.9 Decoder usage

The physical decoder can be one of the largest memory consumers.

Therefore V0.8 residual/context training should primarily operate in latent space after the backbone is validated and frozen.

Physical-space decoding should be used for:

- validation;
- selected training regularization if scientifically required;
- PhysicsConstraint evaluation;
- long-rollout physical metrics.

It should not be redundantly executed for every auxiliary residual probe if this does not affect the scientific question.

---

## 36.10 Adaptive Koopman cost

The adaptive generator should remain computationally modest.

Low-rank modulation

\[
A_t
=
A_0+
U\operatorname{diag}(\eta_t)V^\top
\]

is preferable to producing a fully unrestricted matrix at every time step.

With moderate

\[
d_K
\]

and low rank

\[
r,
\]

matrix-exponential propagation remains manageable on a single GPU.

The model should avoid a large ensemble of high-dimensional Koopman experts unless the low-rank route clearly fails and mixture modeling is scientifically justified.

---

## 36.11 CFD generation versus network training

The architecture program should distinguish two costs:

### CFD/data generation

Can be performed offline and may use CPU resources, large system memory, and storage.

### Neural-network training

Must remain compatible with one RTX 5080.

The benchmark should therefore avoid requiring extremely expensive high-fidelity transient CFD generation for every training epoch.

Prefer reusable offline trajectories.

---

## 36.12 CPU-first and single-GPU workflow

The preferred workflow remains:

\[
\boxed{
\text{local CPU/MPS smoke test}
\rightarrow
\text{Git commit/push}
\rightarrow
\text{single RTX 5080 formal run}.
}
\]

All critical tensor-shape, causal-history, residual-alignment, checkpoint, and routing tests must remain runnable without the RTX 5080.

GPU runs should be reserved for:

- formal context training;
- larger history sweeps;
- adaptive-Koopman rollout;
- multi-seed validation;
- full physical decoding.

---

# 37. Fallback benchmark if cylinder wake is too expensive

If the selected cylinder-wake solver or dataset generation proves unnecessarily expensive for the architectural question, a lower-cost fallback is

\[
\boxed{
\text{2D lid-driven cavity with time-varying lid velocity}.
}
\]

The operating condition can be changed through

\[
U_{\rm lid}(t).
\]

Advantages include:

- simple geometry;
- simple boundary control;
- no open outflow boundary;
- relatively low computational cost;
- straightforward smooth and abrupt condition changes.

However, if computationally feasible on one RTX 5080, the cylinder wake remains the preferred benchmark because its transient and oscillatory dynamics are richer for testing history-dependent context.

---

# 38. Updated post-V0.7 experimental program

The recommended overall experiment sequence is now:

### V0.7

Develop and validate the general

\[
\boxed{
\text{Residual Structure Assessment}
}
\]

and obtain R1/R2/R3 routing while retaining residual magnitude as a diagnostic.

### V0.8

Use a candidate transient flow problem and first verify that it is R3.

Then train

\[
\boxed{
z_{t-H:t}
\rightarrow
Attention
\rightarrow
c_t
\rightarrow
(\hat r^0,\hat m)
}
\]

under fixed operating conditions.

If the benchmark is classified R2 instead, use the instantaneous MLP route and record that scientific result rather than forcing R3.

### V0.9

On the same geometry/PDE, introduce controlled smooth and abrupt boundary-condition or Reynolds-number changes.

Activate

\[
\boxed{
c_t
\rightarrow
\eta_t
\rightarrow
A_t.
}
\]

Compare known-condition and latent-inferred-condition settings.

### V1.0

Evaluate unseen transition times, rates, and operating-condition pairs.

Quantify

\[
r^0,
\qquad
r^{op},
\qquad
r^{rem},
\qquad
\Gamma_{op}.
\]

### V1.1

Reassess the remaining residual.

Only introduce \(z_R\) if its significance and learnability justify a closure model.

### V1.2

Perform controlled joint fine-tuning only after every preceding architectural component has independently demonstrated value.

---

# 39. Final hardware-aware design principle

The benchmark should be difficult enough to expose the need for dynamic context and adaptive Koopman dynamics, but not so large that computational scale becomes the dominant research problem.

The preferred principle is

\[
\boxed{
\text{minimum physical and model complexity required to falsify the architectural hypothesis}.
}
\]

For the present project this means:

\[
\boxed{
\text{2D flow}
+
\text{moderate spatial resolution}
+
\text{compact }z_K
+
\text{small context model}
+
\text{low-rank adaptive Koopman}
+
\text{single RTX 5080}.
}
\]

This constraint should be treated as part of the architecture design rather than as an afterthought.
