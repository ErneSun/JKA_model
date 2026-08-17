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

## 7. Attention input

Attention must not depend on future ground-truth residuals at inference time.

Define the causal latent history

\[
\mathcal H_t=
\{z_{t-H+1}^K,\ldots,z_t^K\}.
\]

A more general future form is

\[
\mathcal I_t
=
(
z_{t-H:t}^K,
a_{t-H:t},
\mu_{t-H:t},
o_{t-H:t}^{exo},
\Delta t
).
\]

Then

\[
\boxed{
c_t=\mathcal A_\psi(\mathcal I_t)
}
\]

where \(c_t\) is the learned dynamic context representation.

## 8. Residual as teacher, not inference-time input

During training, \(r_{t+1}^{0}\) is available and provides supervision:

\[
\mathcal H_t
\rightarrow
\mathcal A_\psi
\rightarrow
c_t
\rightarrow
\hat r_{t+1}^{0}.
\]

Define the residual head

\[
\boxed{
\hat r_{t+1}^{0}
=
R_\phi(c_t,z_t^K,\Delta t_t)
}
\]

with

\[
\boxed{
L_R
=
\left\|
\hat r_{t+1}^{0}
-
r_{t+1}^{0}
\right\|^2.
}
\]

Thus residual teaches Attention which part of the upcoming dynamics the nominal Koopman model is likely to miss.

## 9. Dynamic context representation \(c_t\)

Define

\[
\boxed{
c_t\in\mathbb R^{d_c},
\qquad
d_c\ll d_K.
}
\]

Its semantic role is

\[
\boxed{
c_t = \text{compact temporal dynamic context}
}
\]

with the distinction

\[
\boxed{
z_t^K = \text{where the system is}
}
\]

and

\[
\boxed{
c_t = \text{what dynamical context the system is currently in}.
}
\]

It may implicitly encode onset of regime change, dominant-frequency changes, damping/growth changes, changing mode importance, upcoming Koopman mismatch, transient evolution, or parameter-dependent dynamical context.

No explicit \(c_t^{true}\) is required; \(c_t\) is learned through downstream tasks.

## 10. Recommended Attention construction

A simple causal temporal Attention model can use

\[
x_i
=
P_z z_i^K
+
P_t(\Delta t_i)
+
P_\mu\mu_i
+\cdots
\]

followed by

\[
h_{t-H:t}
=
\operatorname{Attention}_{\rm causal}(x_{t-H:t}).
\]

Use the last causal token \(h_t\) and project

\[
\boxed{
c_t=P_c h_t.
}
\]

This preserves causal semantics without requiring a special CLS token.

## 11. Three context heads

### 11.1 Residual head

\[
\boxed{
\hat r_{t+1}^{0}
=
R_\phi(c_t,z_t^K,\Delta t_t)
}
\]

Question: what will the nominal Koopman model fail to explain next?

### 11.2 Adequacy/change head

\[
\boxed{
\hat m_{t+1}
=
Q_\chi(c_t)
}
\]

Question: how inadequate is the nominal Koopman model likely to be?

A later classifier may estimate \(p_{{\rm change},t}\), but continuous mismatch prediction should be preferred before hard labels are introduced.

### 11.3 Operator-adaptation head

\[
\boxed{
\eta_t=G_\omega(c_t)
}
\]

Question: how should the current Koopman dynamics change?

## 12. Context-conditioned Koopman adaptation

Attention should not output an unrestricted \(d_K\times d_K\) matrix.

The preferred first route is low-rank modulation:

\[
\boxed{
A_t
=
A_0
+
U\,\operatorname{diag}(\eta_t)\,V^\top
}
\]

where

\[
U,V\in\mathbb R^{d_K\times r},
\qquad
r\ll d_K.
\]

Then

\[
\eta_t\in\mathbb R^r
\]

controls only a restricted family of generator variations.

Adaptive propagation:

\[
\boxed{
z_{t+1}^{A}
=
e^{A_t\Delta t_t}z_t^K.
}
\]

## 13. Alternative route: Mixture of Koopman operators

For more discrete regime switching, use \(A_1,\ldots,A_M\) and

\[
\alpha_t=\operatorname{softmax}(Wc_t),
\qquad
\sum_{m=1}^{M}\alpha_{t,m}=1.
\]

Then

\[
\boxed{
A_t
=
\sum_{m=1}^{M}
\alpha_{t,m}A_m.
}
\]

This is a secondary route for clearly multi-regime systems; low-rank modulation remains the primary first experiment.

## 14. Residual decomposition after operator adaptation

Nominal residual:

\[
r_{t+1}^{0}
=
z_{t+1}^{\rm true}
-
e^{A_0\Delta t_t}z_t^K.
\]

Operator-explainable part:

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

Remaining residual:

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

This is a central mathematical decomposition of the revised architecture.

## 15. Interpretation of the decomposition

\[
\boxed{
r^{op}
}
\]

is the residual explainable by adapting Koopman dynamics.

\[
\boxed{
r^{rem}
}
\]

is what remains unexplained even after adaptive operator selection/modulation.

This prevents the residual closure from being forced to compensate for an inappropriate fixed operator.

## 16. Operator-explained fraction

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

- \(\Gamma_{op}\approx1\): adaptation explains most nominal residual;
- \(\Gamma_{op}\approx0\): adaptation explains little;
- negative values: adaptation worsened the mismatch.

## 17. Optional remaining residual closure

Only if \(r^{rem}\) remains non-negligible, stable, and learnable should an explicit closure state be introduced:

\[
\boxed{
z_t^R
=
M_\rho(c_t,z_{t-H:t}^K).
}
\]

Then

\[
\boxed{
\Delta z_{t+1}^{R}
=
C_R(z_t^R).
}
\]

Final prediction:

\[
\boxed{
\hat z_{t+1}
=
e^{A_t\Delta t_t}z_t^K
+
\Delta z_{t+1}^{R}.
}
\]

The semantic role of \(z_R\) becomes

\[
\boxed{
z_R
=
\text{unresolved closure state remaining after adaptive Koopman dynamics}.
}
\]

## 18. PhysicsConstraint

Decode

\[
\hat U_{t+1}=D(\hat z_{t+1})
\]

and apply

\[
\boxed{
PhysicsConstraint(\hat U_{t+1}).
}
\]

The long-term architecture is

\[
\boxed{
\text{JEPA representation}
+
\text{dynamic-context Attention}
+
\text{adaptive Koopman}
+
\text{optional residual closure}
+
\text{PhysicsConstraint}.
}
\]

## 19. Roles of the main variables

| Object | Mathematical role |
|---|---|
| \(U_t\) | full physical state |
| \(z_t^K\) | current dynamical coordinate |
| JEPA | stabilizes predictive representation |
| \(A_0\) | nominal/persistent Koopman dynamics |
| \(r^0\) | nominal Koopman mismatch |
| \(m_t\) | Koopman inadequacy indicator |
| \(c_t\) | compact temporal dynamic context |
| \(\eta_t\) | operator-adaptation coordinates |
| \(A_t\) | context-conditioned Koopman generator |
| \(r^{op}\) | residual explained by operator adaptation |
| \(r^{rem}\) | residual remaining after adaptive Koopman |
| \(z_R\) | unresolved closure state |
| PhysicsConstraint | physical admissibility/consistency |

## 20. Main architectural changes

### Change 1 — Residual is no longer automatically identified with \(z_R\)

Old:

\[
r\rightarrow z_R\rightarrow Attention\rightarrow correction.
\]

New:

\[
\boxed{
r^0
\rightarrow
\text{learnability and Koopman-adequacy analysis}
}
\]

followed by context learning and operator adaptation.

### Change 2 — Attention still learns residual structure

\[
\boxed{
\mathcal H_t
\rightarrow
Attention
\rightarrow
c_t
\rightarrow
\hat r^0
}
\]

Residual remains a primary teacher signal, but residual prediction is no longer the sole purpose of Attention.

### Change 3 — Introduce \(c_t\)

Instead of

\[
Attention\rightarrow\Delta z,
\]

use

\[
\boxed{
Attention\rightarrow c_t
}
\]

then

\[
c_t
\rightarrow
(\hat r^0,\hat m,\eta_t).
\]

### Change 4 — Attention can feed back into Koopman dynamics

\[
\boxed{
A_t=\mathcal G(A_0,c_t).
}
\]

This allows the model to represent a changing local dynamical law.

## 21. Identifiability issue

If adaptive operator and residual correction are trained freely at the same time, both can explain the same mismatch.

Therefore

\[
r^0=r^{op}+r^{rem}
\]

is not uniquely identifiable under unconstrained joint training.

The architecture must therefore be developed in stages.

# 22. Revised Version Roadmap

## V0.7 — Residual Learnability & Koopman Adequacy

### Core question

\[
\boxed{
r_{t+1}^{0}
=
z_{t+1}^{\rm true}
-
e^{A_0\Delta t_t}z_t^K
}
\]

Does this residual contain stable, non-random, learnable structure?

### Keep the current V0.7 work

- residual extraction;
- residual statistics;
- zero closure baseline;
- linear predictor;
- tiny MLP probe;
- history diagnostic;
- Mori–Zwanzig-inspired interpretation;
- closed-loop probe.

### New emphasis

Report

\[
\boxed{
\text{RESIDUAL LEARNABILITY}
=
\text{STRONG / MODERATE / WEAK / NONE}
}
\]

using held-out metrics such as \(R^2\), NRMSE, and MSE.

### Koopman adequacy

Analyze

\[
m_t=\|r_{t+1}^{0}\|
\]

and robust normalized variants.

Check whether high mismatch is structured in time/state space and whether it identifies transient or poorly modeled regions.

### Change-signal potential

Analyze

\[
m_t
\]

and

\[
\Delta m_t=m_t-m_{t-1}
\]

as potential future supervision for context/change detection.

### Memory classification

Existing Markovian/short-memory/long-memory diagnostics may remain, but are secondary because the classification is problem- and representation-dependent.

### V0.7 final gates

1. Is \(r^0\) meaningfully learnable?
2. Can causal history predict \(r^0\) or its magnitude?
3. Does residual provide a useful Koopman inadequacy signal?

## V0.8 — Residual-Supervised Temporal Context Attention

### Core question

\[
\boxed{
\text{Can causal temporal history produce a compact }c_t
\text{ that predicts nominal Koopman mismatch?}
}
\]

### Context

\[
\boxed{
c_t
=
\mathcal A_\psi(z_{t-H:t}^K,\Delta t)
}
\]

with later extension to \(a,\mu,o^{exo}\).

### Residual head

\[
\boxed{
\hat r_{t+1}^{0}
=
R_\phi(c_t,z_t^K,\Delta t_t)
}
\]

with

\[
\boxed{
L_R
=
\|\hat r_{t+1}^{0}-r_{t+1}^{0}\|^2.
}
\]

### Adequacy head

\[
\boxed{
\hat m_{t+1}=Q_\chi(c_t)
}
\]

with

\[
L_Q
=
|\hat m_{t+1}-m_{t+1}|^2.
\]

### Critical restriction

Keep

\[
A=A_0.
\]

V0.8 must not yet modify the Koopman operator.

### Main comparisons

- history MLP;
- causal Attention;
- parameter-matched controls;
- shuffled-history controls.

### V0.8 output

A validated dynamic-context representation \(c_t\), plus residual and adequacy probe heads.

## V0.9 — Context-Conditioned Adaptive Koopman

### Core question

\[
\boxed{
\text{Can }c_t
\text{ adapt Koopman dynamics and explain nominal residual?}
}
\]

### Critical training rule

Disable direct residual correction during the primary experiment.

Only allow

\[
c_t\rightarrow A_t.
\]

### Primary route: low-rank modulation

\[
\eta_t=G_\omega(c_t)
\]

\[
\boxed{
A_t
=
A_0
+
U\operatorname{diag}(\eta_t)V^\top.
}
\]

### Alternative: mixture of Koopman generators

\[
\alpha_t=\operatorname{softmax}(Wc_t)
\]

\[
\boxed{
A_t=\sum_m\alpha_{t,m}A_m.
}
\]

### Primary loss

\[
\boxed{
L_A
=
\left\|
z_{t+1}^{\rm true}
-
e^{A_t\Delta t_t}z_t^K
\right\|^2.
}
\]

Include multi-step rollout.

### Regularization

For example

\[
L_{\Delta A}
=
\|\Delta A_t\|_F^2.
\]

Temporal smoothness may be tested carefully but must not suppress genuine abrupt switching.

### Auxiliary heads

V0.8 residual/adequacy heads may remain as auxiliary losses, but predicted residual must not enter the primary state prediction.

## V1.0 — Adaptive Koopman Physical World Model Baseline

Architecture:

\[
\boxed{
PhysicsConstraint
+
JEPA\ z_K
+
Temporal\ Context
+
Adaptive\ Koopman
}
\]

Recompute

\[
\boxed{
r_{t+1}^{rem}
=
z_{t+1}^{\rm true}
-
e^{A_t\Delta t_t}z_t^K.
}
\]

Compare against \(r^0\) using

\[
\boxed{
\Gamma_{op}
=
1-
\frac{
\mathbb E\|r^{rem}\|^2
}{
\mathbb E\|r^0\|^2+\epsilon
}.
}
\]

Core question:

> How much of the original residual was caused by an inappropriate fixed Koopman operator?

If \(r^{rem}\approx0\), an explicit \(z_R\) may not be necessary.

## V1.1 — Remaining Residual Closure

Proceed only if \(r^{rem}\) remains non-negligible and learnable.

\[
\boxed{
z_t^R=M_\rho(c_t,z_{t-H:t}^K)
}
\]

\[
\boxed{
\Delta z_{t+1}^{R}=C_R(z_t^R)
}
\]

and

\[
\boxed{
\hat z_{t+1}
=
e^{A_t\Delta t_t}z_t^K
+
\Delta z_{t+1}^{R}.
}
\]

Here

\[
\boxed{
z_R
=
\text{unresolved closure after adaptive Koopman dynamics}.
}
\]

## V1.2 — Controlled Joint Fine-Tuning

Only after all earlier modules are independently validated should controlled joint training be attempted.

Possible trainable modules:

\[
E_\theta,
\quad
\mathcal A_\psi,
\quad
A_t,
\quad
z_R,
\quad
D.
\]

Strict regularization and matched ablations are required so that Attention or residual closure does not absorb all dynamics and destroy Koopman interpretability.

# 23. Roadmap summary

\[
\boxed{
V0.7
=
Residual\ Learnability
+
Koopman\ Adequacy
}
\]

\[
\Downarrow
\]

\[
\boxed{
V0.8
=
Residual\text{-}Supervised\ Attention
+
Dynamic\ Context\ c_t
}
\]

\[
\Downarrow
\]

\[
\boxed{
V0.9
=
Context\text{-}Conditioned\ Adaptive\ Koopman
}
\]

\[
\Downarrow
\]

\[
\boxed{
V1.0
=
Adaptive\ Koopman\ World\ Model
+
Residual\ Reassessment
}
\]

\[
\Downarrow
\]

if necessary,

\[
\boxed{
V1.1
=
Remaining\ Residual\ Closure
+
z_R
}
\]

\[
\Downarrow
\]

\[
\boxed{
V1.2
=
Controlled\ Joint\ Fine\text{-}Tuning
}
\]

# 24. Final mathematical core

\[
\boxed{
z_t^K=E_\theta(U_t)
}
\]

\[
\boxed{
r_{t+1}^{0}
=
z_{t+1}^{\rm true}
-
e^{A_0\Delta t_t}z_t^K
}
\]

\[
\boxed{
c_t
=
Attention(z_{t-H:t}^K,\ldots)
}
\]

with

\[
\boxed{
c_t
\rightarrow
(\hat r^0,\hat m,\eta_t)
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

and

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

followed, only if necessary, by

\[
\boxed{
\hat z_{t+1}
=
e^{A_t\Delta t_t}z_t
+
C_R(z_t^R).
}
\]

## 25. Final interpretation

The revised architecture is

\[
\boxed{
\text{JEPA Representation}
+
\text{Residual Diagnosis}
+
\text{Attention Dynamic Context}
+
\text{Adaptive Koopman}
+
\text{Optional Residual Closure}
+
\text{PhysicsConstraint}
}
\]

The key conceptual improvement is that residual is no longer treated as a single monolithic correction target.

Instead,

\[
\boxed{
\text{Residual first tells us why and where nominal Koopman is inadequate.}
}
\]

Attention learns the temporal structure of this inadequacy, the learned context adapts the operator itself, and only the genuinely unexplained remainder is delegated to a final closure module.
