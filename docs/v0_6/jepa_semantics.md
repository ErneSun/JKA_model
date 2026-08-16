# JEPA semantic contract

Two future embeddings deliberately coexist:

- `z_k_online_future = E_theta(U_future)` is used by every inherited V0.5
  Koopman/generator/reconstruction term.
- `z_k_jepa_target = stopgrad(E_bar(U_future))` is used only by JEPA.

\[
L_{J,1}=\|\hat z_1-z^J_1\|_2^2,
\qquad
L_{J,m}=\frac1{H-1}\sum_{k=2}^{H}\|\hat z_k-z^J_k\|_2^2.
\]

Every prediction is closed-loop through matrix exponentials; no future latent is fed
back. Setting both JEPA weights to zero returns the V0.5 total exactly and skips the
target forward. A mandatory test perturbs `E_bar`: JEPA changes while V0.5 remains
identical. Perturbing `E_theta` changes the Koopman target.
