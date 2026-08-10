# Code walkthrough

1. `data/advection_diffusion_2d.py` evaluates the analytic Fourier solution in float64.
2. `problems/advection_diffusion_2d.py` binds the dataset/spec/constraints.
3. `physics/operators.py` provides endpoint-free periodic derivatives and area integration.
4. `models/field_koopman_autoencoder.py` defines the circular CNN, continuous core, decoder.
5. `losses/field_koopman.py` joins latent, reconstruction, variance, mass, and operator terms.
6. `src/train/train_v0_5.py` is the only loop and persistence owner.
7. `src/eval/evaluate_v0_5.py` performs held-out multi-horizon evaluation.
8. `scripts/` and `gpu_validation/v0_5/scripts/` are thin command-line wrappers.

Important implementation mismatch resolved: the older 1-D operators assume a duplicated
endpoint, while V0.5 explicitly requires `endpoint=False`. Separate 2-D operators were
added instead of changing the legacy functions, preserving V0.2 behavior.
