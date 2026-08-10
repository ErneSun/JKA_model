# GPU test plan

GPU is required to validate the actual CUDA kernels, AMP behavior, memory use, throughput,
and server resume workflow; CPU already validates contracts, analytic numerics, gradient
connectivity, deterministic training, evaluation integration, and artifact structure.

| Gate | Config | Pass criterion |
|---|---|---|
| Preflight/parity | `gpu_smoke.yaml` | CUDA/cuDNN/device/configs present; encoder, Koopman step, decoder, mass, and operator FP32 parity within recorded tolerances |
| FP32 + AMP smoke | `gpu_smoke.yaml` | finite training/evaluation and nonzero finite total plus isolated-physics gradients for encoder/decoder/A |
| Full baseline | `gpu_full.yaml` | complete run, exact intermediate-checkpoint resume, finite held-out evaluation |
| Ablation | `gpu_full_no_physics.yaml` | same protocol with physics disabled; compare against full baseline |
| Profile | `gpu_smoke.yaml` | exactly three diagnostic steps; retain CPU/CUDA time, shapes, and memory trace |

FP32 is the scientific baseline. AMP selects BF16 when supported and FP16 otherwise. The
continuous generator, `matrix_exp`, inverse normalization, finite differences, quadrature,
and physics reductions remain FP32 precision islands.

On failure, preserve stdout/stderr and the run directory, record commit/config/environment,
mark the failing checklist item, and stop before changing architecture or thresholds. Fixes
must return through the normal local review and exact-commit server workflow.
