# Testing and validation

## Mandatory CPU correctness

CPU tests cover exact data shape/alignment, analytic mass, `Dx`, `Dy`, `Dxx`, and `Dyy`,
observed second-order grid convergence, circular padding, rollout shape, finite operator
residual, isolated physics-only gradients for encoder/generator/decoder, training records,
exact resume, and held-out evaluation.

Mandatory local commands:

```bash
pytest
python scripts/smoke_v0_1.py
python scripts/smoke_v0_2.py
python scripts/smoke_v0_3.py
python scripts/smoke_v0_4.py
python scripts/smoke_v0_5.py
python scripts/explain_v0_5.py
python scripts/train_v0_5.py --config configs/v0_5/advection_diffusion_2d_cpu_tiny_train.yaml --device cpu
```

## Independent GPU validation

The remote sequence is documented under `gpu_validation/v0_5/`. It separately checks
CUDA/cuDNN/device capability, component-level CPU/GPU parity, FP32 and capability-selected
AMP, isolated physics gradients, bounded profiling, full training, and exact resume.

## Scientific acceptance

Scientific review compares held-out short/medium/long field rollouts against persistence,
reviews spectrum/frequency/decay and latent non-collapse, compares physics-on against the
required no-physics ablation, and inspects mass/operator diagnostics. Local CPU success
never substitutes for GPU or scientific acceptance.
