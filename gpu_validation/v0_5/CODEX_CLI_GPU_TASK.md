# Codex CLI GPU task

Validate V0.5 at the exact reviewed Git commit; record commit, branch, and dirty state before
running. Read `gpu_validation/v0_5/{README.md,GPU_TEST_PLAN.md,gpu_validation_checklist.md}`,
`docs/v0_5/{architecture.md,physics.md,evaluation.md,status.md}`, and the resolved configs.

Run, in order: `gpu_preflight.py`, `gpu_smoke.py`, full FP32 `gpu_train.py`, an actual resume
from an intermediate full-run checkpoint, `gpu_resume_check.py`, the no-physics full ablation, `gpu_evaluate.py`
for both full runs, `inspect_v0_5_run.py`, and the bounded `gpu_profile.py`. Use only
`gpu_smoke.yaml`, `gpu_full.yaml`, and `gpu_full_no_physics.yaml`; BF16 is preferred when
supported, otherwise FP16. Never substitute CPU output for GPU evidence.

Record run IDs, config hashes, checkpoint selection, learned/true frequency and decay,
short/medium/long model and persistence errors, latent non-collapse, mass/operator metrics,
epoch time, samples/s, peak VRAM, and profiler paths. Compare resumed FP32 final weights to
the uninterrupted deterministic run.

Forbidden: architecture/loss/data changes, tolerance relaxation, silent retries, deletion of
failed artifacts, alternate trainer/evaluator implementations, or any V0.6 work. Stop on a
missing config, dirty/unexpected revision, unavailable CUDA, non-finite value, failed parity,
missing physics-only gradient, resume mismatch, or incomplete artifact.

Final report format: environment and revision; commands; run/artifact table; FP32-vs-AMP;
resume equality; physics-vs-ablation; scientific metrics; profiler summary; checklist A-R;
technical PASS/FAIL; scientific PASS/FAIL or PENDING with explicit reason. Update
`docs/v0_5/status.md` only when every claimed gate has retained evidence.
