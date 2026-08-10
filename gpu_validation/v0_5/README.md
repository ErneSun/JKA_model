# V0.5 GPU validation package

This directory validates the same canonical implementation used on CPU. It contains no
model, loss, trainer, checkpoint, or evaluator implementation.

Run only from a clean checkout of the reviewed commit. The GPU package wraps the canonical
trainer/evaluator and contains no alternate model implementation.

## 1. Update and verify revision

```bash
git pull --ff-only
git rev-parse HEAD
git branch --show-current
git status --short
```

## 2. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## 3. Preflight and component parity

```bash
python gpu_validation/v0_5/scripts/gpu_preflight.py
```

## 4. FP32 and AMP smoke

```bash
python gpu_validation/v0_5/scripts/gpu_smoke.py
```

## 5. Full training and resume

```bash
python gpu_validation/v0_5/scripts/gpu_train.py --config gpu_validation/v0_5/configs/gpu_full.yaml
python gpu_validation/v0_5/scripts/gpu_train.py --config gpu_validation/v0_5/configs/gpu_full.yaml --resume-from runs/v0_5/gpu/<run-id>/checkpoints/epoch_XXXX.pt
python gpu_validation/v0_5/scripts/gpu_resume_check.py --uninterrupted-run runs/v0_5/gpu/<run-id> --resumed-run runs/v0_5/gpu/<resumed-run-id>
```

## 6. Required no-physics ablation

```bash
python gpu_validation/v0_5/scripts/gpu_train.py --config gpu_validation/v0_5/configs/gpu_full_no_physics.yaml
```

## 7. Held-out evaluation and immutable summary

```bash
python gpu_validation/v0_5/scripts/gpu_evaluate.py --run-dir runs/v0_5/gpu/<run-id>
python gpu_validation/v0_5/scripts/gpu_evaluate.py --run-dir runs/v0_5/gpu/<ablation-run-id>
```

## 8. Inspect and profile

```bash
python scripts/inspect_v0_5_run.py --run-dir runs/v0_5/gpu/<run-id>
python gpu_validation/v0_5/scripts/gpu_profile.py
```

## 9. Record the result

Keep the generated `<run-id>_summary.md` and `<run-id>_metrics.json` files in `results/`, link both run
directories, attach profiler output, and update the checklist. A technical PASS means the
commands and finite numerical gates passed; scientific PASS additionally requires reviewed
full-vs-ablation, persistence, long-rollout, spectrum, mass, and operator evidence.

Do not label GPU or scientific acceptance as PASS until all gates in
`gpu_validation_checklist.md` are backed by artifacts under `results/`.
