# V0.5 GPU validation package

This directory validates the same canonical implementation used on CPU. It contains no
model, loss, trainer, checkpoint, or evaluator implementation.

Run only from a clean checkout of the reviewed commit. The GPU package wraps the canonical
trainer/evaluator and contains no alternate model implementation.

## One-command complete validation (recommended)

Activate an existing CUDA-enabled Python environment, verify that `git status --short` is
empty, and run:

```bash
python gpu_validation/v0_5/scripts/gpu_validate_all.py
```

The command runs the local regression gates, GPU preflight/parity, FP32 and capability-selected
AMP smoke, the full physics run, an exact epoch-75 resume, the full no-physics ablation,
`best_forecast`, post-warmup forecast, `best_physics`, post-warmup physics, and `last`
evaluation for both full runs, and the bounded profiler. The final scientific gate uses the
post-warmup forecast checkpoints so an early checkpoint cannot bypass the physics objective.
It retains only the numbered epoch checkpoint required for resume instead of all 150 numbered
checkpoints. Large checkpoints and traces remain under ignored `runs/`; compact JSON/Markdown
evidence is exported to `gpu_validation/v0_5/results/<validation-id>/`.

Every subprocess has a persistent log and state entry. If a step fails, the final console output
prints an exact continuation command containing `--validation-id`; rerunning it skips all prior
PASS steps and preserves the failed run rather than deleting it. A completed command exits zero
when the workflow executed correctly even when `scientific_status=FAIL`; read
`final_validation.md` for the acceptance decision.

Do not use `--skip-local-gates` unless pytest, ruff, mypy, and diff-check already passed on the
exact same commit.

## One-command incremental scientific revalidation

After the original CUDA parity, profiler, and exact-resume gates have passed, model/loss changes
do not require repeating those environment gates. From a clean checkout, run:

```bash
python gpu_validation/v0_5/scripts/gpu_validate_science.py --validation-id v05-science-final
```

This runs the affected FP32/AMP model smoke and paired physics/no-physics training for seeds
47, 53, and 59, evaluates only `best_forecast_post_warmup`, and exports mean/median/std plus
per-seed hard gates. A failed/interrupted session prints the same command for resumable
continuation and skips steps that already passed.

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
