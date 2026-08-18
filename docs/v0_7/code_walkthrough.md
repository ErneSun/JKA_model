# Code walkthrough

- `src/jka_model/residual/cache.py`: builds detached online residuals and verifies cache fingerprints.
- `src/jka_model/residual/dataset.py`: fixed-history, split-safe, dt-aligned windows and shuffled control.
- `src/jka_model/residual/closures.py`: five closure variants and frozen-backbone composite.
- `src/jka_model/residual/diagnostics.py`: residual statistics, prediction metrics, evidence labels.
- `src/jka_model/residual/assessment.py`: validation-first significance/predictability/history gates and R0–R3 routing.
- `src/jka_model/residual/rollout.py`: teacher-free corrected latent rollout.
- `src/jka_model/residual/memory.py`: exact 144-record provenance validation, secondary memory aggregation, primary assessment integration, plots, and reports.
- `src/jka_model/residual/checkpoint.py`: standalone schema-7 checkpoint validation.
- `src/train/prepare_v0_7.py`: canonical cache preparation.
- `src/train/train_v0_7.py`: closure-only trainer.
- `src/eval/evaluate_v0_7.py`: teacher-forced and physical closed-loop evaluation.
- `gpu_validation/v0_7/scripts/gpu_validate_all.py`: one-command multi-seed orchestration and report generation.
- `scripts/compare_residual_memory_v0_7.py`: comparison-only entry point; it never retrains.
- `scripts/explain_v0_7.py`: operational H=1/H=2/H=4 explanation using completed results.
- `src/jka_model/utils/versioned_runs.py`: collision-safe `<id>-rN` session allocation.

GPU scripts are thin wrappers around the canonical train/evaluate functions; there is no second trainer.
