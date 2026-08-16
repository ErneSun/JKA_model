# V0.6 GPU validation

This package calls canonical `train_v0_6()` and `evaluate_v0_6()`; it contains no
copied model or loss implementation. Raw runs/checkpoints stay under ignored `runs/`.
Compact summaries are written to `gpu_validation/v0_6/results/<validation-id>/`.

Prerequisites: the validated V0.5 seed-47/53/59 runs and checkpoints must exist below
`runs/v0_5/gpu/`, and the current commit must be installed in `.venv`.

Fresh one-command validation:

```bash
.venv/bin/python gpu_validation/v0_6/scripts/gpu_validate_all.py --validation-id v06-final-$(date -u +%Y%m%dT%H%M%SZ) --v0-5-root runs/v0_5/gpu --seeds 47 53 59
```

It runs the test suite, CUDA preflight, FP32 and AMP smoke, then matched no-JEPA/JEPA
full training for each seed from the same V0.5 checkpoint. Use `--skip-pytest` or
`--skip-smoke` only when that exact commit has already passed those steps.

The workflow is intentionally non-silent. Every validation step reports
`START/PASS/FAIL`, and subprocess output is shown live while also being preserved under
the session `logs/` directory. Formal GPU training reports its start and final summary
without printing every epoch; the complete epoch metrics remain in `epoch_metrics.csv`
and `step_metrics.jsonl`. CPU smoke/development training keeps per-epoch terminal output.

The command exits nonzero and preserves all artifacts when any gate fails. Do not resume
a failed scientific comparison as a fresh run; assign a new validation id after an
audited code/config change.
