# Server task

Run from repository root after `git pull` and installation:

```bash
.venv/bin/python gpu_validation/v0_6/scripts/gpu_validate_all.py --validation-id v06-final-$(date -u +%Y%m%dT%H%M%SZ) --v0-5-root runs/v0_5/gpu --seeds 47 53 59
```

Constraints:

- DO NOT redesign JEPA.
- DO NOT add Attention, residual state, control/action modules or a new predictor.
- DO NOT replace the continuous Koopman matrix exponential.
- DO NOT alter split/normalizer/model capacity between JEPA and control.
- DO NOT silently change hyperparameters.
- DO NOT patch core code on the server without a new reviewed Git commit.
- Preserve failures, logs and checkpoints.

Return `runs/v0_6/gpu/validation_sessions/<id>/artifacts/summary.json`, compact
`gpu_validation/v0_6/results/<id>/`, logs, resolved configs and any failed-run reports.
