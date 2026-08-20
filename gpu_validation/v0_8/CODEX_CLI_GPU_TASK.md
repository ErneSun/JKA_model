# Remote GPU task

No Codex agent is needed on the server. In a clean checkout:

```bash
source .venv/bin/activate && python gpu_validation/v0_8/scripts/gpu_validate_all.py --validation-id v08-final-$(date -u +%Y%m%dT%H%M%SZ) --seeds 47 53 59
```

Commit/push the compact `gpu_validation/v0_8/results/<resolved-id>/` report. Large datasets,
checkpoints and raw run logs remain under `runs/v0_8/<resolved-id>/` and should not be committed.

