# Remote GPU task

Run the single command in `README.md` from the repository root with `.venv` active.

Hard restrictions:

- DO NOT unfreeze the V0.6 backbone.
- DO NOT retrain JEPA.
- DO NOT substitute the EMA target encoder for the online encoder.
- DO NOT reuse a residual cache with a different checkpoint hash.
- DO NOT call teacher-forced metrics closed-loop forecasting.
- DO NOT classify memory from one H or ACF alone; complete all configured H and controls.
- DO NOT retrain from the comparison-only script.
- DO NOT claim an exact Mori–Zwanzig kernel.

Return the generated `gpu_validation/v0_7/results/<resolved-id>/` directory. Large checkpoints and raw `runs/` artifacts may stay on the server.
