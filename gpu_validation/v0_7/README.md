# V0.7 GPU validation

This package performs the frozen-V0.6 Residual Structure Assessment on CUDA. It trains probe closures only, preserves the 144-record `3 backbone/data × 3 closure-init` matrix, selects `S_R/P_R/G_H` decisions on validation, confirms the locked result on test, and emits an R0/R1/R2/R3/INCONCLUSIVE route. Memory class is secondary.

```bash
python gpu_validation/v0_7/scripts/gpu_validate_all.py --validation-id v07-final-$(date -u +%Y%m%dT%H%M%SZ) --v0-6-root runs/v0_6/gpu --seeds 47 53 59
```

If the requested ID already exists, the workflow automatically uses `-r1`, `-r2`, etc. Raw artifacts live under `runs/v0_7/<resolved-id>/`; review files live under `gpu_validation/v0_7/results/<resolved-id>/`. Success requires an exact 144 records, provenance pass, all reports, and `completion.json`; failure records stage/run/count/checkpoint context in `failure.json`. Formal GPU epochs remain compact while full metrics stay in logs.
