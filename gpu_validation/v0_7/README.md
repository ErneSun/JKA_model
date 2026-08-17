# V0.7 GPU validation

This package validates the frozen-V0.6 residual/closure experiment on CUDA. The canonical workflow is non-silent, trains closure heads only, sweeps `H=[1,2,4,8,16]` with matched and shuffled controls over three independent closure initialization seeds, creates collision-safe run IDs, and writes a compact result bundle only after comparison completes.

```bash
python gpu_validation/v0_7/scripts/gpu_validate_all.py --validation-id v07-final-$(date -u +%Y%m%dT%H%M%SZ) --v0-6-root runs/v0_6/gpu --seeds 47 53 59
```

If the requested ID already exists, the workflow automatically uses `-r1`, `-r2`, etc. Raw artifacts live under `runs/v0_7/<resolved-id>/`; review files live under `gpu_validation/v0_7/results/<resolved-id>/`. A successful bundle contains `completion.json`; a failed/incomplete bundle contains `failure.json`. Formal GPU epochs remain compact in the terminal while full epoch metrics stay in logs.
