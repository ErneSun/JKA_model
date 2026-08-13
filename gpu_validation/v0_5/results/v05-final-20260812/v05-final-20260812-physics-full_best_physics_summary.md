# V0.5 GPU result summary

- technical status: **PASS**
- scientific acceptance: **PENDING_REVIEW**
- run: `v05-final-20260812-physics-full`
- commit: `f1ecfc0d60fc88d9cf994b3bb26c400b9ee4bf21`; dirty: `False`
- GPU / precision: `NVIDIA GeForce RTX 5080` / `fp32`
- checkpoint / selection epoch: `best_physics` / 2
- selection rule: minimum validation mass plus operator penalty
- long model / persistence RMSE: 0.626957 / 1.03311
- frequency / decay relative error: 0.972791 / 1.26941
- frequency hard gate: **FAIL** (threshold 0.05)
- long mass drift / operator: 0.654437 / 5.48601
- peak VRAM bytes / max samples s^-1: 2888410624 / 146.985
