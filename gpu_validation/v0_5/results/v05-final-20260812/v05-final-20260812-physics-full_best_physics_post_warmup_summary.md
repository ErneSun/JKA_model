# V0.5 GPU result summary

- technical status: **PASS**
- scientific acceptance: **PENDING_REVIEW**
- run: `v05-final-20260812-physics-full`
- commit: `f1ecfc0d60fc88d9cf994b3bb26c400b9ee4bf21`; dirty: `False`
- GPU / precision: `NVIDIA GeForce RTX 5080` / `fp32`
- checkpoint / selection epoch: `best_physics_post_warmup` / 54
- selection rule: minimum validation mass plus operator penalty after physics warmup
- long model / persistence RMSE: 0.645619 / 1.03311
- frequency / decay relative error: 0.950257 / 1.25552
- frequency hard gate: **FAIL** (threshold 0.05)
- long mass drift / operator: 1.38263 / 6.4549
- peak VRAM bytes / max samples s^-1: 2888410624 / 146.985
