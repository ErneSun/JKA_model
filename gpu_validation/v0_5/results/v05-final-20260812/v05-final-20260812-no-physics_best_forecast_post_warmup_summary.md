# V0.5 GPU result summary

- technical status: **PASS**
- scientific acceptance: **PENDING_REVIEW**
- run: `v05-final-20260812-no-physics`
- commit: `f1ecfc0d60fc88d9cf994b3bb26c400b9ee4bf21`; dirty: `False`
- GPU / precision: `NVIDIA GeForce RTX 5080` / `fp32`
- checkpoint / selection epoch: `best_forecast_post_warmup` / 54
- selection rule: minimum validation forecast MSE after physics warmup
- long model / persistence RMSE: 0.636513 / 1.03311
- frequency / decay relative error: 0.950872 / 1.22968
- frequency hard gate: **FAIL** (threshold 0.05)
- long mass drift / operator: 2.69477 / 5.7889
- peak VRAM bytes / max samples s^-1: 2888410624 / 148.242
