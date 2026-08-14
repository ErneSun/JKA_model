# V0.5 GPU result summary

- technical status: **PASS**
- scientific acceptance: **PENDING_REVIEW**
- run: `v05-final-20260813-no_physics-seed47`
- commit: `7695a9366ecc55a35df380666a8a4b729408185e`; dirty: `False`
- GPU / precision: `NVIDIA GeForce RTX 5080` / `fp32`
- checkpoint / selection epoch: `best_forecast_post_warmup` / 14
- selection rule: minimum validation forecast MSE after physics warmup
- long model / persistence RMSE: 0.0111699 / 1.03311
- frequency / decay relative error: 0.00199612 / 0.130913
- frequency hard gate: **PASS** (threshold 0.05)
- decay hard gate: **PASS** (threshold 0.2)
- stability hard gate: **PASS**; spectral abscissa -0.00314855
- beats persistence by horizon: {'short': True, 'medium': True, 'long': True}
- long mass drift / operator: 0.00243658 / 0.000136031
- peak VRAM bytes / max samples s^-1: 2907331072 / 147.76
