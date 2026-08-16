# V0.5 GPU result summary

- technical status: **PASS**
- scientific acceptance: **PENDING_REVIEW**
- run: `v05-final-20260814-physics-seed47`
- commit: `976de084c540e28a21411614ed0c854291ed491c`; dirty: `False`
- GPU / precision: `NVIDIA GeForce RTX 5080` / `fp32`
- checkpoint / selection epoch: `best_forecast_post_warmup` / 134
- selection rule: minimum validation forecast MSE after physics warmup
- long model / persistence RMSE: 0.0151236 / 1.03311
- frequency / decay relative error: 0.00252759 / 0.165942
- frequency hard gate: **PASS** (threshold 0.05)
- decay hard gate: **PASS** (threshold 0.2)
- stability hard gate: **PASS**; spectral abscissa -0.000116175
- beats persistence by horizon: {'short': True, 'medium': True, 'long': True}
- mass hard gates by horizon: {'short': True, 'medium': True, 'long': True}
- operator hard gates by horizon: {'short': True, 'medium': True, 'long': True}
- long mass drift / operator: 0.000775816 / 3.22555e-07
- peak VRAM bytes / max samples s^-1: 2907331072 / 158.802
