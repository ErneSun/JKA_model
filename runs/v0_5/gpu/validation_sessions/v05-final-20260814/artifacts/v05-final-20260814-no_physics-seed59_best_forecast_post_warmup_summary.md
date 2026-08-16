# V0.5 GPU result summary

- technical status: **PASS**
- scientific acceptance: **PENDING_REVIEW**
- run: `v05-final-20260814-no_physics-seed59`
- commit: `976de084c540e28a21411614ed0c854291ed491c`; dirty: `False`
- GPU / precision: `NVIDIA GeForce RTX 5080` / `fp32`
- checkpoint / selection epoch: `best_forecast_post_warmup` / 128
- selection rule: minimum validation forecast MSE after physics warmup
- long model / persistence RMSE: 0.0134034 / 0.916047
- frequency / decay relative error: 0.00245667 / 0.156715
- frequency hard gate: **PASS** (threshold 0.05)
- decay hard gate: **PASS** (threshold 0.2)
- stability hard gate: **PASS**; spectral abscissa -0.000131582
- beats persistence by horizon: {'short': True, 'medium': True, 'long': True}
- mass hard gates by horizon: {'short': True, 'medium': True, 'long': True}
- operator hard gates by horizon: {'short': True, 'medium': True, 'long': True}
- long mass drift / operator: 0.000345039 / 2.11081e-07
- peak VRAM bytes / max samples s^-1: 2907331072 / 156.208
