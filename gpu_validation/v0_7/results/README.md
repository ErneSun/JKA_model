# V0.7 compact results

Each successfully resolved validation ID receives `completion.json`, `summary.json`, `report.md`, `evaluation/history_sweep.csv`, `evaluation/memory_classification.json`, diagnostic plots, and both decision reports. `completion.json` is the machine-readable proof that the whole workflow passed. An incomplete session receives `failure.json` instead. These small review artifacts are intended for Git. Raw caches, logs, and checkpoints remain under ignored `runs/v0_7/<id>/` on the GPU server.
