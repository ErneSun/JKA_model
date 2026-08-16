# V0.7 — Residual learnability and memory characterization

V0.7 follows one primary chain: **Residual Identification → Residual Learnability → Closed-loop Utility → Memory Characterization**. After the validated V0.6 online encoder and continuous Koopman generator are frozen, it asks whether their one-step latent residual is measurable, predictable, useful in rollout, and genuinely benefits from ordered finite history.

The implementation freezes the complete V0.6 backbone and trains only a small direct-Δz closure. It sweeps `H=[1,2,4,8,16]`; `H=1` is exactly the operational Markovian baseline. Every H compares an ordered-history model with a parameter-matched instantaneous control, while H>1 also uses a shuffled-history control. Physics is evaluated but is not part of the closure loss.

Status at construction time: local implementation and CPU integration tests pass; formal multi-seed RTX 5080 evidence remains pending. A negative memory result is a valid V0.7 outcome and does not authorize a larger V0.8 model.

Canonical remote command:

```bash
python gpu_validation/v0_7/scripts/gpu_validate_all.py --validation-id v07-final-$(date -u +%Y%m%dT%H%M%SZ) --v0-6-root runs/v0_6/gpu --seeds 47 53 59
```

Every invocation creates `runs/v0_7/<id>/`; a reused ID becomes `<id>-r1`, then `-r2`, and so on. The compact bundle includes `evaluation/history_sweep.csv`, `evaluation/memory_classification.json`, plots, the residual decision report, and the V0.8 route recommendation under `gpu_validation/v0_7/results/<resolved-id>/`.
