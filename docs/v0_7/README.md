# V0.7 — Residual Structure Assessment and Koopman Adequacy

V0.7 identifies the residual left by the frozen JEPA–Koopman backbone, determines whether that residual is significant and predictably learnable, and tests whether causal history provides information beyond the current resolved state in order to route the next model stage.

Its primary chain is **Residual Significance → Residual Learnability → Conditional History Gain → R0/R1/R2/R3 Route**. Memory class and Mori–Zwanzig-inspired interpretation remain secondary diagnostics; V0.7 does not force a memory result to justify a preselected roadmap.

The implementation freezes the complete V0.6 backbone and trains only a small direct-Δz closure. It sweeps `H=[1,2,4,8,16]`; `H=1` is exactly the operational Markovian baseline. Every H compares an ordered-history model with a parameter-matched instantaneous control, while H>1 also uses a shuffled-history control. Each comparison is repeated with closure initialization seeds `[101,211,307]`, independently of the three backbone/data seeds. Physics is evaluated but is not part of the closure loss.

The protocol is problem-agnostic: every learned correction starts at zero, residual dimensions are scaled only by training-split RMS, validation locks the model/H/preliminary route, and test only confirms that locked decision. Scientific conclusions require consistency across closure initializations and then backbone/data seeds. The final route is `R0` (negligible), `R1` (significant but unlearnable), `R2` (learnable without stable history gain), `R3` (learnable with stable history gain), or `INCONCLUSIVE`.

Status at construction time: local implementation and CPU integration tests pass; formal multi-seed RTX 5080 evidence remains pending. A negative memory result is a valid V0.7 outcome and does not authorize a larger V0.8 model.

The reconciliation against the latest route prompt is recorded in
[`revised_addendum_v2_audit.md`](revised_addendum_v2_audit.md).

Canonical remote command:

```bash
python gpu_validation/v0_7/scripts/gpu_validate_all.py --validation-id v07-final-$(date -u +%Y%m%dT%H%M%SZ) --v0-6-root runs/v0_6/gpu --seeds 47 53 59
```

Every invocation creates `runs/v0_7/<id>/`; a reused ID becomes `<id>-r1`, then `-r2`, and so on. The compact bundle includes `completion.json`, `summary.json`, `evaluation/residual_structure_assessment.json`, `evaluation/history_sweep.csv`, the secondary `memory_classification.json`, seven diagnostic plots, the residual decision report, and the V0.8 route recommendation under `gpu_validation/v0_7/results/<resolved-id>/`.
