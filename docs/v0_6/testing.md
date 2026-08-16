# Testing record

CPU coverage includes hard sync, frozen/eval target, optimizer exclusion, graph-free
target, EMA formula/order/count, online-vs-target semantics, JEPA gradient ownership,
closed-loop multi-step prediction, near-identity, no-JEPA exact control, V0.5
initialization, and V0.6 target/EMA checkpoint restoration.

On 2026-08-16 the full suite passed: `170 passed`. End-to-end CPU smoke passed the
V0.5 checkpoint → hard sync → loss/physics → backward → optimizer → EMA → checkpoint
→ reload → online-only evaluation path. Its two-epoch bootstrap is a software test,
not collapse evidence; scientific gates remain pending GPU.

```bash
PYTHONPATH=src:. pytest -q
PYTHONPATH=src:. python3 scripts/smoke_v0_6.py --device cpu
PYTHONPATH=src:. python3 scripts/explain_v0_6.py
```
