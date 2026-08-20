# V0.7 status

As of 2026-08-20 after removing the magnitude-based residual-discard route:

```text
LOCAL: NOT RERUN — route contract was edited without tests by explicit request
GPU: EXISTING ARTIFACTS RETAINED — no new training or validation was run
SCIENTIFIC: NEEDS RECLASSIFICATION — generated reports predate diagnostic-S_R plus R1-R3 routing
```

The current classifier records residual magnitude but never uses it to discard a residual. Predictability and conditional history gain select R1/R2/R3, followed by locked test confirmation and the existing physics gates. Existing evaluation records can be reclassified later without retraining; no V0.6 checkpoint is stored in Git.
