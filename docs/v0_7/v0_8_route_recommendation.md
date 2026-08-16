# V0.8 route recommendation contract

V0.7 does not implement V0.8. Its completed result selects only a defensible route:

- `MARKOVIAN`: keep the minimal instantaneous closure;
- `SHORT_MEMORY`: use the measured finite effective H;
- `LONG_MEMORY_CANDIDATE`: only then consider a compact recurrent/state-space closure;
- `INCONCLUSIVE`: do not increase memory-model complexity;
- negative closed-loop utility: first diagnose scale and rollout stability.

The generated recommendation is evidence-dependent and never treats software success as scientific support.
