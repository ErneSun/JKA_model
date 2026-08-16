# Training and EMA

Formal V0.6 initializes from the validated V0.5 checkpoint used by its matched control.
Online encoder, `A`, decoder, split, fingerprint and normalizer are checked; then target
is hard-copied. V0.6 resume restores the saved target exactly and never hard-syncs.

After a successful optimizer update,

\[
\bar\theta\leftarrow\tau\bar\theta+(1-\tau)\theta.
\]

The default schedule rises linearly from 0.996 to 1.0 over planned optimizer updates,
following official JEPA momentum-target implementations. `tau`, EMA count,
optimizer-update count, target, optimizer, scheduler, AMP scaler, RNG, split,
normalizer and fingerprint are checkpointed. An AMP-skipped optimizer step also skips
EMA and its counter.

The full objective is the latest V0.5 total plus one- and multi-step JEPA. No V0.5 loss
or warmup semantics are removed.

The preregistered primary comparison uses `lambda_J1=1`, `lambda_Jmulti=1` and 75
continuation epochs; the control uses `0,0` with the same 75 epochs. These values are
fixed before GPU results. Any later coefficient or duration change is a new experiment,
not an in-place repair of the primary result.
