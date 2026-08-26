# Review addendum — Phase-3 audit threshold correction

The original GPU audit completed successfully and all expected seed artifacts are present. Its
reported route classification is superseded by this addendum because the audit compared a
divergence **RMS** value directly with `max_divergence_mse=0.02`. The official cylinder observable
contract compares divergence RMS with the dimensionally matching threshold

\[
\sqrt{0.02}=0.141421\ldots.
\]

All three reconstruction divergence values (`0.04946`, `0.03348`, `0.03392`) are below that RMS
limit. They are also 40.6%–53.8% lower than the corresponding raw-data divergence RMS. No-slip and
outer-boundary errors remain below their MSE gates. Therefore the corrected three-seed decision is:

- reconstruction physics: `PASS` (originally false-failed);
- latent round trip: `FAIL` (`0.3677`–`0.5310`, gate `0.25`);
- nominal tangent: `PASS`;
- Phase-2 condition observer: `NOT_SUPPORTED`;
- corrected next candidate: `JOINT_MARKOV_REPRESENTATION`.

The original JSON is retained unchanged for provenance. The joint workflow reclassifies the raw
seed metrics with the corrected contract and records a separate `audit_reassessment.json` before
training.
