# V0.7 residual decision report

This is the version-controlled interpretation contract. A completed GPU session writes the populated report to `gpu_validation/v0_7/results/<id>/reports/residual_decision_report.md`.

The report begins with:

1. residual magnitude (diagnostic only);
2. residual learnability;
3. conditional history gain;
4. closed-loop utility;
5. secondary memory class;
6. three-condition physics acceptance;
7. final `R1/R2/R3/INCONCLUSIVE` route;
8. evidence-bounded V0.8 recommendation.

It must state that low magnitude never discards a residual, validation route selection, locked test confirmation, effective H in steps and physical time, parameter-matched and shuffled controls, both levels of seed consistency, and each physics gate. Finite-history evidence is not an exact Mori–Zwanzig kernel.
