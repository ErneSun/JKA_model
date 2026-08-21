# V0.8 scientific report

PHYSICAL PROBLEM: cylinder_wake_2d  
BACKBONE STATUS: PASS  
V0.7 ROUTE ON NEW PROBLEM: R3  
CONTEXT FAMILY: HISTORY_MLP  
RESIDUAL PREDICTION: SUPPORTED  
HISTORY VALUE: SUPPORTED  
DYNAMIC CONTEXT: SUPPORTED  
CLOSED LOOP UTILITY: POSITIVE  
PHYSICS STATUS: PASS  
V0.9 OPERATOR-ADAPTATION READINESS: READY

Formal nested run count: 9. Evidence is aggregated first across context initializations within a backbone/data seed and then across backbone/data seeds.

The additive residual is a utility probe. A0 remains frozen; eta_t, adaptive A_t, and persistent z_R are absent. Attention weights, when available, are diagnostics rather than causal explanations.
