# V0.5 complete GPU validation

- validation id: `v05-final-20260812`
- workflow status: **PASS**
- scientific status: **FAIL**
- overall acceptance: **NOT_ACCEPTED**
- scientific checkpoint: `best_forecast_post_warmup`

## Hard gates

- frequency: **FAIL**; relative error 0.963098, threshold 0.05
- long rollout vs persistence: **PASS**; 0.629532 vs 1.03311

## Physics vs no-physics

| Horizon | Metric | Physics | No physics | Relative change |
|---|---:|---:|---:|---:|
| short | rmse | 0.751378 | 0.777501 | -3.360% |
| short | mass_drift | 1.25727 | 2.26328 | -44.449% |
| short | operator | 425.379 | 460.807 | -7.688% |
| medium | rmse | 0.721763 | 0.73976 | -2.433% |
| medium | mass_drift | 1.28824 | 2.45659 | -47.560% |
| medium | operator | 26.5916 | 28.8352 | -7.781% |
| long | rmse | 0.629532 | 0.636513 | -1.097% |
| long | mass_drift | 1.5609 | 2.69477 | -42.077% |
| long | operator | 5.32233 | 5.7889 | -8.060% |

## Checklist

- [x] A_git_state
- [x] B_cuda_environment
- [x] C_gpu_identity
- [x] D_matrix_exp_preflight
- [x] E_cpu_gpu_parity
- [x] F_amp_finite
- [x] G_smoke_gradients
- [x] H_gpu_short_training
- [x] I_no_physics_full
- [x] J_physics_full
- [x] K_exact_resume
- [x] L_held_out_long_rollout
- [x] M_spectrum_reviewed
- [x] N_physics_reviewed
- [x] O_performance_recorded
- [x] P_peak_memory_recorded
- [x] Q_artifacts_saved
- [x] R_acceptance_reviewed
