# GPU validation checklist

- [ ] A. Git commit and clean/dirty state recorded
- [ ] B. CUDA environment recorded
- [ ] C. GPU identity/count/memory recorded
- [ ] D. PyTorch CUDA and `matrix_exp` preflight passed
- [ ] E. Seeded CPU/GPU FP32 parity passed within tolerance
- [ ] F. AMP validation passed without NaN/Inf
- [ ] G. FP32 and AMP smoke passed with encoder/decoder/A gradients
- [ ] H. GPU short training completed
- [ ] I. GPU full no-physics ablation completed
- [ ] J. GPU full physics training completed
- [ ] K. Intermediate-checkpoint resume completed and compared
- [ ] L. Held-out long rollout completed
- [ ] M. Learned/true spectrum reviewed
- [ ] N. Mass and operator metrics reviewed
- [ ] O. Epoch time and samples/s recorded
- [ ] P. Peak allocated GPU memory recorded
- [ ] Q. Immutable result summary and run artifact pointers saved
- [ ] R. Final GPU/scientific acceptance reviewed and status updated
