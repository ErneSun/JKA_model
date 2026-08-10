# Training

Canonical training function: `src.train.train_v0_5.train_v0_5`

CLI: `scripts/train_v0_5.py`

Signature:

```python
train_v0_5(config, *, device=None, resume_from=None, run_name=None)
```

The trainer owns the sole optimization loop. It creates trajectory-level splits, fits
normalization only on train trajectories, constructs canonical windows, trains closed-loop
multi-step forecasts, evaluates validation windows, and writes a checkpoint each epoch.

Resume restores model, optimizer, scheduler, AMP scaler when present, epoch, global step,
Python/NumPy/PyTorch RNG, normalizer, split manifest, and data fingerprint. A config,
fingerprint, or split mismatch fails rather than silently re-splitting.

Run layout is `runs/v0_5/<cpu|gpu>/<run_id>/` with config, metadata, logs, checkpoints,
evaluation, plots, reports, and profiler directories. `logs/epoch_metrics.csv` is the
canonical history.

```bash
# CPU
python scripts/train_v0_5.py --config configs/v0_5/advection_diffusion_2d_cpu_tiny_train.yaml --device cpu

# GPU and resume
python gpu_validation/v0_5/scripts/gpu_train.py --config gpu_validation/v0_5/configs/gpu_full.yaml
python gpu_validation/v0_5/scripts/gpu_train.py --config gpu_validation/v0_5/configs/gpu_full.yaml --resume-from runs/v0_5/gpu/<run-id>/checkpoints/epoch_XXXX.pt

# Held-out evaluation and inspection
python gpu_validation/v0_5/scripts/gpu_evaluate.py --run-dir runs/v0_5/gpu/<run-id>
python scripts/inspect_v0_5_run.py --run-dir runs/v0_5/gpu/<run-id>
```

## Local → Git → GPU Server

After local gates pass, review and commit the V0.5 diff on the user's chosen branch, push
that exact commit, then check it out on the CUDA server. Run the independent commands in
`gpu_validation/v0_5/README.md` and record the commit in the result summary. Do not assume
a branch name and do not relabel local CPU metrics as GPU/scientific results.
