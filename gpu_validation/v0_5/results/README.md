# GPU validation results

Store immutable server summaries and artifact pointers here; never relabel CPU output as GPU
evidence. Large checkpoints and profiler traces remain under ignored `runs/`.

New complete validations should use `scripts/gpu_validate_all.py`. It exports one compact
subdirectory per validation ID containing preflight, smoke, resume, multi-checkpoint evaluation,
profile, A-R checklist, physics-vs-ablation comparison, and the final scientific status. Commit
that result subdirectory; do not commit checkpoints or profiler traces.
