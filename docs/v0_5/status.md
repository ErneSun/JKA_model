# V0.5 status

- LOCAL CPU: PASS
- GIT COMMIT: working tree (not committed by Codex)
- GPU VALIDATION: NOT RUN
- SCIENTIFIC: PENDING_GPU

Last unchanged full-suite baseline: 137 pytest tests and V0.1–V0.5 smoke passed. During
the present V0.5 audit, only affected gates were rerun as requested: 7 targeted V0.5 tests,
canonical V0.5 smoke, isolated-physics gradient diagnostic, run inspector, ruff, mypy, and
diff check passed on CPU. The tiny run is an integration/overfit test only and is not a
scientific acceptance run.

GPU status may be changed only after the independent checklist has server artifacts.
