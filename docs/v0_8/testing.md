# Testing

本阶段遵循收敛测试：本地只运行 V0.8 新物理问题、context 数学合同与 workflow/report 合同；不重复完整
V0.1–V0.7 suite。V0.7 必要残差定义通过 V0.8 集成测试直接复用并核对。

本地必要测试：

```bash
python -m pytest -q tests/test_v0_8_physical_problem.py tests/test_v0_8_context.py tests/test_v0_8_workflow.py
```

远程 RTX 5080 正式流程的一行命令：

```bash
python gpu_validation/v0_8/scripts/gpu_validate_all.py --validation-id v08-final-$(date -u +%Y%m%dT%H%M%SZ) --seeds 47 53 59
```

相同 ID 已存在时自动解析为 `-r1/-r2/...`，不会覆盖旧结果。原始大文件位于
`runs/v0_8/<resolved-id>/`，紧凑报告位于 `gpu_validation/v0_8/results/<resolved-id>/`。

紧凑目录除 locked-test CSV/figures 外，还必须包含 `compact_audit.json`、全部候选的
`candidate_training_summary.csv` 和最终 family 的 `selected_training_curves.csv`。它们保留 physical/grid、
backbone、V0.7 route、validation-only family selection 与训练收敛证据，但不复制 checkpoint 或大型 tensor。

若 G1–G5 已完成，只因 V0.8 判定/报告逻辑更新而需要复核，复用原 checkpoint，仅重跑 G6/G7：

```bash
python gpu_validation/v0_8/scripts/gpu_reassess_existing.py --validation-id <existing-resolved-id>
```
