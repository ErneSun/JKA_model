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

