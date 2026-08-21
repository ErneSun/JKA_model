# V0.9 testing

日常开发只运行新增且必要的测试：

```bash
python -m pytest -q tests/test_v0_9_adaptive.py tests/test_v0_9_physical_problem.py tests/test_v0_9_workflow.py
```

这些测试覆盖低秩数学合同、零更新、bounded coordinates/trust gate、跨预测历史梯度、课程激活、相对传播子增长、
冻结 decoder 可观测量梯度、多时间尺度 adapter 调用、旧 checkpoint 兼容、通用门控的零 baseline/频率分辨率/
inconclusive 行为、condition visibility、residual decomposition、teacher-free rollout、smooth/abrupt schedules、
严格 handoff、独立与 joint nested aggregation、run-ID 与 completion contract。

完整旧 suite 不在每次修改后重复；仅在 cross-cutting/release gate 时执行。正式 RTX 5080 workflow 会先运行上述
targeted tests。每个 major stage 输出 START/PASS/FAIL；正式训练只打印开始和最终摘要，完整 epoch 指标保存在日志。

正式一行命令见 `gpu_validation/v0_9/README.md`。
