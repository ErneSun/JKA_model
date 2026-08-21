# V0.9 single-RTX-5080 validation

从项目根目录运行：

```bash
python gpu_validation/v0_9/scripts/gpu_validate_all.py --validation-id v09-final-$(date -u +%Y%m%dT%H%M%SZ) --seeds 47 53 59
```

命令自动寻找最近的、同时具有 raw run 与 compact result 的严格 `V0.9_READY` V0.8 handoff。需要指定时增加
`--v0-8-id <resolved-v08-id>`。

如果找到的是严格 readiness 字段加入前生成的 V0.8 报告，G1 会使用已有 checkpoint 自动重评
9 个 locked-test context run 后再执行 handoff；不会重新训练 V0.8 backbone/context。

默认 `--v0-8-handoff-policy strict` 用于确认性实验。若 V0.8 聚合结论已经得到
`dynamic_context=SUPPORTED`，但未达到额外的 3/3 V0.9 readiness，可显式使用
`--v0-8-handoff-policy supported` 启动探索性 V0.9。该路线的报告固定标记为
`EXPLORATORY_CONDITIONAL`，即使 V0.9 机制通过也只写 `CONDITIONALLY_SUPPORTED`，且不能产生
`V1.0 READY`。

流程执行 targeted tests、V0.8 audit、controlled data/cache、validation-only rank sweep、known/latent 3×3
训练、locked test、physics rollout 和 compact report。若 ID 已存在，自动分配 `-r1/-r2/...`。

原始文件位于 `runs/v0_9/<resolved-id>/`；紧凑结果位于
`gpu_validation/v0_9/results/<resolved-id>/`。正式运行要求源码与配置 clean；已有 `runs/` 和版本化
`results/` 审计产物不触发 dirty-code 拒绝。
