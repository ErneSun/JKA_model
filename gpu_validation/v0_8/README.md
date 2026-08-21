# V0.8 RTX-5080 validation

从仓库根目录、已激活 `.venv`、已 checkout/拉取目标 commit 后执行：

```bash
python gpu_validation/v0_8/scripts/gpu_validate_all.py --validation-id v08-final-$(date -u +%Y%m%dT%H%M%SZ) --seeds 47 53 59
```

正式流程要求 clean git、CUDA 与 BF16。每个阶段显示 `START/PASS/FAIL`；GPU epoch 不逐个打印，完整
epoch CSV 和 workflow log 保留在 raw run。相同 validation ID 自动使用 `-r1/-r2/...`。

Raw artifacts: `runs/v0_8/<resolved-id>/`  
Compact review: `gpu_validation/v0_8/results/<resolved-id>/`

Compact review 必须自包含审阅所需的小型证据：physical/grid/backbone/route audit、36 个候选训练摘要、
最终 family 的 epoch curves、locked-test metrics、逐 horizon 决策、十张图和 nested scientific report；
checkpoint、dataset 与 diagnostic tensors 仍只保留在 raw artifacts。

已有完整 raw run 只需按新门槛重新审阅时，不重复训练：

```bash
python gpu_validation/v0_8/scripts/gpu_reassess_existing.py --validation-id <existing-resolved-id>
```
