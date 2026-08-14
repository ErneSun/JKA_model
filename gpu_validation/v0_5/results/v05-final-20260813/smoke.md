# V0.5 FP32 and AMP smoke

- status: **PASS**
- AMP precision: `amp_bf16`
- FP32 run: `/home/ai_group/Documents/files/SQ/JKA_model/runs/v0_5/gpu/20260813T015116Z-7dd02305`
- AMP run: `/home/ai_group/Documents/files/SQ/JKA_model/runs/v0_5/gpu/20260813T015119Z-5a9608a8`
- FP32 gradients: `{'encoder': 0.0006900927401147783, 'decoder': 0.3170258700847626, 'generator': 3.253317845519632e-05}`
- AMP gradients: `{'encoder': 0.0006798153626732528, 'decoder': 0.3170907497406006, 'generator': 3.183978333254345e-05}`
- isolated physics gradients: `{'fp32': {'encoder': 0.0007769782096147537, 'decoder': 31.149864196777344, 'generator': 7.746857733081924e-08}, 'amp': {'encoder': 0.000778517103753984, 'decoder': 31.178997039794922, 'generator': 7.748615615810195e-08}}`
