# V0.5 FP32 and AMP smoke

- status: **PASS**
- AMP precision: `amp_bf16`
- FP32 run: `/home/ai_group/Documents/files/SQ/JKA_model/runs/v0_5/gpu/20260812T072633Z-3672bcc8`
- AMP run: `/home/ai_group/Documents/files/SQ/JKA_model/runs/v0_5/gpu/20260812T072635Z-34e0015b`
- FP32 gradients: `{'encoder': 9.751930338097736e-05, 'decoder': 0.16553287208080292, 'generator': 1.1200307881154004e-07}`
- AMP gradients: `{'encoder': 9.639267955208197e-05, 'decoder': 0.16537117958068848, 'generator': 6.614767045221015e-08}`
- isolated physics gradients: `{'fp32': {'encoder': 9.69170723692514e-05, 'decoder': 8.424118995666504, 'generator': 3.393105518778583e-10}, 'amp': {'encoder': 9.711774328025058e-05, 'decoder': 8.444270133972168, 'generator': 3.379636848155343e-10}}`
