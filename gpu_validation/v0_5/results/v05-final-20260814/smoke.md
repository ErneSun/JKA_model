# V0.5 FP32 and AMP smoke

- status: **PASS**
- AMP precision: `amp_bf16`
- FP32 run: `/home/ai_group/Documents/files/SQ/JKA_model/runs/v0_5/gpu/20260815T073418Z-2294e2e3`
- AMP run: `/home/ai_group/Documents/files/SQ/JKA_model/runs/v0_5/gpu/20260815T073421Z-7e5982d5`
- FP32 gradients: `{'encoder': 0.0006884682807140052, 'decoder': 0.21347442269325256, 'generator': 3.258126525906846e-05}`
- AMP gradients: `{'encoder': 0.0006740021635778248, 'decoder': 0.2133675068616867, 'generator': 3.206082692486234e-05}`
- isolated physics gradients: `{'fp32': {'encoder': 1.2016182608931558e-06, 'decoder': 0.047264497727155685, 'generator': 1.1699823054822645e-10}, 'amp': {'encoder': 1.2013940704491688e-06, 'decoder': 0.047228097915649414, 'generator': 1.1688708334567366e-10}}`
