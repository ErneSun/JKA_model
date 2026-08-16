#!/usr/bin/env python3
from __future__ import annotations

import json

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    properties = torch.cuda.get_device_properties(0)
    result = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "memory_gib": properties.total_memory / 1024**3,
        "torch": torch.__version__,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }
    probe = torch.randn(32, 32, device="cuda")
    result["finite_matmul"] = bool(torch.isfinite(probe @ probe.T).all())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["finite_matmul"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
