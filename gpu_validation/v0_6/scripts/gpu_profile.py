#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import torch

from jka_model.config import load_config
from train.train_v0_6 import initialize_v0_6_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gpu_validation/v0_6/configs/gpu_smoke.yaml")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    config = load_config(args.config)
    model = initialize_v0_6_model(config, device="cuda").eval()
    shape = config.advection_diffusion_2d
    x = torch.randn(8, 1, shape.nx, shape.ny, device="cuda")
    dts = torch.full((8, config.data.horizon), shape.base_dt, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.steps):
            model.rollout(x, dts)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "steps_per_second": args.steps / elapsed,
                "peak_gpu_memory": torch.cuda.max_memory_allocated(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
