#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from jka_model.config import load_config  # noqa: E402
from train.train_v0_6 import initialize_v0_6_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gpu_validation/v0_6/configs/gpu_smoke.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    config = load_config(args.config)
    model = initialize_v0_6_model(config, device="cuda")
    print(
        json.dumps(
            {
                "status": "PASS",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "target_frozen": all(
                    not p.requires_grad for p in model.target_encoder.parameters()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
