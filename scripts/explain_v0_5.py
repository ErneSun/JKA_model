#!/usr/bin/env python3
"""Print the auditable V0.5 architecture and unit/gradient routes."""

from __future__ import annotations

import json


def main() -> None:
    print(
        json.dumps(
            {
                "canonical_training": "src.train.train_v0_5.train_v0_5",
                "canonical_evaluation": "src.eval.evaluate_v0_5.evaluate_v0_5",
                "state_shapes": {
                    "field": "[B,C,Nx,Ny]",
                    "window": "[B,H,C,Nx,Ny]",
                    "rollout": "[B,K,C,Nx,Ny]",
                },
                "model_route": (
                    "normalized field -> circular CNN E_K -> exp(A*dt) closed loop "
                    "-> D_train -> normalized field"
                ),
                "physics_route": (
                    "decoded model field -> differentiable inverse normalization -> raw units "
                    "-> mass + trapezoidal PDE operator"
                ),
                "trainable_modules": ["E_K", "A", "D_train"],
                "excluded": [
                    "JEPA",
                    "EMA",
                    "z_r",
                    "GRU",
                    "Transformer",
                    "attention",
                    "MPC",
                    "RL",
                    "V0.6",
                ],
                "local_status": (
                    "CPU correctness only; GPU validation NOT RUN; "
                    "scientific acceptance PENDING_GPU"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
