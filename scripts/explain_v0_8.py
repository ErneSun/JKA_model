#!/usr/bin/env python3
"""Print the V0.8 causal shapes and route-owned context contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from jka_model.config import load_config  # noqa: E402
from jka_model.context import build_dynamic_context_model, load_v0_7_route  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0_8/cylinder_wake_cpu_smoke.yaml", type=Path)
    parser.add_argument("--route-result", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    assert config.koopman and config.v0_8_context and config.cylinder_wake_2d
    if args.route_result is None:
        route_path = Path("/tmp/v0_8_explain_route.json")
        route_path.write_text(
            json.dumps({"residual_route": "R3", "locked_history_steps": 4}),
            encoding="utf-8",
        )
    else:
        route_path = args.route_result
    route = load_v0_7_route(route_path)
    if route.context_family is None:
        print(f"route={route.residual_route} context_training=STOP")
        return
    history = int(route.history_length or 1)
    model = build_dynamic_context_model(
        config.v0_8_context,
        family=route.context_family,
        latent_dim=config.koopman.state_dim,
        parameter_dim=config.data.parameter_dim,
        history=history,
    )
    batch = 2
    physical = torch.randn(
        batch,
        history,
        3,
        config.cylinder_wake_2d.nx,
        config.cylinder_wake_2d.ny,
    )
    z_history = torch.randn(batch, history, config.koopman.state_dim)
    history_dts = torch.full((batch, history - 1), config.cylinder_wake_2d.snapshot_dt)
    next_dt = torch.full((batch, 1), config.cylinder_wake_2d.snapshot_dt)
    parameters = torch.tensor([config.cylinder_wake_2d.reynolds_number, 1.0, 1.0]).repeat(batch, 1)
    context, residual, adequacy = model(z_history, history_dts, next_dt, parameters)
    print(f"physical_history={tuple(physical.shape)}")
    print(f"z_history={tuple(z_history.shape)}")
    print(f"residual_target={tuple(residual.shape)}")
    print(f"route={route.residual_route}")
    print(f"context_family={route.context_family}")
    print(f"c_t={tuple(context.shape)} adequacy={tuple(adequacy.shape)}")
    print("A0_frozen=True eta_t=False adaptive_A_t=False persistent_z_R=False")


if __name__ == "__main__":
    main()
