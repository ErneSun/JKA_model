#!/usr/bin/env python3
"""Print one transparent V0.6 optimizer/EMA step."""

from __future__ import annotations

import torch
from torch.optim import Adam

from jka_model.config import load_config
from jka_model.data import (
    ChannelStandardizer,
    TrajectoryWindowDataset,
    collate_problem_batches,
    make_split_manifest,
    select_split,
)
from jka_model.losses import compute_field_jepa_loss
from jka_model.models import normalized_parameter_distance
from jka_model.problems import create_problem_adapter
from jka_model.training.ema import EMATracker
from train.train_v0_6 import initialize_v0_6_model


def main() -> None:
    config = load_config("configs/v0_6/advection_diffusion_2d_cpu_smoke.yaml")
    adapter = create_problem_adapter(config)
    records = adapter.build_dataset(seed=config.training.seed)
    spec = adapter.build_problem_spec()
    manifest = make_split_manifest(records, config.data.split)
    normalizer = ChannelStandardizer(eps=config.data.normalization.eps).fit(records, manifest, spec)
    windows = TrajectoryWindowDataset(
        select_split(records, manifest, "train"),
        history=config.data.history,
        horizon=config.data.horizon,
        normalizer=normalizer,
    )
    batch = collate_problem_batches([windows[0], windows[1]]).to(dtype=torch.float32)
    model = initialize_v0_6_model(config, device="cpu")
    optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    tracker = EMATracker(config.ema, total_updates=2)
    current = batch.context_states_model[:, -1]
    z_k_online_current = model.encode(current)
    z_k_online_future = model.encode(batch.future_states_model)
    z_k_jepa_target = model.encode_target(batch.future_states_model)
    z_k_pred = model.core.rollout(z_k_online_current, batch.future_dts)[:, 1:]
    losses = compute_field_jepa_loss(
        model,
        batch,
        normalizer,
        spec,
        config.field_loss,
        config.jepa_loss,
        adapter.build_physics_constraints(),
        physics_scale=1.0,
    )
    distance_before = normalized_parameter_distance(model)
    losses.total.backward()
    print("z_k_online_current", z_k_online_current[0].detach().tolist())
    print("z_k_online_future", z_k_online_future[0, 0].detach().tolist())
    print("z_k_jepa_target", z_k_jepa_target[0, 0].tolist())
    print("z_k_pred", z_k_pred[0, 0].detach().tolist())
    print("L_K", float(losses.v0_5.koopman_one_step.detach()))
    print("L_JEPA", float((losses.jepa_one_step + losses.jepa_multi_step).detach()))
    print("online_grad", next(model.online_encoder.parameters()).grad is not None)
    print("target_grad", next(model.target_encoder.parameters()).grad)
    print("A_grad", model.core.A.grad is not None)
    optimizer.step()
    distance_after_optimizer = normalized_parameter_distance(model)
    tau = tracker.update_after_optimizer(model)
    print("EMA_distance_before", distance_before)
    print("EMA_distance_after_optimizer", distance_after_optimizer)
    print("EMA_distance_after_EMA", normalized_parameter_distance(model))
    print("tau", tau)


if __name__ == "__main__":
    main()
