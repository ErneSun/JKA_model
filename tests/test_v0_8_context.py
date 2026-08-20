from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from jka_model.config import load_config
from jka_model.context import (
    CausalAttentionContextEncoder,
    ContextWindowDataset,
    build_dynamic_context_model,
    context_corrected_latent_rollout,
    load_v0_7_route,
    residual_training_scales,
    select_context_family,
)
from jka_model.residual import ResidualCache, ResidualTrajectory
from jka_model.residual.cache import file_sha256, save_residual_cache
from train.train_v0_8 import train_v0_8


def _cache(backbone_sha: str = "a" * 64) -> ResidualCache:
    trajectories = []
    generator = torch.Generator().manual_seed(91)
    for split_index, split in enumerate(("train", "validation", "test")):
        for index in range(2):
            latent = torch.randn(9, 8, generator=generator)
            residual = 0.2 * latent[:-1] + 0.1 * torch.roll(latent[:-1], 1, 0)
            if split != "train":
                residual = residual + 0.01 * split_index
            trajectories.append(
                ResidualTrajectory(
                    trajectory_id=f"{split}-{index}",
                    split=split,
                    latents=latent,
                    dts=torch.full((8,), 0.1),
                    parameters=torch.tensor([100.0, 1.0, 1.0]),
                    residuals=residual,
                )
            )
    return ResidualCache(
        trajectories=tuple(trajectories),
        backbone_checkpoint_sha256=backbone_sha,
        backbone_config_hash="b" * 64,
        data_fingerprint="c" * 64,
        split_manifest={
            "train": ["train-0", "train-1"],
            "validation": ["validation-0", "validation-1"],
            "test": ["test-0", "test-1"],
            "seed": 1,
        },
        normalizer_state={"kind": "standard", "eps": 1e-6},
    )


def _inputs(history: int = 4):
    return (
        torch.randn(3, history, 8),
        torch.full((3, history - 1), 0.1),
        torch.full((3, 1), 0.1),
        torch.randn(3, 3),
    )


def test_updated_router_has_only_r1_r2_r3_and_inconclusive(tmp_path: Path) -> None:
    assert select_context_family("R1") is None
    assert select_context_family("R2") == "instantaneous"
    assert select_context_family("R3") == "attention"
    assert select_context_family("INCONCLUSIVE") is None
    with pytest.raises(ValueError, match="R0 is obsolete"):
        select_context_family("R0")
    path = tmp_path / "route.json"
    path.write_text(
        json.dumps({"residual_route": "R3", "locked_history_steps": 4}),
        encoding="utf-8",
    )
    assert load_v0_7_route(path).history_length == 4


def test_context_shapes_zero_output_and_current_state_semantics() -> None:
    config = load_config("configs/v0_8/cylinder_wake_cpu_smoke.yaml")
    assert config.v0_8_context is not None
    model = build_dynamic_context_model(
        config.v0_8_context,
        family="instantaneous",
        latent_dim=8,
        parameter_dim=3,
        history=4,
    )
    values = _inputs()
    context, residual, adequacy = model(*values)
    assert context.shape == (3, 4)
    torch.testing.assert_close(residual, torch.zeros_like(residual))
    torch.testing.assert_close(adequacy, torch.zeros_like(adequacy))
    changed = list(values)
    changed[0] = values[0].clone()
    changed[0][:, :-1] += 1000
    context_changed, _, _ = model(*changed)
    torch.testing.assert_close(context, context_changed)
    assert not torch.allclose(context[0], context[1])


def test_r3_instantaneous_capacity_control_is_parameter_matched() -> None:
    config = load_config("gpu_validation/v0_8/configs/gpu_cylinder_context.yaml")
    assert config.v0_8_context is not None
    attention = build_dynamic_context_model(
        config.v0_8_context,
        family="attention",
        latent_dim=32,
        parameter_dim=3,
        history=8,
    )
    control = build_dynamic_context_model(
        config.v0_8_context,
        family="instantaneous_matched",
        latent_dim=32,
        parameter_dim=3,
        history=8,
    )
    attention_count = sum(parameter.numel() for parameter in attention.parameters())
    control_count = sum(parameter.numel() for parameter in control.parameters())
    assert abs(control_count - attention_count) / attention_count <= 0.10


def test_attention_is_causal_at_every_history_token() -> None:
    encoder = CausalAttentionContextEncoder(8, 4, 4, 3, 16, 2, 2, 2, 0.0).eval()
    values = list(_inputs())
    before = encoder.encode_sequence(*values)
    values[0] = values[0].clone()
    values[0][:, 2:] += 500.0
    after = encoder.encode_sequence(*values)
    torch.testing.assert_close(before[:, :2], after[:, :2], atol=1e-6, rtol=1e-6)
    assert encoder.last_attention_weights is not None
    weights = encoder.last_attention_weights
    assert torch.all(weights[..., torch.triu(torch.ones(4, 4), diagonal=1).bool()] == 0)


def test_history_alignment_no_future_and_shuffled_control() -> None:
    cache = _cache()
    plain = ContextWindowDataset(cache, "train", history=4)
    shuffled = ContextWindowDataset(
        cache, "train", history=4, shuffle_older_history=True, shuffle_seed=7
    )
    sample = plain[0]
    trajectory = cache.select("train")[0]
    index = int(sample["target_index"])
    torch.testing.assert_close(sample["history_z"], trajectory.latents[index - 3 : index + 1])
    torch.testing.assert_close(sample["next_dt"], trajectory.dts[index : index + 1])
    torch.testing.assert_close(sample["target_residual"], trajectory.residuals[index])
    torch.testing.assert_close(sample["history_z"][-1], shuffled[0]["history_z"][-1])


def test_residual_and_adequacy_scales_are_train_only() -> None:
    cache = _cache()
    residual_scale, adequacy_scale, fingerprint = residual_training_scales(cache)
    train = torch.cat([item.residuals for item in cache.select("train")])
    torch.testing.assert_close(residual_scale, train.square().mean(0).sqrt())
    torch.testing.assert_close(
        adequacy_scale,
        train.square().mean(-1).sqrt().square().mean().sqrt(),
    )
    assert len(fingerprint) == 64


def test_rollout_recycles_predictions_not_future_truth() -> None:
    class Core:
        def step(self, z, dt):
            return z + dt.unsqueeze(-1)

    class Context(torch.nn.Module):
        def forward(self, history_z, history_dts, next_dt, parameters):
            del history_dts, parameters
            correction = 0.1 * history_z[:, -1]
            return correction[:, :2], correction, correction[:, :1]

    history = torch.ones(1, 3, 4)
    rollout, _, _ = context_corrected_latent_rollout(
        Context(),
        Core(),
        history,
        torch.full((1, 2), 0.1),
        torch.full((1, 3), 0.1),
        torch.empty(1, 0),
    )
    current = history[:, -1]
    for step in range(3):
        current = current + 0.1 + 0.1 * current
        torch.testing.assert_close(rollout[:, step + 1], current)


def test_context_training_checkpoint_resume_restores_exact_state(tmp_path: Path) -> None:
    backbone = tmp_path / "backbone.pt"
    backbone.write_bytes(b"fingerprinted frozen backbone")
    cache = _cache(file_sha256(backbone))
    cache_path = tmp_path / "cache.pt"
    save_residual_cache(cache, cache_path)
    route = tmp_path / "route.json"
    route.write_text(json.dumps({"residual_route": "R2"}), encoding="utf-8")
    config = load_config("configs/v0_8/cylinder_wake_cpu_smoke.yaml")
    first = train_v0_8(
        config,
        backbone_checkpoint=backbone,
        residual_cache=cache_path,
        v0_7_route_result=route,
        run_dir=tmp_path / "first",
        device="cpu",
    )
    assert first.latest_checkpoint and first.best_checkpoint
    resumed = train_v0_8(
        config,
        backbone_checkpoint=backbone,
        residual_cache=cache_path,
        v0_7_route_result=route,
        run_dir=tmp_path / "resumed",
        device="cpu",
        resume_from=first.latest_checkpoint,
    )
    assert resumed.start_epoch == first.completed_epochs
    first_payload = torch.load(first.best_checkpoint, weights_only=False)
    resumed_payload = torch.load(resumed.best_checkpoint, weights_only=False)
    for name, value in first_payload["best_context_state"].items():
        torch.testing.assert_close(value, resumed_payload["best_context_state"][name])
