from __future__ import annotations

import json
from pathlib import Path

import torch

from jka_model.adaptive import (
    AdaptiveCache,
    AdaptiveKoopmanModel,
    AdaptiveTrajectory,
    LowRankAdaptiveOperator,
    adaptive_latent_rollout,
    operator_explained_fraction,
    residual_decomposition,
    save_adaptive_cache,
)
from jka_model.config import ProjectConfig, V08ContextConfig, V09AdaptiveConfig, load_config
from jka_model.context import build_dynamic_context_model
from jka_model.residual import ResidualCache, ResidualTrajectory
from jka_model.residual.cache import file_sha256, save_residual_cache
from jka_model.training import TrainStage, configure_train_stage
from train.train_v0_8 import train_v0_8
from train.train_v0_9 import train_v0_9


def _context(latent_dim: int = 6, history: int = 3):
    config = V08ContextConfig(
        family="history_mlp", context_dim=4, history_length=history, width=8, heads=1
    )
    dynamic = build_dynamic_context_model(
        config,
        family="history_mlp",
        latent_dim=latent_dim,
        parameter_dim=3,
        history=history,
    )
    return dynamic.context_encoder


def test_low_rank_zero_initialization_exactly_recovers_nominal_and_is_differentiable() -> None:
    torch.manual_seed(3)
    nominal = torch.randn(6, 6) * 0.01
    adapter = LowRankAdaptiveOperator(
        nominal,
        4,
        V09AdaptiveConfig(rank=2, rank_candidates=(1, 2, 4), width=8),
    )
    context = torch.randn(5, 4)
    z = torch.randn(5, 6)
    dt = torch.full((5, 1), 0.1)
    prediction, eta, delta, adapted = adapter.step(z, context, dt)
    expected = torch.einsum(
        "bij,bj->bi", torch.linalg.matrix_exp(nominal.unsqueeze(0) * 0.1), z
    )
    assert torch.equal(eta, torch.zeros_like(eta))
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(adapted, nominal.unsqueeze(0).expand_as(adapted))
    assert torch.allclose(prediction, expected, atol=1e-7, rtol=1e-6)
    prediction.square().mean().backward()
    assert adapter.operator_coordinate_head[-1].weight.grad is not None
    assert torch.isfinite(adapter.operator_coordinate_head[-1].weight.grad).all()


def test_known_and_latent_condition_visibility_is_strict() -> None:
    nominal = torch.zeros(6, 6)
    known = LowRankAdaptiveOperator(
        nominal,
        4,
        V09AdaptiveConfig(
            condition_mode="known", rank=2, rank_candidates=(1, 2), width=8
        ),
    )
    latent = LowRankAdaptiveOperator(
        nominal,
        4,
        V09AdaptiveConfig(rank=2, rank_candidates=(1, 2), width=8),
    )
    context = torch.randn(2, 4)
    try:
        known.coordinates(context)
    except ValueError as error:
        assert "requires current" in str(error)
    else:
        raise AssertionError("known mode accepted missing condition")
    try:
        latent.coordinates(context, torch.randn(2, 2))
    except ValueError as error:
        assert "forbids" in str(error)
    else:
        raise AssertionError("latent mode accepted condition leakage")


def test_residual_decomposition_and_gamma_operator_contract() -> None:
    truth = torch.tensor([[2.0, 1.0]])
    nominal_prediction = torch.tensor([[0.0, 1.0]])
    adapted_prediction = torch.tensor([[1.5, 1.0]])
    nominal, explained, remaining = residual_decomposition(
        truth, nominal_prediction, adapted_prediction
    )
    assert torch.allclose(nominal, explained + remaining)
    assert torch.allclose(operator_explained_fraction(nominal, remaining), torch.tensor(0.9375))


def test_only_operator_adapter_is_trainable_and_rollout_uses_predicted_history() -> None:
    context = _context()
    adapter = LowRankAdaptiveOperator(
        torch.zeros(6, 6),
        4,
        V09AdaptiveConfig(rank=2, rank_candidates=(1, 2), width=8),
    )
    model = AdaptiveKoopmanModel(context, adapter)
    groups = configure_train_stage(model, TrainStage.ADAPTIVE)
    assert groups == {"context_encoder": False, "operator_adapter": True}
    assert not any(parameter.requires_grad for parameter in model.context_encoder.parameters())
    result = adaptive_latent_rollout(
        model,
        torch.randn(2, 3, 6),
        torch.full((2, 2), 0.1),
        torch.full((2, 4), 0.1),
        torch.randn(2, 3),
        None,
    )
    assert result["adapted"].shape == (2, 5, 6)
    assert result["eta"].shape == (2, 4, 2)
    assert torch.isfinite(result["adapted"]).all()


def test_v0_9_operator_only_training_writes_reloadable_checkpoint(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(29)
    backbone = tmp_path / "backbone.pt"
    backbone.write_bytes(b"frozen-backbone-fingerprint")
    backbone_sha = file_sha256(backbone)
    residual_items: list[ResidualTrajectory] = []
    latent_by_id: dict[str, torch.Tensor] = {}
    for split in ("train", "validation", "test"):
        for index in range(2):
            trajectory_id = f"{split}-{index}"
            latent = torch.randn(9, 8, generator=generator)
            latent_by_id[trajectory_id] = latent
            residual_items.append(
                ResidualTrajectory(
                    trajectory_id=trajectory_id,
                    split=split,
                    latents=latent,
                    dts=torch.full((8,), 0.1),
                    parameters=torch.tensor([100.0, 1.0, 1.0]),
                    residuals=0.05 * latent[:-1],
                )
            )
    split_manifest = {
        split: [f"{split}-0", f"{split}-1"]
        for split in ("train", "validation", "test")
    }
    residual_cache = ResidualCache(
        trajectories=tuple(residual_items),
        backbone_checkpoint_sha256=backbone_sha,
        backbone_config_hash="b" * 64,
        data_fingerprint="d" * 64,
        split_manifest={**split_manifest, "seed": 17},
        normalizer_state={"kind": "standard", "eps": 1e-6},
    )
    residual_cache_path = tmp_path / "residual_cache.pt"
    save_residual_cache(residual_cache, residual_cache_path)
    route_path = tmp_path / "route.json"
    route_path.write_text(json.dumps({"residual_route": "R2"}), encoding="utf-8")
    context_result = train_v0_8(
        load_config("configs/v0_8/cylinder_wake_cpu_smoke.yaml"),
        backbone_checkpoint=backbone,
        residual_cache=residual_cache_path,
        v0_7_route_result=route_path,
        run_dir=tmp_path / "context",
        device="cpu",
    )
    assert context_result.best_checkpoint is not None
    context_sha = file_sha256(context_result.best_checkpoint)
    nominal_generator = torch.diag(torch.linspace(-0.02, -0.01, 8))
    adaptive_items = []
    for split in ("train", "validation", "test"):
        for index in range(2):
            trajectory_id = f"{split}-{index}"
            latent = latent_by_id[trajectory_id]
            nominal = torch.einsum(
                "ij,tj->ti", torch.linalg.matrix_exp(0.1 * nominal_generator), latent[:-1]
            )
            adaptive_items.append(
                AdaptiveTrajectory(
                    trajectory_id=trajectory_id,
                    split=split,
                    schedule_type="smooth" if index == 0 else "abrupt",
                    transition_index=4,
                    latents=latent,
                    dts=torch.full((8,), 0.1),
                    context_parameters=torch.tensor([100.0, 1.0, 1.0]),
                    conditions=torch.stack(
                        (torch.linspace(80.0, 120.0, 8), torch.ones(8)), dim=-1
                    ),
                    nominal_residuals=latent[1:] - nominal,
                )
            )
    adaptive_cache = AdaptiveCache(
        trajectories=tuple(adaptive_items),
        backbone_checkpoint_sha256=backbone_sha,
        backbone_config_hash="b" * 64,
        context_checkpoint_sha256=context_sha,
        data_fingerprint="e" * 64,
        split_manifest={**split_manifest, "seed": 17},
        normalizer_state={"kind": "standard", "eps": 1e-6},
        nominal_generator=nominal_generator,
    )
    adaptive_cache_path = tmp_path / "adaptive_cache.pt"
    save_adaptive_cache(adaptive_cache, adaptive_cache_path)

    payload = load_config("gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml").to_dict()
    v0_8 = load_config("configs/v0_8/cylinder_wake_cpu_smoke.yaml")
    payload["koopman"] = v0_8.koopman.to_dict() if v0_8.koopman else None
    payload["field_autoencoder"] = (
        v0_8.field_autoencoder.to_dict() if v0_8.field_autoencoder else None
    )
    payload["v0_8_context"] = (
        v0_8.v0_8_context.to_dict() if v0_8.v0_8_context else None
    )
    payload["v0_9_adaptive"].update(
        {"rank": 2, "rank_candidates": [1, 2, 4], "width": 8}
    )
    payload["v0_9_training"].update(
        {"epochs": 1, "batch_size": 8, "patience": 1, "precision": "fp32"}
    )
    config = ProjectConfig.from_dict(payload)
    result = train_v0_9(
        config,
        context_checkpoint=context_result.best_checkpoint,
        adaptive_cache=adaptive_cache_path,
        run_dir=tmp_path / "adaptive",
        device="cpu",
    )
    assert result.completed_epochs == 1
    assert result.latest_checkpoint.is_file()
    assert result.best_checkpoint.is_file()
    saved = torch.load(result.best_checkpoint, map_location="cpu", weights_only=False)
    assert saved["train_stage"] == "adaptive"
    assert saved["adaptive_cache_fingerprint"] == adaptive_cache.fingerprint
