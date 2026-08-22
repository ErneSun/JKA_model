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
    adaptive_stabilization_objective,
    curriculum_state,
    differentiable_adaptive_rollout,
    load_adaptive_checkpoint,
    operator_explained_fraction,
    relative_propagator_growth_loss,
    residual_decomposition,
    save_adaptive_cache,
)
from jka_model.config import (
    ProjectConfig,
    V08ContextConfig,
    V09AdaptiveConfig,
    V09TrainingConfig,
    load_config,
    stable_config_hash,
)
from jka_model.context import build_dynamic_context_model
from jka_model.observables import ObservableLossResult
from jka_model.residual import ResidualCache, ResidualTrajectory
from jka_model.residual.cache import file_sha256, save_residual_cache
from jka_model.training import TrainStage, configure_train_stage
from train.train_v0_8 import train_v0_8
from train.train_v0_9 import _stabilized_loss_bundle, train_v0_9


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


def test_bounded_coordinates_and_trust_gate_preserve_exact_nominal_initialization() -> None:
    adapter = LowRankAdaptiveOperator(
        torch.zeros(6, 6),
        4,
        V09AdaptiveConfig(
            rank=2,
            rank_candidates=(1, 2),
            width=8,
            bounded_coordinates=True,
            eta_max=0.2,
            trust_gate=True,
        ),
    )
    context = torch.randn(3, 4)
    eta, gate = adapter.adaptation_parameters(context)
    assert torch.equal(eta, torch.zeros_like(eta))
    assert torch.allclose(gate, torch.full_like(gate, 0.2), atol=1e-6)
    with torch.no_grad():
        adapter.operator_coordinate_head[-1].bias.fill_(20.0)
    bounded, _ = adapter.adaptation_parameters(context)
    assert torch.all(bounded.abs() <= 0.2 + 1e-7)


def test_rollout_curriculum_and_relative_growth_contract() -> None:
    config = V09TrainingConfig(
        epochs=11,
        rollout_horizons=(4, 8, 16),
        rollout_start_fractions=(0.0, 0.5, 0.8),
        rollout_weights=(1.0, 0.5, 0.25),
        lambda_rollout=1.0,
        lambda_physics=0.1,
        physics_start_fraction=0.5,
        physics_ramp_duration_fraction=0.25,
        physics_horizon=8,
    )
    assert curriculum_state(config, 0).active_horizons == (4,)
    middle = curriculum_state(config, 5)
    assert middle.active_horizons == (4, 8)
    assert middle.physics_scale == 0.0
    assert curriculum_state(config, 6).physics_scale > 0.0
    assert curriculum_state(config, 8).physics_scale == 1.0
    final = curriculum_state(config, 10, validation=True)
    assert final.active_horizons == (4, 8, 16)
    assert final.physics_scale == 1.0

    nominal = torch.zeros(2, 2)
    dts = torch.full((1, 2), 0.1)
    assert relative_propagator_growth_loss(
        nominal.expand(1, 2, 2, 2), nominal, dts, margin=0.0
    ) == 0
    amplified = torch.eye(2).reshape(1, 1, 2, 2).expand(1, 2, 2, 2)
    assert relative_propagator_growth_loss(amplified, nominal, dts, margin=0.0) > 0


def test_closed_loop_rollout_retains_adapter_gradients() -> None:
    model = AdaptiveKoopmanModel(
        _context(),
        LowRankAdaptiveOperator(
            torch.zeros(6, 6),
            4,
            V09AdaptiveConfig(rank=2, rank_candidates=(1, 2), width=8),
        ),
    )
    rollout = differentiable_adaptive_rollout(
        model,
        torch.randn(2, 3, 6),
        torch.full((2, 2), 0.1),
        torch.full((2, 4), 0.1),
        torch.randn(2, 3),
        None,
    )
    rollout["adapted"][:, -1].square().mean().backward()
    gradient = model.operator_adapter.operator_coordinate_head[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert not any(parameter.grad is not None for parameter in model.context_encoder.parameters())


def test_complete_stabilization_objective_is_finite_and_differentiable() -> None:
    model = AdaptiveKoopmanModel(
        _context(),
        LowRankAdaptiveOperator(
            -0.01 * torch.eye(6),
            4,
            V09AdaptiveConfig(
                rank=2,
                rank_candidates=(1, 2),
                width=8,
                bounded_coordinates=True,
                trust_gate=True,
            ),
        ),
    )
    training = V09TrainingConfig(
        epochs=2,
        rollout_horizons=(2, 4),
        rollout_start_fractions=(0.0, 0.5),
        rollout_weights=(1.0, 0.5),
        lambda_rollout=1.0,
        lambda_propagator_growth=0.1,
        physics_horizon=1,
    )
    batch = {
        "history_z": torch.randn(2, 3, 6),
        "history_dts": torch.full((2, 2), 0.1),
        "future_dts": torch.full((2, 4), 0.1),
        "future_conditions": torch.zeros(2, 4, 2),
        "target_latents": torch.randn(2, 4, 6),
        "context_parameters": torch.randn(2, 3),
    }
    result = adaptive_stabilization_objective(
        model,
        batch,
        torch.ones(6),
        torch.zeros(2),
        torch.ones(2),
        training,
        "latent_inferred",
        curriculum_state(training, 1),
        torch.tensor([True, False]),
    )
    assert torch.isfinite(result.total)
    assert "rollout_gain_h4" in result.terms
    result.total.backward()
    gradient = model.operator_adapter.operator_coordinate_head[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()


def test_stabilized_bundle_applies_problem_observables_at_multiple_horizons() -> None:
    model = AdaptiveKoopmanModel(
        _context(),
        LowRankAdaptiveOperator(
            -0.01 * torch.eye(6),
            4,
            V09AdaptiveConfig(rank=2, rank_candidates=(1, 2), width=8),
        ),
    )
    payload = load_config("gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml").to_dict()
    payload["v0_9_adaptive"].update(
        {"rank": 2, "rank_candidates": [1, 2], "width": 8}
    )
    payload["v0_9_training"].update(
        {
            "epochs": 2,
            "rollout_horizons": [2, 4],
            "rollout_start_fractions": [0.0, 0.0],
            "rollout_weights": [1.0, 0.5],
            "physics_horizon": 2,
            "observable_horizons": [2, 4],
            "observable_horizon_weights": [0.25, 0.75],
            "observable_horizon_probabilities": [],
            "phase1_enabled": False,
            "lambda_physics": 0.1,
            "lambda_observable_noninferiority": 0.1,
        }
    )
    config = ProjectConfig.from_dict(payload)

    class FakeObservables:
        def __init__(self) -> None:
            self.horizons: list[int] = []

        def target_batch(self, trajectory_ids, target_indices, horizon, limit):
            self.horizons.append(horizon)
            return torch.zeros(limit, 6), {"horizon": horizon}

        def loss(self, predicted_latent, target_raw, metadata):
            value = (predicted_latent - target_raw).square().mean()
            return ObservableLossResult(value, {"observable_mock": value})

    observables = FakeObservables()
    batch = {
        "history_z": torch.randn(2, 3, 6),
        "history_dts": torch.full((2, 2), 0.1),
        "future_dts": torch.full((2, 4), 0.1),
        "future_conditions": torch.zeros(2, 4, 2),
        "target_latents": torch.randn(2, 4, 6),
        "context_parameters": torch.randn(2, 3),
        "schedule_type": ["smooth", "abrupt"],
        "trajectory_id": ["a", "b"],
        "target_index": torch.tensor([2, 3]),
    }
    total, terms = _stabilized_loss_bundle(
        model,
        batch,
        torch.ones(6),
        torch.zeros(2),
        torch.ones(2),
        config,
        epoch=1,
        physical=observables,  # type: ignore[arg-type]
        validation=True,
    )
    assert observables.horizons == [2, 4]
    assert "observable_mock_h2" in terms and "observable_mock_h4" in terms
    assert torch.isfinite(total)
    total.backward()
    gradient = model.operator_adapter.operator_coordinate_head[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()


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
        {
            "rank": 2,
            "rank_candidates": [1, 2, 4],
            "width": 8,
            "bounded_coordinates": False,
            "trust_gate": False,
        }
    )
    payload["v0_9_training"].update(
        {
            "epochs": 1,
            "batch_size": 8,
            "rollout_batch_size": 4,
            "patience": 1,
            "precision": "fp32",
            "rollout_horizons": [2, 4],
            "rollout_start_fractions": [0.0, 0.0],
            "rollout_weights": [1.0, 0.5],
            "lambda_rollout": 1.0,
            "lambda_propagator_growth": 0.1,
                "physics_horizon": 1,
                "lambda_physics": 0.0,
                    "observable_horizons": [],
                    "observable_horizon_weights": [],
                    "observable_horizon_probabilities": [],
                    "phase1_enabled": False,
                }
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

    legacy_config = saved["config"]
    for name in ("bounded_coordinates", "eta_max", "trust_gate", "trust_gate_bias"):
        legacy_config["v0_9_adaptive"].pop(name)
    for name in (
        "rollout_horizons",
        "rollout_start_fractions",
        "rollout_weights",
        "rollout_batch_size",
        "rollout_stride",
        "lambda_rollout",
        "lambda_propagator_growth",
        "propagator_growth_margin",
        "operator_burden_target",
        "physics_start_fraction",
        "physics_ramp_duration_fraction",
        "physics_horizon",
        "physics_batch_size",
        "lambda_physics",
        "physics_velocity_weight",
        "physics_vorticity_weight",
        "physics_divergence_weight",
        "physics_boundary_weight",
        "physics_lift_weight",
        "physics_drag_weight",
        "observable_names",
        "observable_component_weights",
        "observable_horizons",
        "observable_horizon_weights",
        "lambda_observable_noninferiority",
        "observable_noninferiority_margin",
        "observable_noninferiority_floor",
        "rank_sweep_epochs",
    ):
        legacy_config["v0_9_training"].pop(name)
    for name in (
        "min_dynamic_over_static_gain",
        "observable_pair_pass_fraction",
        "frequency_resolution_bins",
    ):
        legacy_config["v0_9_evaluation"].pop(name)
    saved["config_hash"] = stable_config_hash(legacy_config)
    legacy_path = tmp_path / "legacy_v0_9.pt"
    torch.save(saved, legacy_path)
    reloaded = load_adaptive_checkpoint(legacy_path)
    assert reloaded["config"]["v0_9_training"]["rollout_stride"] == 1
