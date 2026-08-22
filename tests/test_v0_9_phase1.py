from __future__ import annotations

import pytest
import torch

from jka_model.adaptive import (
    AdaptiveKoopmanModel,
    LowRankAdaptiveOperator,
    curriculum_state,
    observable_error_attribution,
)
from jka_model.config import (
    ProjectConfig,
    V08ContextConfig,
    V09AdaptiveConfig,
    V09TrainingConfig,
    load_config,
)
from jka_model.context import build_dynamic_context_model
from jka_model.observables import (
    ObservableLossResult,
    fit_robust_observable_scales,
    standardized_huber,
)
from jka_model.optimization import InequalityAugmentedLagrangian, gradient_cosine_matrix
from train.train_v0_9 import _stabilized_loss_bundle


def _model() -> AdaptiveKoopmanModel:
    context_config = V08ContextConfig(
        family="history_mlp", context_dim=4, history_length=3, width=8, heads=1
    )
    context = build_dynamic_context_model(
        context_config,
        family="history_mlp",
        latent_dim=6,
        parameter_dim=3,
        history=3,
    ).context_encoder
    return AdaptiveKoopmanModel(
        context,
        LowRankAdaptiveOperator(
            -0.01 * torch.eye(6),
            4,
            V09AdaptiveConfig(rank=2, rank_candidates=(1, 2), width=8),
        ),
    )


def test_robust_scales_are_serializable_and_huber_limits_outliers() -> None:
    state = fit_robust_observable_scales(
        {
            "velocity": torch.tensor([-1.0, 0.0, 1.0, 1000.0]),
            "boundary": torch.zeros(4),
        },
        method="mad",
        epsilon=1.0e-6,
        split_fingerprint="train-only",
        relative_floors={"boundary": ("velocity", 1.0e-3)},
    )
    assert state.split_fingerprint == "train-only"
    assert state.scales["boundary"] >= state.scales["velocity"] * 1.0e-3
    error = torch.tensor([0.0, state.scales["velocity"], 1000.0], requires_grad=True)
    loss = standardized_huber(error, state.scales["velocity"], delta=1.0)
    loss.backward()
    assert torch.isfinite(loss) and error.grad is not None
    assert state.from_dict(state.to_dict()) == state


def test_augmented_lagrangian_updates_only_positive_inequality_pressure() -> None:
    manager = InequalityAugmentedLagrangian(
        ("divergence", "boundary", "burden"),
        initial_penalty=1.0,
        penalty_growth=2.0,
        maximum_penalty=8.0,
    )
    variable = torch.tensor(2.0, requires_grad=True)
    penalty = manager.penalty(
        {
            "divergence": variable - 1.0,
            "boundary": -variable,
            "burden": variable - 3.0,
        }
    )
    penalty.backward()
    assert variable.grad is not None and variable.grad > 0
    diagnostics = manager.update(
        {"divergence": 1.0, "boundary": -2.0, "burden": -1.0}
    )
    assert diagnostics["multiplier_divergence"] > 0
    assert diagnostics["multiplier_boundary"] == 0
    restored = InequalityAugmentedLagrangian(
        ("divergence", "boundary", "burden"),
        initial_penalty=1.0,
        penalty_growth=2.0,
        maximum_penalty=8.0,
    )
    restored.load_state_dict(manager.state_dict())
    assert restored.state_dict() == manager.state_dict()


def test_gradient_geometry_detects_opposite_objectives_without_mutating_grads() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    geometry = gradient_cosine_matrix(
        {"aligned": parameter.sum(), "opposed": -parameter.sum()},
        (parameter,),
    )
    assert geometry.cosine[0, 1] < -0.999
    assert geometry.minimum_off_diagonal_cosine < -0.999
    assert parameter.grad is None


def test_phase1_curriculum_samples_one_training_horizon_and_all_validation_horizons() -> None:
    config = V09TrainingConfig(
        epochs=2,
        rollout_horizons=(2, 4),
        rollout_start_fractions=(0.0, 0.0),
        rollout_weights=(1.0, 0.5),
        lambda_rollout=1.0,
        lambda_physics=0.1,
        physics_start_fraction=0.0,
        physics_horizon=2,
        phase1_enabled=True,
        observable_names=("divergence", "boundary"),
        observable_component_weights=(1.0, 1.0),
        observable_horizons=(2, 4, 8),
        observable_horizon_weights=(0.2, 0.3, 0.5),
        observable_horizon_probabilities=(0.25, 0.25, 0.5),
    )
    training = curriculum_state(config, 1)
    validation = curriculum_state(config, 1, validation=True)
    assert len(training.observable_horizons) == 1
    assert validation.observable_horizons == (2, 4, 8)
    assert validation.observable_weights == (0.2, 0.3, 0.5)


def test_phase1_bundle_exposes_differentiable_constraints() -> None:
    payload = load_config("gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml").to_dict()
    payload["v0_9_adaptive"].update({"rank": 2, "rank_candidates": [1, 2], "width": 8})
    payload["v0_9_training"].update(
        {
            "epochs": 2,
            "rollout_horizons": [2, 4],
            "rollout_start_fractions": [0.0, 0.0],
            "rollout_weights": [1.0, 0.5],
            "physics_start_fraction": 0.0,
            "physics_ramp_duration_fraction": 0.0,
            "physics_horizon": 2,
            "observable_names": ["velocity", "divergence", "boundary"],
            "observable_component_weights": [1.0, 0.5, 0.2],
            "observable_horizons": [2, 4],
            "observable_horizon_weights": [0.4, 0.6],
            "observable_horizon_probabilities": [0.5, 0.5],
            "force_correlation_weight": 0.0,
            "force_spectrum_weight": 0.0,
        }
    )
    config = ProjectConfig.from_dict(payload)

    class FakePhysical:
        def target_batch(self, trajectory_ids, target_indices, horizon, limit):
            return torch.zeros(limit, 6), {}

        def loss(self, predicted, target, metadata):
            base = (predicted - target).square().mean()
            return ObservableLossResult(
                base,
                {
                    "observable_velocity": base,
                    "observable_divergence": 0.5 * base,
                    "observable_boundary": 0.25 * base,
                },
            )

    manager = InequalityAugmentedLagrangian(
        ("divergence", "boundary", "burden"),
        initial_penalty=1.0,
        penalty_growth=2.0,
        maximum_penalty=10.0,
    )
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
        _model(),
        batch,
        torch.ones(6),
        torch.zeros(2),
        torch.ones(2),
        config,
        epoch=1,
        physical=FakePhysical(),  # type: ignore[arg-type]
        augmented_lagrangian=manager,
        validation=True,
    )
    assert all(f"constraint_{name}" in terms for name in manager.names)
    assert torch.isfinite(total)
    total.backward()


def test_error_attribution_preserves_increment_signs() -> None:
    rows = observable_error_attribution(
        {
            "data": {"divergence": 0.1},
            "reconstruction": {"divergence": 0.3},
            "nominal": {"divergence": 0.5},
            "adaptive": {"divergence": 0.4},
        }
    )
    assert rows[0]["representation_increment"] == pytest.approx(0.2)
    assert rows[0]["adaptive_dynamics_increment"] < 0
