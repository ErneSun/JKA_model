from __future__ import annotations

import torch

from jka_model.adaptive import (
    AdaptiveKoopmanModel,
    FactorizedAdaptiveOperator,
    adaptive_latent_rollout,
    adaptive_stabilization_objective,
    condition_observer_metrics,
    condition_targets,
    conditional_centering_loss,
    curriculum_state,
    matched_history_pairs,
    phase2_training_state,
)
from jka_model.config import (
    CylinderWake2DConfig,
    V08ContextConfig,
    V09AdaptiveConfig,
    V09ConditionConfig,
    V09Phase2Config,
    V09TrainingConfig,
)
from jka_model.context import build_dynamic_context_model
from jka_model.data import cylinder_condition_schedule
from jka_model.evaluation import (
    GateStatus,
    MetricDirection,
    MetricGateSpec,
    evaluate_metric_gate,
)


def _operator(mode: str = "latent_inferred") -> FactorizedAdaptiveOperator:
    return FactorizedAdaptiveOperator(
        -0.02 * torch.eye(10),
        8,
        V09AdaptiveConfig(
            condition_mode=mode,
            rank=4,
            rank_candidates=(2, 4),
            width=12,
            bounded_coordinates=True,
            trust_gate=True,
        ),
        V09Phase2Config(static_rank=2, dynamic_rank=2, observer_width=12),
    )


def test_factorized_operator_exactly_starts_at_a0_and_remains_differentiable() -> None:
    adapter = _operator()
    context = torch.randn(5, 8)
    z = torch.randn(5, 10)
    dt = torch.full((5, 1), 0.15)
    prediction, coordinates, delta, adapted = adapter.step(z, context, dt)
    expected = torch.einsum(
        "bij,bj->bi",
        torch.linalg.matrix_exp(adapter.nominal_generator.unsqueeze(0) * dt[:, :, None]),
        z,
    )
    assert torch.equal(coordinates, torch.zeros_like(coordinates))
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(adapted, adapter.nominal_generator.unsqueeze(0).expand_as(adapted))
    assert float(adapter.cross_basis_orthogonality_loss().detach()) < 1.0e-12
    assert torch.allclose(prediction, expected, atol=1e-7, rtol=1e-6)
    prediction.square().mean().backward()
    assert adapter.dynamic_coordinate_head[-1].weight.grad is not None


def test_phase2_known_and_latent_condition_visibility_is_strict() -> None:
    context = torch.randn(3, 8)
    known = _operator("known")
    latent = _operator("latent_inferred")
    try:
        known.phase2_components(context)
    except ValueError as error:
        assert "requires normalized" in str(error)
    else:
        raise AssertionError("known Phase-2 adapter accepted a missing condition")
    try:
        latent.phase2_components(context, torch.randn(3, 3))
    except ValueError as error:
        assert "forbids" in str(error)
    else:
        raise AssertionError("latent Phase-2 adapter leaked the condition label")


def test_condition_branch_is_invariant_to_history_after_condition_is_fixed() -> None:
    adapter = _operator("known")
    with torch.no_grad():
        adapter.static_coordinate_head[-1].bias.fill_(0.2)
    condition = torch.randn(3, 3)
    first = adapter.phase2_components(torch.randn(3, 8), condition)
    second = adapter.phase2_components(torch.randn(3, 8), condition)
    assert torch.equal(first["static_coordinates"], second["static_coordinates"])
    assert torch.equal(first["static_delta"], second["static_delta"])


def test_latent_condition_is_bounded_detached_and_generator_growth_limited() -> None:
    adapter = _operator("latent_inferred")
    with torch.no_grad():
        adapter.static_coordinate_head[-1].bias.fill_(100.0)
        adapter.dynamic_coordinate_head[-1].bias.fill_(100.0)
    components = adapter.phase2_components(1.0e6 * torch.randn(4, 8))
    assert torch.isfinite(components["q_hat"]).all()
    assert float(components["q_hat"].detach().abs().max()) <= adapter.observer_output_limit
    assert components["q_hat"].requires_grad
    assert not components["q_used"].requires_grad
    delta = components["delta"]
    assert torch.allclose(delta, components["static_delta"] + components["dynamic_delta"])
    symmetric = 0.5 * (delta + delta.transpose(-1, -2))
    burden = torch.linalg.matrix_norm(symmetric, ord="fro")
    assert float(burden.detach().max()) <= adapter.symmetric_delta_budget + 1.0e-6


def test_phase2_continuous_stage_contract_and_oracle_boundary() -> None:
    phase2 = V09Phase2Config(
        enabled=True,
        static_rank=2,
        dynamic_rank=2,
        observer_width=12,
    )
    static = phase2_training_state(
        phase2,
        epoch=0,
        epochs=100,
        condition_mode="latent_inferred",
        observer_ready=False,
    )
    dynamic = phase2_training_state(
        phase2,
        epoch=40,
        epochs=100,
        condition_mode="latent_inferred",
        observer_ready=False,
    )
    observer = phase2_training_state(
        phase2,
        epoch=80,
        epochs=100,
        condition_mode="latent_inferred",
        observer_ready=False,
    )
    joint = phase2_training_state(
        phase2,
        epoch=80,
        epochs=100,
        condition_mode="latent_inferred",
        observer_ready=True,
    )
    assert static.name == "static_oracle" and static.active_components == "static"
    assert static.use_oracle_condition and static.observer_weight == 0.0
    assert dynamic.name == "dynamic_residual_oracle" and dynamic.detach_static
    assert observer.name == "observer_calibration" and observer.observer_only
    assert joint.name == "latent_joint_refinement" and not joint.use_oracle_condition
    assert (
        static.delta_budget
        < dynamic.delta_budget
        <= observer.delta_budget
        == phase2.symmetric_delta_budget
    )


def test_phase2_dynamic_residual_freezes_static_branch_and_uses_total_projection() -> None:
    adapter = _operator("latent_inferred")
    context = torch.randn(4, 8)
    oracle = torch.randn(4, 3)
    with torch.no_grad():
        adapter.static_coordinate_head[-1].bias.fill_(0.25)
        adapter.dynamic_coordinate_head[-1].bias.fill_(0.25)
    components = adapter.phase2_components(
        context,
        condition_override=oracle,
        detach_static=True,
        delta_budget=0.10,
    )
    assert torch.equal(components["q_used"], oracle)
    components["delta"].square().mean().backward()
    assert adapter.static_coordinate_head[-1].weight.grad is None
    assert adapter.static_left_factor.grad is None
    assert adapter.dynamic_coordinate_head[-1].weight.grad is not None
    symmetric = 0.5 * (components["delta"] + components["delta"].transpose(-1, -2))
    assert float(torch.linalg.matrix_norm(symmetric, ord="fro").detach().max()) <= 0.100001


def test_condition_rate_is_causal_and_centering_has_no_variance_floor() -> None:
    conditions = torch.tensor([[80.0, 0.8], [80.0, 0.8], [100.0, 1.0]])
    targets = condition_targets(conditions, torch.tensor([0.5, 0.5, 0.5]))
    assert torch.equal(targets[:, 2], torch.tensor([0.0, 0.0, 40.0]))
    zero = torch.zeros(6, 2, requires_grad=True)
    loss = conditional_centering_loss(zero, torch.randn(6, 3), bandwidth=1.0)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert zero.grad is not None


def test_matched_pair_audit_requires_same_present_and_different_history() -> None:
    pairs = matched_history_pairs(
        torch.tensor([[0.0], [0.01], [2.0]]),
        torch.tensor([[0.0], [0.01], [3.0]]),
        torch.tensor([[[0.0]], [[1.0]], [[2.0]]]),
        torch.tensor([[[0.0]], [[0.7]], [[0.1]]]),
        condition_tolerance=0.1,
        latent_tolerance=0.1,
        minimum_history_separation=0.5,
        minimum_future_separation=0.5,
        group_ids=("route-a", "route-b", "route-c"),
    )
    assert len(pairs) == 1
    assert (pairs[0].first, pairs[0].second) == (0, 1)


def test_extended_schedule_families_cover_up_down_and_cycle() -> None:
    cylinder = CylinderWake2DConfig(
        num_trajectories=12,
        num_steps=80,
        nx=48,
        ny=24,
        x_min=-4.0,
        x_max=8.0,
        y_min=-3.0,
        y_max=3.0,
        cylinder_diameter=1.0,
        reynolds_number=100.0,
        lattice_inflow_velocity=0.08,
        solver_steps_per_snapshot=1,
        time_varying_boundary=True,
    )
    condition = V09ConditionConfig(schedule_types=V09Phase2Config().schedule_variants)
    up = cylinder_condition_schedule(cylinder, condition, "smooth_up_fast")
    down = cylinder_condition_schedule(cylinder, condition, "smooth_down_slow")
    cycle = cylinder_condition_schedule(cylinder, condition, "cyclic_fast_short")
    cycle_long = cylinder_condition_schedule(cylinder, condition, "cyclic_slow_long")
    assert float(up["reynolds_number"][0]) < float(up["reynolds_number"][-1])
    assert float(down["reynolds_number"][0]) > float(down["reynolds_number"][-1])
    assert float(cycle["reynolds_number"].max()) > float(cycle["reynolds_number"][0])
    assert abs(float(cycle["reynolds_number"][-1] - cycle["reynolds_number"][0])) < 1e-5
    assert cycle_long["dwell_steps"] > cycle["dwell_steps"] > 0


def test_condition_observer_metrics_are_component_auditable() -> None:
    target = torch.randn(20, 3)
    metrics = condition_observer_metrics(target.clone(), target)
    assert metrics["normalized_rmse"] == 0.0
    assert metrics["minimum_r2"] == 1.0
    assert "condition_rate_r2" in metrics


def test_phase2_rollout_objective_connects_observer_and_factorized_operator() -> None:
    context_config = V08ContextConfig(
        family="history_mlp", context_dim=8, history_length=3, width=12, heads=1
    )
    context = build_dynamic_context_model(
        context_config,
        family="history_mlp",
        latent_dim=10,
        parameter_dim=3,
        history=3,
    ).context_encoder
    model = AdaptiveKoopmanModel(context, _operator())
    training = V09TrainingConfig(
        epochs=2,
        rollout_horizons=(2, 4),
        rollout_start_fractions=(0.0, 0.0),
        rollout_weights=(1.0, 0.5),
        lambda_rollout=1.0,
    )
    batch = {
        "history_z": torch.randn(4, 3, 10),
        "history_dts": torch.full((4, 2), 0.15),
        "future_dts": torch.full((4, 4), 0.15),
        "future_conditions": torch.randn(4, 4, 2),
        "future_condition_targets": torch.randn(4, 4, 3),
        "target_latents": torch.randn(4, 4, 10),
        "context_parameters": torch.randn(4, 3),
    }
    result = adaptive_stabilization_objective(
        model,
        batch,
        torch.ones(10),
        torch.zeros(3),
        torch.ones(3),
        training,
        "latent_inferred",
        curriculum_state(training, 1, validation=True),
        torch.ones(4, dtype=torch.bool),
        V09Phase2Config(enabled=True, static_rank=2, dynamic_rank=2, observer_width=12),
    )
    assert torch.isfinite(result.total)
    assert set(("condition_observer", "condition_centering", "basis_cross_orthogonality")).issubset(
        result.terms
    )
    result.total.backward()
    assert model.operator_adapter.condition_observer[-1].weight.grad is not None


def test_growth_trust_region_keeps_h80_closed_loop_finite() -> None:
    context_config = V08ContextConfig(
        family="history_mlp", context_dim=8, history_length=3, width=12, heads=1
    )
    context = build_dynamic_context_model(
        context_config,
        family="history_mlp",
        latent_dim=10,
        parameter_dim=3,
        history=3,
    ).context_encoder
    adapter = _operator("latent_inferred")
    with torch.no_grad():
        adapter.static_coordinate_head[-1].bias.fill_(100.0)
        adapter.dynamic_coordinate_head[-1].bias.fill_(100.0)
    model = AdaptiveKoopmanModel(context, adapter).eval()
    result = adaptive_latent_rollout(
        model,
        torch.randn(2, 3, 10),
        torch.full((2, 2), 0.15),
        torch.full((2, 80), 0.15),
        torch.randn(2, 3),
        None,
    )
    assert torch.isfinite(result["adapted"]).all()
    assert float(result["adapted"].abs().max()) < 100.0


def test_gate_accepts_float32_boundary_noise_but_not_a_material_miss() -> None:
    spec = MetricGateSpec("frequency", MetricDirection.LOWER_IS_BETTER, threshold=0.1)
    boundary = evaluate_metric_gate(0.100000005, spec)
    material = evaluate_metric_gate(0.10001, spec)
    assert boundary.status is GateStatus.PASS
    assert material.status is GateStatus.FAIL
    assert boundary.details["comparison_tolerance"] > 0
