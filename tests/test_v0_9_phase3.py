from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from jka_model.adaptive import AdaptiveCache, AdaptiveTrajectory, FactorizedAdaptiveOperator
from jka_model.config import ProjectConfig, V09Phase3Config, load_config
from jka_model.data import TrajectoryDataset, TrajectoryRecord
from jka_model.manifold import (
    MatchedRouteContract,
    MatureCheckpointTracker,
    StreamFunctionPhysicalDecoder2D,
    assert_online_reencoding_required,
    centered_linear_cka,
    central_difference_2d,
    classify_matched_phase3_run,
    classify_phase3_joint_run,
    classify_phase3_metrics,
    classify_phase3_route,
    configure_phase3_route,
    decoded_physical_supervision,
    nested_route_support,
    orthogonal_procrustes_nrmse,
    phase3_checkpoint_key,
    physical_manifold_metrics,
    representation_effective_rank,
)


def test_phase3_config_roundtrip_and_frozen_phase2_dependency() -> None:
    source = load_config("gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml")
    assert source.v0_9_phase3 == V09Phase3Config(
        enabled=True,
        source_phase2_result="v09-added-p2-physical-20260824T105209Z",
    )
    assert ProjectConfig.from_dict(source.to_dict()).stable_hash == source.stable_hash
    payload = source.to_dict()
    payload["v0_9_phase2"]["enabled"] = False
    try:
        ProjectConfig.from_dict(payload)
    except ValueError as error:
        assert "Phase-3 requires" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Phase-3 must not detach from its Phase-2 evidence contract")


def test_streamfunction_decoder_is_differentiable_and_interior_divergence_free() -> None:
    torch.manual_seed(7)
    model = StreamFunctionPhysicalDecoder2D(4, 12, 10, hidden_dim=8, dx=0.2, dy=0.3)
    latent = torch.randn(2, 4, requires_grad=True)
    valid = torch.ones(12, 10, dtype=torch.bool)
    field = model(latent, valid_mask=valid, inlet_velocity=torch.ones(2))
    divergence = central_difference_2d(field[:, 0], 0.2, -2) + central_difference_2d(
        field[:, 1], 0.3, -1
    )
    # Boundary lifting changes boundary stencils; the curl identity is audited in the interior.
    assert float(divergence[:, 2:-2, 2:-2].detach().abs().max()) < 2.0e-5
    assert torch.equal(field[:, 0, 0], torch.ones_like(field[:, 0, 0]))
    assert torch.equal(field[:, 1, 0], torch.zeros_like(field[:, 1, 0]))
    field.square().mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()


def test_physical_metrics_enforce_cylinder_no_slip() -> None:
    field = torch.zeros(1, 3, 8, 6)
    valid = torch.ones(8, 6, dtype=torch.bool)
    valid[3:5, 2:4] = False
    clean = physical_manifold_metrics(field, valid_mask=valid, dx=1.0, dy=1.0)
    assert clean["divergence_rms"] == 0
    assert clean["boundary_no_slip_mse"] == 0
    field[:, 0, 3:5, 2:4] = 1.0
    dirty = physical_manifold_metrics(field, valid_mask=valid, dx=1.0, dy=1.0)
    assert dirty["boundary_no_slip_mse"] > 0


def test_decoded_physical_supervision_is_dimensionless_and_differentiable() -> None:
    torch.manual_seed(11)
    target = torch.randn(2, 3, 10, 8)
    predicted = (target + 0.1 * torch.randn_like(target)).requires_grad_(True)
    terms = decoded_physical_supervision(
        predicted,
        target,
        torch.ones(2, 10, 8, dtype=torch.bool),
        dx=0.2,
        dy=0.3,
    )
    loss = terms["field"] + terms["velocity"] + terms["vorticity"]
    assert float(loss.detach()) > 0
    loss.backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    exact = decoded_physical_supervision(
        target,
        target,
        torch.ones(10, 8, dtype=torch.bool),
        dx=0.2,
        dy=0.3,
    )
    assert exact["field"] == 0 and exact["velocity"] == 0 and exact["vorticity"] == 0


def test_trainable_phase3_routes_reject_frozen_latent_cache() -> None:
    assert_online_reencoding_required("frozen", uses_frozen_latent_cache=True)
    for route in ("joint", "from_scratch"):
        try:
            assert_online_reencoding_required(route, uses_frozen_latent_cache=True)
        except ValueError as error:
            assert "online re-encoding" in str(error)
        else:  # pragma: no cover
            raise AssertionError("trainable representation route accepted stale latents")
        assert_online_reencoding_required(route, uses_frozen_latent_cache=False)


def test_phase3_route_ownership_and_matched_contract() -> None:
    modules = [nn.Linear(2, 2) for _ in range(4)]
    declaration = configure_phase3_route(
        "frozen",
        backbone=modules[0],
        context_encoder=modules[1],
        operator=modules[2],
        physical_decoder=modules[3],
    )
    assert declaration["backbone_frozen"]
    assert not any(
        parameter.requires_grad for module in modules for parameter in module.parameters()
    )
    reference = MatchedRouteContract("split", 47, 701, 80, ("a", "b"), "gates")
    reference.assert_matched(MatchedRouteContract("split", 47, 701, 80, ("a", "b"), "gates"))
    try:
        reference.assert_matched(
            MatchedRouteContract("split", 47, 809, 80, ("a", "b"), "gates")
        )
    except ValueError as error:
        assert "matched" in str(error)
    else:  # pragma: no cover
        raise AssertionError("mismatched operator seed was accepted")


def test_coordinate_invariant_representation_diagnostics_ignore_latent_gauge() -> None:
    torch.manual_seed(19)
    reference = torch.randn(128, 8)
    orthogonal = torch.linalg.qr(torch.randn(8, 8)).Q
    candidate = 3.5 * (reference @ orthogonal) + torch.randn(8) * 4.0
    assert float(centered_linear_cka(candidate, reference)) > 0.99999
    assert float(orthogonal_procrustes_nrmse(candidate, reference)) < 1.0e-5
    assert float(representation_effective_rank(candidate)) > 1.0


def test_from_scratch_route_preserves_constructor_initialization() -> None:
    torch.manual_seed(23)
    backbone, context, operator = (nn.Linear(3, 3) for _ in range(3))
    original = backbone.weight.detach().clone()
    declaration = configure_phase3_route(
        "from_scratch",
        backbone=backbone,
        context_encoder=context,
        operator=operator,
        physical_decoder=None,
    )
    torch.testing.assert_close(backbone.weight, original)
    assert all(
        parameter.requires_grad
        for module in (backbone, context, operator)
        for parameter in module.parameters()
    )
    assert declaration["requires_online_reencoding"]


def test_from_scratch_operator_can_train_its_nominal_generator() -> None:
    config = load_config("gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml")
    assert config.v0_8_context and config.v0_9_adaptive and config.v0_9_phase2
    operator = FactorizedAdaptiveOperator(
        torch.zeros(16, 16),
        config.v0_8_context.context_dim,
        config.v0_9_adaptive,
        config.v0_9_phase2,
        trainable_nominal=True,
    )
    assert isinstance(operator.nominal_generator, nn.Parameter)
    assert operator.nominal_generator.requires_grad
    with torch.no_grad():
        operator.nominal_generator.copy_(torch.eye(16))
    projected = operator.project_trainable_nominal_stability_()
    assert projected <= 1.0e-6
    symmetric = 0.5 * (
        operator.nominal_generator + operator.nominal_generator.transpose(-1, -2)
    )
    assert float(torch.linalg.eigvalsh(symmetric.detach())[-1]) <= 1.0e-6


def test_phase3_route_classification_prefers_joint_when_physics_passes() -> None:
    audits = [
        {
            "reconstruction_physics_status": "PASS",
            "roundtrip_status": "FAIL",
            "tangent_status": "PASS",
        }
        for _ in range(3)
    ]
    decision = classify_phase3_route(
        audits,
        {
            "condition_observer": "NOT_SUPPORTED",
            "dynamic_operator_adaptation": "NOT_SUPPORTED",
        },
    )
    assert decision == {
        "reconstruction_physics_status": "PASS",
        "roundtrip_status": "FAIL",
        "nominal_tangent_status": "PASS",
        "next_candidate": "JOINT_MARKOV_REPRESENTATION",
    }


def test_phase3_divergence_rms_is_compared_to_sqrt_mse_threshold() -> None:
    # Representative values from the returned three-seed audit.  0.049 RMS is
    # below sqrt(0.02), even though it is above the MSE value 0.02 itself.
    status = classify_phase3_metrics(
        divergence_degradation=-0.239,
        boundary_degradation=0.043,
        reconstruction_divergence_rms=0.04946,
        reconstruction_boundary_mse=0.00216,
        reconstruction_outer_boundary_mse=0.02003,
        roundtrip_nrmse=0.4006,
        nominal_tangent_divergence=0.0281,
        max_divergence_mse=0.02,
        max_boundary_mse=0.05,
        max_reconstruction_physics_degradation=0.10,
        max_roundtrip_nrmse=0.25,
        max_tangent_divergence=0.10,
    )
    assert status == {
        "reconstruction_physics_status": "PASS",
        "roundtrip_status": "FAIL",
        "tangent_status": "PASS",
    }


def test_phase3_raw_field_windows_reencode_original_states() -> None:
    from jka_model.manifold import RawFieldAdaptiveRolloutDataset

    states = torch.arange(6 * 3 * 4 * 4, dtype=torch.float32).reshape(6, 3, 4, 4)
    dts = torch.full((5,), 0.1)
    record = TrajectoryRecord(
        "trajectory-a",
        states,
        dts,
        mu_static=torch.tensor([1.0]),
        valid_mask=torch.ones(4, 4, dtype=torch.bool),
    )
    trajectory = AdaptiveTrajectory(
        trajectory_id="trajectory-a",
        split="train",
        schedule_type="smooth_ramp",
        transition_index=2,
        latents=torch.zeros(6, 2),
        dts=dts,
        context_parameters=torch.tensor([1.0]),
        conditions=torch.stack((torch.arange(5), torch.ones(5)), dim=-1).float(),
        nominal_residuals=torch.zeros(5, 2),
    )
    cache = AdaptiveCache(
        trajectories=(trajectory,),
        backbone_checkpoint_sha256="backbone",
        backbone_config_hash="config",
        context_checkpoint_sha256="context",
        data_fingerprint="data",
        split_manifest={"train": ["trajectory-a"], "validation": [], "test": []},
        normalizer_state={},
        nominal_generator=torch.zeros(2, 2),
    )
    dataset = RawFieldAdaptiveRolloutDataset(
        cache,
        TrajectoryDataset([record]),
        "train",
        history=2,
        horizon=2,
        stride=1,
    )
    assert len(dataset) == 3
    first = dataset[0]
    assert torch.equal(first["history_raw"], states[:2])
    assert torch.equal(first["target_raw"], states[2:4])
    assert first["future_condition_targets"].shape == (2, 3)


def test_returned_phase3_audit_reassesses_to_joint_route() -> None:
    from gpu_validation.v0_9.scripts.gpu_validate_phase3_joint import _corrected_audit

    root = Path("gpu_validation/v0_9/results/v09-added-p3-audit-20260826T043840Z")
    decision = json.loads(
        (root / "evaluation/phase3_route_decision.json").read_text(encoding="utf-8")
    )
    phase2_summary = json.loads(
        Path(
            "gpu_validation/v0_9/results/"
            "v09-added-p2-physical-20260824T105209Z/summary.json"
        ).read_text(encoding="utf-8")
    )
    config = load_config("gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml")
    reassessed = _corrected_audit(decision, phase2_summary, config)
    assert all(
        row["reconstruction_physics_status"] == "PASS" for row in reassessed["seeds"]
    )
    assert reassessed["corrected"] == {
        "reconstruction_physics_status": "PASS",
        "roundtrip_status": "FAIL",
        "nominal_tangent_status": "PASS",
        "next_candidate": "JOINT_MARKOV_REPRESENTATION",
    }


def test_phase3_early_stopping_patience_starts_after_curriculum_maturity() -> None:
    tracker = MatureCheckpointTracker(earliest_epoch=49, patience=16)
    for epoch in range(1, 49):
        selected, stop = tracker.consider(epoch, (float(epoch),))
        assert not selected and not stop
        assert tracker.stale_epochs == 0
    selected, stop = tracker.consider(49, (1.0,))
    assert selected and not stop and tracker.best_epoch == 49
    for epoch in range(50, 65):
        selected, stop = tracker.consider(epoch, (2.0,))
        assert not selected and not stop
    selected, stop = tracker.consider(65, (2.0,))
    assert not selected and stop


def test_phase3_joint_representation_lr_waits_for_physical_supervision() -> None:
    from train.train_v0_9_phase3 import _phase3_learning_rate_scales

    assert _phase3_learning_rate_scales(
        epoch=0, epochs=80, representation_start=0.35, representation_ramp=0.25
    ) == (0.0, 1.0)
    representation, operator = _phase3_learning_rate_scales(
        epoch=32, epochs=80, representation_start=0.35, representation_ramp=0.25
    )
    assert 0.0 < representation < operator == 1.0
    assert _phase3_learning_rate_scales(
        epoch=79, epochs=80, representation_start=0.35, representation_ramp=0.25
    ) == (0.5, 0.5)


def test_phase3_checkpoint_selection_and_report_use_all_declared_gates() -> None:
    config = load_config("gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml")
    assert config.v0_9_phase3 and config.v0_9_phase2 and config.v0_9_evaluation
    passing = {
        "total": 10.0,
        "physical_manifold_violation": 0.0,
        "representation_drift": 0.09,
        "roundtrip": 0.24,
        "observer_normalized_rmse": 0.40,
        "observer_minimum_r2": 0.30,
        "representation_linear_cka": 0.90,
        "representation_procrustes_nrmse": 0.20,
        "representation_effective_rank": 4.0,
        "rollout_gain_h8": 0.021,
        "rollout_gain_h16": 0.022,
        "rollout_gain_h32": 0.023,
        "rollout_gain_h80": 0.024,
    }
    for horizon in config.v0_9_evaluation.rollout_horizons:
        passing[f"decoded_field_relative_l2_h{horizon}"] = 0.30
        passing[f"decoded_velocity_relative_l2_h{horizon}"] = 0.20
        passing[f"decoded_vorticity_relative_l2_h{horizon}"] = 0.40
    drifting = {**passing, "total": 1.0, "representation_drift": 0.11}
    passing_key = phase3_checkpoint_key(
        passing,
        config.v0_9_phase3,
        config.v0_9_evaluation,
        config.v0_9_phase2,
        condition_mode="latent_inferred",
    )
    drifting_key = phase3_checkpoint_key(
        drifting,
        config.v0_9_phase3,
        config.v0_9_evaluation,
        config.v0_9_phase2,
        condition_mode="latent_inferred",
    )
    assert passing_key < drifting_key
    decoded_worse = {
        **passing,
        **{
            f"decoded_field_relative_l2_h{horizon}": 0.35
            for horizon in config.v0_9_evaluation.rollout_horizons
        },
        "total": 0.1,
    }
    decoded_worse_key = phase3_checkpoint_key(
        decoded_worse,
        config.v0_9_phase3,
        config.v0_9_evaluation,
        config.v0_9_phase2,
        condition_mode="latent_inferred",
    )
    assert passing_key < decoded_worse_key
    gate = classify_phase3_joint_run(
        passing,
        config.v0_9_phase3,
        config.v0_9_evaluation,
        config.v0_9_phase2,
        condition_mode="latent_inferred",
    )
    assert all(gate.values())
    no_long_skill = {**passing, "rollout_gain_h80": -0.001}
    failed = classify_phase3_joint_run(
        no_long_skill,
        config.v0_9_phase3,
        config.v0_9_evaluation,
        config.v0_9_phase2,
        condition_mode="latent_inferred",
    )
    assert not failed["predictive"] and not failed["strict_joint"]


def test_returned_joint_result_has_zero_strict_passes_under_complete_contract() -> None:
    config = load_config("gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml")
    assert config.v0_9_phase3 and config.v0_9_phase2 and config.v0_9_evaluation
    result = json.loads(
        Path(
            "gpu_validation/v0_9/results/"
            "v09-added-p3-joint-20260826T053347Z/evaluation/joint_summary.json"
        ).read_text(encoding="utf-8")
    )
    gates = [
        classify_phase3_joint_run(
            row["locked_test"],
            config.v0_9_phase3,
            config.v0_9_evaluation,
            config.v0_9_phase2,
            condition_mode=row["condition_mode"],
        )
        for row in result["runs"]
    ]
    assert sum(gate["physics"] for gate in gates) == 18
    assert sum(gate["representation_drift"] for gate in gates) == 8
    assert sum(gate["roundtrip"] for gate in gates) == 12
    assert sum(gate["representation_feasible"] for gate in gates) == 4
    assert sum(gate["predictive"] for gate in gates) == 0
    assert sum(gate["strict_joint"] for gate in gates) == 0


def test_matched_route_requires_decoded_physical_gain_not_only_latent_gain() -> None:
    config = load_config("gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml")
    assert config.v0_9_phase3 and config.v0_9_phase2 and config.v0_9_evaluation
    frozen: dict[str, float] = {}
    candidate: dict[str, float] = {
        "total": 1.0,
        "physical_manifold_violation": 0.0,
        "representation_drift": 5.0,
        "roundtrip": 0.10,
        "observer_normalized_rmse": 0.40,
        "observer_minimum_r2": 0.30,
        "representation_linear_cka": 0.5,
        "representation_procrustes_nrmse": 0.7,
        "representation_effective_rank": 6.0,
    }
    for horizon in config.v0_9_evaluation.rollout_horizons:
        candidate[f"rollout_gain_h{horizon}"] = 0.10
        for quantity in ("field", "velocity", "vorticity"):
            name = f"decoded_{quantity}_relative_l2_h{horizon}"
            frozen[name] = 1.0
            candidate[name] = 0.90
    passing = classify_matched_phase3_run(
        candidate,
        frozen,
        config.v0_9_phase3,
        config.v0_9_evaluation,
        config.v0_9_phase2,
        route="from_scratch",
        condition_mode="latent_inferred",
    )
    assert passing["matched_route_pass"]
    candidate["decoded_vorticity_relative_l2_h80"] = 1.01
    failed = classify_matched_phase3_run(
        candidate,
        frozen,
        config.v0_9_phase3,
        config.v0_9_evaluation,
        config.v0_9_phase2,
        route="from_scratch",
        condition_mode="latent_inferred",
    )
    assert not failed["decoded_vorticity_noninferiority"]
    assert not failed["matched_route_pass"]


def test_nested_route_support_requires_both_condition_modes_per_seed() -> None:
    rows = []
    for seed in (47, 53, 59):
        for mode in ("known", "latent_inferred"):
            for operator_seed in (701, 809, 907):
                passed = not (seed == 59 or (mode == "latent_inferred" and operator_seed == 907))
                rows.append(
                    {
                        "seed": seed,
                        "condition_mode": mode,
                        "operator_seed": operator_seed,
                        "gates": {"matched_route_pass": passed},
                    }
                )
    result = nested_route_support(rows, required_fraction=2.0 / 3.0)
    assert result["backbone_pass_fraction"] == 2.0 / 3.0
    assert result["supported"]


def test_phase3_matched_workflow_builds_explicit_from_scratch_config(tmp_path: Path) -> None:
    from gpu_validation.v0_9.scripts.gpu_validate_phase3_joint import _resolved_config
    from gpu_validation.v0_9.scripts.gpu_validate_phase3_routes import _aggregate

    config = _resolved_config(
        Path("gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml"),
        phase2_id="phase2-source",
        dataset_path=tmp_path / "data.pt",
        run_root=tmp_path / "run",
        seed=47,
        condition_mode="latent_inferred",
        operator_seed=701,
        route="from_scratch",
    )
    assert "phase3-from_scratch" in config.tags
    assert config.v0_9_adaptive and config.v0_9_adaptive.condition_mode == "latent_inferred"
    assert config.v0_9_phase3 and config.v0_9_phase3.source_phase2_result == "phase2-source"
    aggregate = _aggregate(
        [
            {"locked_test": {"common": 1.0, "first_only": 2.0}},
            {"locked_test": {"common": 3.0, "second_only": 4.0}},
        ]
    )
    assert aggregate == {"common": 2.0}


def test_from_scratch_residual_scale_uses_only_training_trajectories() -> None:
    from train.train_v0_9_phase3 import _from_scratch_residual_scale

    class IdentityNormalizer:
        @staticmethod
        def transform(value: torch.Tensor) -> torch.Tensor:
            return value

    class TinyBackbone:
        @staticmethod
        def encode_target(value: torch.Tensor) -> torch.Tensor:
            flattened = value.flatten(1)
            return torch.stack((flattened.mean(dim=1), flattened[:, 0]), dim=-1)

    adaptive = SimpleNamespace(
        operator_adapter=SimpleNamespace(nominal_generator=torch.zeros(2, 2))
    )
    train_states = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 1, 2, 2)
    excluded_states = torch.full((5, 1, 2, 2), 1.0e9)
    records = [
        SimpleNamespace(
            trajectory_id="train",
            states_raw=train_states,
            dts=torch.ones(4),
        ),
        SimpleNamespace(
            trajectory_id="test",
            states_raw=excluded_states,
            dts=torch.ones(4),
        ),
    ]
    scale = _from_scratch_residual_scale(
        TinyBackbone(),
        adaptive,
        IdentityNormalizer(),
        records,
        {"train"},
        torch.device("cpu"),
    )
    expected_latents = TinyBackbone.encode_target(train_states)
    expected = (expected_latents[1:] - expected_latents[:-1]).square().mean(dim=0).sqrt()
    torch.testing.assert_close(scale, expected)
