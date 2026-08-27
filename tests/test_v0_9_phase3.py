from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from jka_model.adaptive import AdaptiveCache, AdaptiveTrajectory
from jka_model.config import ProjectConfig, V09Phase3Config, load_config
from jka_model.data import TrajectoryDataset, TrajectoryRecord
from jka_model.manifold import (
    MatchedRouteContract,
    MatureCheckpointTracker,
    StreamFunctionPhysicalDecoder2D,
    assert_online_reencoding_required,
    central_difference_2d,
    classify_phase3_joint_run,
    classify_phase3_metrics,
    classify_phase3_route,
    configure_phase3_route,
    phase3_checkpoint_key,
    physical_manifold_metrics,
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
        "rollout_gain_h8": 0.021,
        "rollout_gain_h16": 0.022,
        "rollout_gain_h32": 0.023,
        "rollout_gain_h80": 0.024,
    }
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
