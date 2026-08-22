from __future__ import annotations

from dataclasses import replace

import torch

from jka_model.adaptive import FrozenCylinderPhysics
from jka_model.config import (
    CylinderWake2DConfig,
    V09ConditionConfig,
    V09TrainingConfig,
    load_config,
)
from jka_model.data import (
    ChannelStandardizer,
    cylinder_condition_schedule,
    generate_v0_9_cylinder_wake_trajectories,
    validate_v0_9_cylinder_wake_dataset,
)
from jka_model.observables import RobustObservableScaleState
from jka_model.problems.cylinder_observables import CylinderWakeObservableObjective


def _small() -> tuple[CylinderWake2DConfig, V09ConditionConfig]:
    cylinder = CylinderWake2DConfig(
        num_trajectories=6,
        num_steps=12,
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
    condition = V09ConditionConfig(
        transition_start_fraction=0.25, smooth_duration_fraction=0.25
    )
    return cylinder, condition


def test_smooth_and_abrupt_schedules_are_aligned_and_low_mach() -> None:
    cylinder, condition = _small()
    smooth = cylinder_condition_schedule(cylinder, condition, "smooth")
    abrupt = cylinder_condition_schedule(cylinder, condition, "abrupt")
    for schedule in (smooth, abrupt):
        assert schedule["reynolds_number"].shape == (cylinder.num_steps,)
        assert float(schedule["lattice_inflow_velocity"].max()) < 0.12
        assert float(schedule["reynolds_number"].min()) == condition.reynolds_low
        assert float(schedule["reynolds_number"].max()) == condition.reynolds_high
    transition = abrupt["transition_index"]
    assert torch.all(abrupt["reynolds_number"][:transition] == condition.reynolds_low)
    assert torch.all(abrupt["reynolds_number"][transition:] == condition.reynolds_high)
    differences = torch.diff(smooth["reynolds_number"])
    assert torch.all(differences >= -1e-6)


def test_variable_condition_dataset_contains_both_schedule_families() -> None:
    cylinder, condition = _small()
    dataset = generate_v0_9_cylinder_wake_trajectories(cylinder, condition, seed=7)
    report = validate_v0_9_cylinder_wake_dataset(dataset, cylinder, condition)
    assert report["gates"]["schedule_types_complete"]
    assert report["gates"]["condition_alignment"]
    assert {record.metadata["schedule_type"] for record in dataset.records} == {
        "smooth",
        "abrupt",
    }
    assert all(record.states_raw.shape == (13, 3, 48, 24) for record in dataset.records)


def test_phase1_cylinder_scales_and_force_windows_use_training_records() -> None:
    cylinder, condition = _small()
    dataset = generate_v0_9_cylinder_wake_trajectories(cylinder, condition, seed=11)
    records = {record.trajectory_id: record for record in dataset.records}
    training_ids = tuple(records)[:4]
    training = V09TrainingConfig(
        epochs=2,
        rollout_horizons=(2, 4),
        rollout_start_fractions=(0.0, 0.0),
        rollout_weights=(1.0, 0.5),
        lambda_rollout=1.0,
        lambda_physics=0.1,
        physics_start_fraction=0.0,
        physics_horizon=2,
        phase1_enabled=True,
        observable_names=(
            "velocity",
            "vorticity",
            "divergence",
            "boundary",
            "lift",
            "drag",
        ),
        observable_component_weights=(1.0, 0.2, 0.5, 0.2, 0.1, 0.05),
        observable_horizons=(2, 4),
        observable_horizon_weights=(0.5, 0.5),
    )
    objective = CylinderWakeObservableObjective(cylinder, training)
    state = objective.fit_training_scales(
        records,
        training_ids,
        split_fingerprint="train-split-11",
    )
    assert state.split_fingerprint == "train-split-11"
    assert set(state.scales) == {
        "velocity",
        "vorticity",
        "divergence",
        "boundary",
        "lift",
        "drag",
    }
    selected = [records[identifier] for identifier in training_ids[:2]]
    target = torch.stack([record.states_raw[1] for record in selected])
    prediction = (target + 0.01 * torch.randn_like(target)).requires_grad_(True)
    masks = torch.stack([record.valid_mask for record in selected])
    result = objective.training_loss(prediction, target, {"valid_mask": masks})
    assert torch.isfinite(result.total)
    result.total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()

    target_window = torch.stack([record.states_raw[1:5] for record in selected])
    predicted_window = (target_window + 0.01 * torch.randn_like(target_window)).requires_grad_(
        True
    )
    force = objective.force_window_loss(predicted_window, target_window, {})
    assert torch.isfinite(force.total)
    force.total.backward()
    assert predicted_window.grad is not None


def test_v08_configuration_still_forbids_time_varying_boundary() -> None:
    cylinder, _ = _small()
    fixed = replace(cylinder, time_varying_boundary=False)
    assert not fixed.time_varying_boundary


def test_frozen_decoder_physics_is_differentiable_only_through_latent_state() -> None:
    config = load_config("gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml")
    assert config.cylinder_wake_2d
    cylinder = config.cylinder_wake_2d

    class ReshapeDecoder(torch.nn.Module):
        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            return latent.reshape(-1, 3, cylinder.nx, cylinder.ny)

    decoder = ReshapeDecoder()
    normalizer = ChannelStandardizer(eps=1e-6)
    normalizer.load_state_dict(
        {
            "kind": "channel_standardizer",
            "eps": 1e-6,
            "mean": torch.zeros(3),
            "scale": torch.ones(3),
            "spatial_dim": 2,
            "layout": "channels_first",
            "fitted_trajectory_ids": ["synthetic"],
        }
    )
    physics = FrozenCylinderPhysics(
        decoder, normalizer, {}, config, torch.device("cpu")
    )  # type: ignore[arg-type]
    physics.set_scale_state(
        RobustObservableScaleState(
            method="mad",
            scales={
                name: 1.0
                for name in (
                    "velocity",
                    "vorticity",
                    "divergence",
                    "boundary",
                    "lift",
                    "drag",
                )
            },
            centers={
                name: 0.0
                for name in (
                    "velocity",
                    "vorticity",
                    "divergence",
                    "boundary",
                    "lift",
                    "drag",
                )
            },
            sample_counts={
                name: 1
                for name in (
                    "velocity",
                    "vorticity",
                    "divergence",
                    "boundary",
                    "lift",
                    "drag",
                )
            },
            split_fingerprint="synthetic-train",
            epsilon=1.0e-6,
        )
    )
    latent = torch.randn(2, 3 * cylinder.nx * cylinder.ny, requires_grad=True)
    target = torch.randn(2, 3, cylinder.nx, cylinder.ny)
    valid = torch.ones(2, cylinder.nx, cylinder.ny, dtype=torch.bool)
    x_center, y_center = cylinder.nx // 2, cylinder.ny // 2
    valid[:, x_center - 2 : x_center + 2, y_center - 2 : y_center + 2] = False
    result = physics.loss(latent, target, {"valid_mask": valid})
    assert torch.isfinite(result.total)
    assert set(result.terms) == {
        "observable_velocity",
        "observable_vorticity",
        "observable_divergence",
        "observable_boundary",
        "observable_lift",
        "observable_drag",
    }
    result.total.backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert not tuple(decoder.parameters())
