from __future__ import annotations

from dataclasses import replace

import torch

from jka_model.adaptive import FrozenCylinderPhysics
from jka_model.config import (
    CylinderWake2DConfig,
    V09ConditionConfig,
    load_config,
)
from jka_model.data import (
    ChannelStandardizer,
    cylinder_condition_schedule,
    generate_v0_9_cylinder_wake_trajectories,
    validate_v0_9_cylinder_wake_dataset,
)


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
