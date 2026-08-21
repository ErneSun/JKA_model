from __future__ import annotations

from dataclasses import replace

import torch

from jka_model.config import CylinderWake2DConfig, V09ConditionConfig
from jka_model.data import (
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
