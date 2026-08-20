from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from jka_model.config import load_config
from jka_model.data import (
    ChannelStandardizer,
    D2Q9CylinderWakeSolver,
    TrajectoryWindowDataset,
    collate_problem_batches,
    cylinder_solid_mask,
    generate_cylinder_wake_2d_trajectories,
    load_cylinder_wake_dataset,
    make_split_manifest,
    save_cylinder_wake_dataset,
    select_split,
    validate_cylinder_wake_dataset,
    validate_trajectories_against_spec,
)
from jka_model.losses import compute_field_jepa_loss
from jka_model.problems import create_problem_adapter
from train.train_v0_6 import initialize_v0_6_model


@pytest.fixture(scope="module")
def cylinder_case():
    config = load_config("configs/v0_8/cylinder_wake_cpu_smoke.yaml")
    assert config.cylinder_wake_2d is not None
    dataset = generate_cylinder_wake_2d_trajectories(
        config.cylinder_wake_2d, seed=config.training.seed
    )
    return config, dataset


def test_cylinder_geometry_mask_and_fixed_boundary(cylinder_case) -> None:
    config, _ = cylinder_case
    cylinder = config.cylinder_wake_2d
    assert cylinder is not None
    mask = cylinder_solid_mask(cylinder)
    assert mask.shape == (cylinder.nx, cylinder.ny)
    assert 0 < int(mask.sum()) < mask.numel()
    solver = D2Q9CylinderWakeSolver(cylinder, seed=3)
    for _ in range(3):
        solver.step()
    state = solver.state()
    assert torch.isfinite(state).all()
    torch.testing.assert_close(state[:2, mask], torch.zeros_like(state[:2, mask]))
    torch.testing.assert_close(state[0, 0], torch.ones_like(state[0, 0]), atol=1e-6, rtol=0)


def test_transient_schema_split_and_problem_adapter(cylinder_case) -> None:
    config, dataset = cylinder_case
    validate_trajectories_against_spec(dataset.records, dataset.problem_spec)
    record = dataset.records[0]
    assert record.states_raw.shape == (11, 3, 48, 24)
    assert record.dts.shape == (10,)
    assert record.valid_mask is not None and record.coordinates is not None
    manifest = make_split_manifest(dataset.records, config.data.split)
    split_sets = [set(manifest.train), set(manifest.validation), set(manifest.test)]
    assert not (split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2])
    adapter = create_problem_adapter(config)
    assert adapter.build_problem_spec().name == "cylinder_wake_2d"
    assert set(adapter.build_physics_constraints()) == {"mass", "operator"}


def test_offline_dataset_roundtrip_and_contract_guard(cylinder_case, tmp_path: Path) -> None:
    config, dataset = cylinder_case
    cylinder = config.cylinder_wake_2d
    assert cylinder is not None
    path = tmp_path / "cylinder.pt"
    save_cylinder_wake_dataset(dataset, cylinder, path)
    restored = load_cylinder_wake_dataset(path, cylinder)
    torch.testing.assert_close(restored.records[0].states_raw, dataset.records[0].states_raw)
    with pytest.raises(ValueError, match="contract mismatch"):
        load_cylinder_wake_dataset(path, replace(cylinder, reynolds_number=90.0))


def test_physical_smoke_gate_is_finite_and_transient(cylinder_case) -> None:
    config, dataset = cylinder_case
    assert config.cylinder_wake_2d is not None
    report = validate_cylinder_wake_dataset(
        dataset, config.cylinder_wake_2d, require_shedding=False
    )
    assert report["status"] == "PASS"
    assert report["gates"]["finite_fields"]
    assert report["gates"]["nontrivial_transient"]


def test_cylinder_backbone_reuses_v06_contract_with_nonperiodic_padding() -> None:
    cylinder = load_config("configs/v0_8/cylinder_wake_cpu_smoke.yaml")
    periodic = load_config("configs/v0_6/advection_diffusion_2d_cpu_smoke.yaml")
    cylinder_model = initialize_v0_6_model(cylinder, device="cpu")
    periodic_model = initialize_v0_6_model(periodic, device="cpu")
    assert type(cylinder_model.online_encoder) is type(periodic_model.online_encoder)
    assert cylinder_model.online_encoder.padding_mode == "zeros"
    assert periodic_model.online_encoder.padding_mode == "circular"


def test_cylinder_jepa_loss_routes_grid_spec_to_divergence_constraint(cylinder_case) -> None:
    config, dataset = cylinder_case
    assert config.field_loss is not None and config.jepa_loss is not None
    manifest = make_split_manifest(dataset.records, config.data.split)
    normalizer = ChannelStandardizer(eps=config.data.normalization.eps).fit(
        dataset.records, manifest, dataset.problem_spec
    )
    windows = TrajectoryWindowDataset(
        select_split(dataset.records, manifest, "validation"),
        history=config.data.history,
        horizon=config.data.horizon,
        normalizer=normalizer,
    )
    batch = collate_problem_batches([windows[0]]).to(dtype=torch.float32)
    model = initialize_v0_6_model(config, device="cpu")
    constraints = create_problem_adapter(config).build_physics_constraints()

    losses = compute_field_jepa_loss(
        model,
        batch,
        normalizer,
        dataset.problem_spec,
        config.field_loss,
        config.jepa_loss,
        constraints,
        physics_scale=1.0,
    )

    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.v0_5.mass)
