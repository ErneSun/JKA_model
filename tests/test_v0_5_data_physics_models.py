"""CPU-only numerical and differentiability checks for V0.5."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from jka_model.config import ProjectConfig, load_config
from jka_model.data import (
    ChannelStandardizer,
    TrajectoryWindowDataset,
    collate_problem_batches,
    generate_advection_diffusion_2d_trajectories,
    make_split_manifest,
    select_split,
    validate_trajectories_against_spec,
)
from jka_model.losses import compute_field_koopman_loss
from jka_model.physics import (
    AdvectionDiffusionOperatorConstraint2D,
    AdvectionDiffusionSpectralStepConstraint2D,
    periodic_first_derivative_2d,
    periodic_second_derivative_2d,
    weighted_integral_2d,
)
from jka_model.problems import create_problem_adapter
from train.train_v0_5 import initialize_v0_5_model

CONFIG = "configs/v0_5/advection_diffusion_2d_cpu_smoke.yaml"


def test_v0_5_programmatic_config_has_stable_serialization_types() -> None:
    config = load_config(CONFIG)
    assert config.field_loss is not None
    replaced = replace(config, field_loss=replace(config.field_loss, lambda_k=10))
    restored = ProjectConfig.from_dict(replaced.to_dict())
    assert restored.stable_hash == replaced.stable_hash


def test_v0_5_data_shapes_alignment_and_analytic_mass() -> None:
    config = load_config(CONFIG)
    assert config.advection_diffusion_2d is not None
    dataset = generate_advection_diffusion_2d_trajectories(
        config.advection_diffusion_2d, seed=config.training.seed
    )
    validate_trajectories_against_spec(dataset.records, dataset.problem_spec)
    record = dataset.records[0]
    assert record.states_raw.shape == (9, 1, 8, 8)
    assert record.coordinates is not None and record.coordinates.shape == (2, 8, 8)
    assert record.cell_weights is not None and record.cell_weights.shape == (8, 8)
    mass = weighted_integral_2d(record.states_raw, record.cell_weights).squeeze(-1)
    expected = float(record.metadata["mean"]) * dataset.reference.domain_area
    assert torch.allclose(mass, torch.full_like(mass, expected), atol=1e-12, rtol=1e-12)


def test_v0_5_spatial_derivatives_against_analytic_and_converge() -> None:
    errors = {"dx": [], "dy": [], "dxx": [], "dyy": []}
    for nx in (32, 64, 128):
        ny = nx
        x = torch.arange(nx, dtype=torch.float64) * (2 * math.pi / nx)
        y = torch.arange(ny, dtype=torch.float64) * (2 * math.pi / ny)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        field = torch.sin(2 * xx + 3 * yy)
        spacing = 2 * math.pi / nx
        numerical = {
            "dx": periodic_first_derivative_2d(field, spacing, -2),
            "dy": periodic_first_derivative_2d(field, spacing, -1),
            "dxx": periodic_second_derivative_2d(field, spacing, -2),
            "dyy": periodic_second_derivative_2d(field, spacing, -1),
        }
        analytic = {
            "dx": 2 * torch.cos(2 * xx + 3 * yy),
            "dy": 3 * torch.cos(2 * xx + 3 * yy),
            "dxx": -4 * field,
            "dyy": -9 * field,
        }
        for name in errors:
            errors[name].append(float((numerical[name] - analytic[name]).square().mean().sqrt()))
        # The boundary entries exercise torch.roll across both periodic seams.
        assert torch.allclose(numerical["dx"][[0, -1]], analytic["dx"][[0, -1]], atol=0.06)
        assert torch.allclose(numerical["dy"][:, [0, -1]], analytic["dy"][:, [0, -1]], atol=0.18)
    for name, values in errors.items():
        assert values[2] < values[1] < values[0], (name, values)
        observed_order = math.log(values[1] / values[2], 2)
        assert 1.8 < observed_order < 2.2, (name, observed_order)


def test_v0_5_spatial_operators_reject_invalid_grid_arguments() -> None:
    field = torch.zeros(4, 4)
    with pytest.raises(ValueError, match="positive spacing"):
        periodic_first_derivative_2d(field, 0.0, -2)
    with pytest.raises(ValueError, match="axis"):
        periodic_second_derivative_2d(field, 1.0, 0)
    with pytest.raises(ValueError, match=">=4"):
        periodic_second_derivative_2d(torch.zeros(3, 4), 1.0, -2)


def test_v0_5_model_shapes_circular_padding_and_physics_gradients() -> None:
    config = load_config(CONFIG)
    assert config.advection_diffusion_2d and config.field_loss
    dataset = generate_advection_diffusion_2d_trajectories(
        config.advection_diffusion_2d, seed=config.training.seed
    )
    manifest = make_split_manifest(dataset.records, config.data.split)
    normalizer = ChannelStandardizer(eps=config.data.normalization.eps).fit(
        dataset.records, manifest, dataset.problem_spec
    )
    windows = TrajectoryWindowDataset(
        select_split(dataset.records, manifest, "train"),
        history=config.data.history,
        horizon=config.data.horizon,
        normalizer=normalizer,
    )
    batch = collate_problem_batches([windows[0], windows[1]]).to(dtype=torch.float32)
    model = initialize_v0_5_model(config, device="cpu")
    constraints = create_problem_adapter(config).build_physics_constraints()
    assert all(
        layer.padding_mode == "circular"
        for layer in model.modules()
        if isinstance(layer, torch.nn.Conv2d)
    )
    prediction, latent = model.rollout(batch.context_states_model[:, -1], batch.future_dts)
    assert prediction.shape == batch.future_states_model.shape
    assert latent.shape == (2, config.data.horizon + 1, 4)
    losses = compute_field_koopman_loss(
        model,
        batch,
        normalizer,
        dataset.problem_spec,
        config.field_loss,
        constraints,
        physics_scale=1.0,
    )
    assert torch.isfinite(losses.total)
    expected_without_forecast = (
        config.field_loss.lambda_k * losses.koopman_one_step
        + config.field_loss.lambda_generator * losses.generator_consistency
        + config.field_loss.lambda_multi * losses.koopman_multi_step
        + config.field_loss.lambda_rec * losses.reconstruction
        + config.field_loss.lambda_var * losses.variance
        + config.field_loss.lambda_stability * losses.stability
        + config.field_loss.lambda_physics
        * (
            config.field_loss.lambda_mass * losses.mass
            + config.field_loss.lambda_operator * losses.operator
        )
    )
    assert torch.allclose(
        losses.total - expected_without_forecast,
        config.field_loss.lambda_forecast * losses.forecast_model,
    )
    physics_only = (
        config.field_loss.lambda_mass * losses.mass
        + config.field_loss.lambda_operator * losses.operator
    )
    physics_only.backward()
    assert model.core.A.grad is not None and model.core.A.grad.norm() > 0
    assert any(
        parameter.grad is not None and parameter.grad.norm() > 0
        for parameter in model.encoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.norm() > 0
        for parameter in model.decoder.parameters()
    )
    model.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        amp_losses = compute_field_koopman_loss(
            model,
            batch,
            normalizer,
            dataset.problem_spec,
            config.field_loss,
            constraints,
            physics_scale=1.0,
        )
    assert amp_losses.mass.dtype == torch.float32
    assert amp_losses.operator.dtype == torch.float32
    amp_losses.total.backward()
    assert model.core.A.grad is not None and torch.isfinite(model.core.A.grad).all()


def test_v0_5_encoder_preserves_periodic_phase_and_generator_starts_stable() -> None:
    config = load_config(CONFIG)
    assert config.advection_diffusion_2d is not None
    dataset = generate_advection_diffusion_2d_trajectories(
        config.advection_diffusion_2d, seed=config.training.seed
    )
    model = initialize_v0_5_model(config, device="cpu")
    field = dataset.records[0].states_raw[0:1].to(torch.float32)
    shifted = torch.roll(field, shifts=1, dims=-1)
    assert not torch.allclose(model.encode(field), model.encode(shifted))
    eigenvalues = torch.linalg.eigvals(model.core.A.detach())
    assert float(eigenvalues.real.max()) < 0


def test_v0_5_true_transition_operator_residual_is_finite_and_small() -> None:
    config = load_config(CONFIG)
    assert config.advection_diffusion_2d
    dataset = generate_advection_diffusion_2d_trajectories(config.advection_diffusion_2d, seed=2)
    record = dataset.records[0]
    constraint = AdvectionDiffusionOperatorConstraint2D()
    value = constraint.loss(
        record.states_raw[1:2],
        prev_state_raw=record.states_raw[0:1],
        dt=record.dts[0:1],
        spec=dataset.problem_spec,
        metadata={
            "mu_static": record.mu_static.unsqueeze(0),
            "cell_weights": record.cell_weights.unsqueeze(0),
        },
    )[constraint.name]
    assert torch.isfinite(value)
    assert float(value) < 0.05


def test_v0_5_spectral_step_is_exact_for_reference_and_differentiable() -> None:
    config = load_config(CONFIG)
    assert config.advection_diffusion_2d
    dataset = generate_advection_diffusion_2d_trajectories(config.advection_diffusion_2d, seed=2)
    record = dataset.records[0]
    prediction = record.states_raw[1:2].clone().requires_grad_(True)
    constraint = AdvectionDiffusionSpectralStepConstraint2D()
    value = constraint.loss(
        prediction,
        prev_state_raw=record.states_raw[0:1],
        dt=record.dts[0:1],
        spec=dataset.problem_spec,
        metadata={"mu_static": record.mu_static.unsqueeze(0)},
    )[constraint.name]
    assert torch.isfinite(value)
    assert float(value.detach()) < 1e-24
    value.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
