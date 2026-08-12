from __future__ import annotations

import pytest
import torch

from jka_model.config import SplitConfig, ToyAdvectionDiffusionConfig
from jka_model.data import (
    ChannelStandardizer,
    TrajectoryWindowDataset,
    generate_advection_diffusion_trajectories,
    make_split_manifest,
    select_split,
)
from jka_model.physics import (
    ChannelMeanProbe,
    ChannelRMSProbe,
    DiscretePDEResidualConstraint,
    FiniteValueConstraint,
    MassConservationConstraint,
    PeriodicBoundaryConstraint,
    StateAdmissibilityConstraint,
    create_constraint,
    evaluate_batch_probes,
    evaluate_constraints,
    evaluate_probes,
    register_constraint,
)


def _make_batch():
    records, spec = generate_advection_diffusion_trajectories(
        ToyAdvectionDiffusionConfig(num_trajectories=6), seed=12
    )
    manifest = make_split_manifest(records, SplitConfig(seed=7))
    normalizer = ChannelStandardizer().fit(records, manifest, spec)
    dataset = TrajectoryWindowDataset(
        select_split(records, manifest, "train"), history=4, horizon=2, normalizer=normalizer
    )
    return dataset[0], spec


def test_constraints_run_on_raw_state_and_remain_differentiable() -> None:
    batch, spec = _make_batch()
    batch.context_states_model.fill_(1e9)
    batch.future_states_model.fill_(-1e9)
    batch.future_states_raw.requires_grad_(True)
    terms = evaluate_constraints(
        [
            FiniteValueConstraint(),
            StateAdmissibilityConstraint(lower=0.0, upper=2.0),
            PeriodicBoundaryConstraint(),
            MassConservationConstraint(),
            DiscretePDEResidualConstraint(),
        ],
        batch,
        spec,
    )
    assert set(terms) == {
        "finite_values",
        "state_admissibility",
        "periodic_boundary",
        "mass_conservation",
        "discrete_pde_residual",
    }
    assert terms["periodic_boundary"] < 1e-20
    # The two conserved masses are independent FP32 reductions. Their exact rounding can
    # vary with the CPU/PyTorch reduction kernel, so compare the squared penalty against a
    # scale-aware roundoff bound instead of requiring an effectively bitwise-zero result.
    raw_state = batch.future_states_raw[:, 0]
    mass_scale = raw_state.abs().amax() * batch.cell_weights.abs().sum(dim=-1).amax()
    mass_delta_atol = 32.0 * torch.finfo(raw_state.dtype).eps * mass_scale
    mass_penalty_atol = float(mass_delta_atol.detach().square())
    torch.testing.assert_close(
        terms["mass_conservation"],
        torch.zeros_like(terms["mass_conservation"]),
        rtol=0.0,
        atol=mass_penalty_atol,
    )
    shifted = raw_state.detach().clone().add_(0.1)
    shifted_mass_penalty = MassConservationConstraint().loss(
        shifted,
        prev_state_raw=batch.context_states_raw[:, -1],
        metadata={
            "cell_weights": batch.cell_weights,
            "valid_mask": batch.valid_mask,
        },
    )["mass_conservation"]
    assert shifted_mass_penalty > 0.01
    sum(terms.values()).backward()
    assert batch.future_states_raw.grad is not None


def test_registry_errors_and_optional_probes() -> None:
    batch, spec = _make_batch()
    assert isinstance(create_constraint("finite_values"), FiniteValueConstraint)
    bounded = create_constraint(
        {"name": "state_admissibility", "parameters": {"lower": 0.0}}
    )
    assert isinstance(bounded, StateAdmissibilityConstraint)
    with pytest.raises(KeyError, match="unregistered"):
        create_constraint("does_not_exist")
    with pytest.raises(ValueError, match="already registered"):
        register_constraint("finite_values", FiniteValueConstraint)
    assert evaluate_probes([], batch.future_states_raw[:, 0], spec) == {}
    values = evaluate_probes(
        [ChannelMeanProbe(), ChannelRMSProbe()], batch.future_states_raw[:, 0], spec
    )
    assert set(values) == {"channel_mean", "channel_rms"}
    assert values["channel_mean"].shape == (1, 1)
    raw_measurement = evaluate_batch_probes([ChannelMeanProbe()], batch, spec)["channel_mean"]
    batch.future_states_model.add_(1e6)
    after_model_change = evaluate_batch_probes([ChannelMeanProbe()], batch, spec)["channel_mean"]
    torch.testing.assert_close(raw_measurement, after_model_change)


def test_finite_and_periodic_constraints_detect_deliberate_violations() -> None:
    batch, spec = _make_batch()
    corrupted = batch.future_states_raw[:, 0].clone()
    corrupted[..., 3] = float("nan")
    corrupted[..., 4] = float("inf")
    finite = FiniteValueConstraint().loss(corrupted)["finite_values"]
    assert torch.isfinite(finite) and finite > 0

    periodic = PeriodicBoundaryConstraint()
    valid = periodic.loss(batch.future_states_raw[:, 0])["periodic_boundary"]
    broken = batch.future_states_raw[:, 0].clone()
    broken[..., -1] += 0.25
    invalid = periodic.loss(broken)["periodic_boundary"]
    assert valid < 1e-20
    assert invalid > 0.01


def test_pde_residual_improves_with_time_and_space_resolution() -> None:
    coarse_config = ToyAdvectionDiffusionConfig(
        num_trajectories=1,
        num_steps=4,
        nx=33,
        base_dt=0.06,
        variable_dt=False,
    )
    fine_config = ToyAdvectionDiffusionConfig(
        num_trajectories=1,
        num_steps=4,
        nx=129,
        base_dt=0.015,
        variable_dt=False,
    )
    coarse_records, coarse_spec = generate_advection_diffusion_trajectories(
        coarse_config, seed=21, dtype=torch.float64
    )
    fine_records, fine_spec = generate_advection_diffusion_trajectories(
        fine_config, seed=21, dtype=torch.float64
    )
    constraint = DiscretePDEResidualConstraint()

    def residual(records, problem_spec):
        record = records[0]
        assert record.mu_static is not None
        result = constraint.evaluate(
            record.states_raw[1:2],
            prev_state_raw=record.states_raw[0:1],
            dt=record.dts[0:1],
            spec=problem_spec,
            metadata={"mu_static": record.mu_static.unsqueeze(0)},
        )
        return result.penalty

    assert residual(fine_records, fine_spec) < residual(coarse_records, coarse_spec)
