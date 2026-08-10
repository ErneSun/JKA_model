from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from jka_model.config import (
    DampedOscillatorConfig,
    DirectIdentificationConfig,
    DuffingConfig,
    ProjectConfig,
    load_config,
)
from jka_model.data import (
    generate_damped_oscillator_trajectories,
    generate_duffing_trajectories,
    trajectory_transition_tensors,
)
from jka_model.evaluation import evaluate_rollout
from jka_model.metrics import dominant_oscillatory_mode
from jka_model.models import ContinuousKoopmanCore
from jka_model.training import (
    IdentificationResult,
    initialize_direct_koopman,
    one_step_mse,
    train_direct_koopman,
)
from jka_model.utils import Checkpoint, capture_rng_state, load_checkpoint, save_checkpoint


@pytest.fixture(scope="module")
def identified_oscillator():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "v0_3_smoke.yaml")
    assert config.oscillator is not None
    assert config.identification is not None
    records, spec = generate_damped_oscillator_trajectories(
        config.oscillator, seed=config.training.seed, dtype=torch.float64
    )
    states, targets, dts = trajectory_transition_tensors(records)
    core = initialize_direct_koopman(
        2,
        seed=config.training.seed,
        init_scale=config.identification.init_scale,
        dtype=torch.float64,
    )
    result = train_direct_koopman(core, states, targets, dts, config.identification)
    return config, records, spec, core, result, states, targets, dts


def test_direct_state_identification_frequency(identified_oscillator) -> None:
    config, _, _, core, result, _, _, dts = identified_oscillator
    assert config.oscillator is not None
    assert not torch.allclose(dts, dts[:1].expand_as(dts))
    _, learned_omega = dominant_oscillatory_mode(core.spectrum())
    true_omega = math.sqrt(config.oscillator.omega0**2 - config.oscillator.gamma**2)
    relative_error = abs(learned_omega - true_omega) / true_omega
    assert relative_error < 0.01
    assert result.final_loss < result.initial_loss * 1e-4


def test_constant_dt_direct_state_identification_frequency() -> None:
    data_config = DampedOscillatorConfig(
        omega0=2.0,
        gamma=0.15,
        base_dt=0.04,
        variable_dt=False,
        dt_jitter=0.0,
        num_steps=160,
        num_trajectories=12,
    )
    records, _ = generate_damped_oscillator_trajectories(
        data_config, seed=13, dtype=torch.float64
    )
    states, targets, dts = trajectory_transition_tensors(records)
    assert torch.allclose(dts, dts[:1].expand_as(dts))
    fit_config = DirectIdentificationConfig(
        epochs=700, learning_rate=0.03, init_scale=0.05
    )
    core = initialize_direct_koopman(
        2, seed=13, init_scale=fit_config.init_scale, dtype=torch.float64
    )
    result = train_direct_koopman(core, states, targets, dts, fit_config)
    _, learned_omega = dominant_oscillatory_mode(core.spectrum())
    true_omega = math.sqrt(data_config.omega0**2 - data_config.gamma**2)
    relative_error = abs(learned_omega - true_omega) / true_omega
    assert relative_error < 0.01
    assert result.final_loss < result.initial_loss * 1e-4


def test_100_step_rollout_finite(identified_oscillator) -> None:
    _, records, _, core, _, _, _, _ = identified_oscillator
    record = records[0]
    prediction = core.rollout(record.states_raw[0], record.dts[:100])
    assert prediction.shape == (101, 2)
    assert torch.isfinite(prediction).all()
    assert core.spectrum().growth_rates.max() < 0


def test_koopman_beats_persistence_on_linear_oscillator(identified_oscillator) -> None:
    _, records, _, core, _, _, _, _ = identified_oscillator
    record = records[0]
    prediction = core.rollout(record.states_raw[0], record.dts[:100])
    metrics = evaluate_rollout(prediction, record.states_raw[:101])
    assert metrics.finite
    assert metrics.rollout_mse < 0.01 * metrics.persistence_mse


def test_identification_one_step_mse_is_finite(identified_oscillator) -> None:
    _, _, _, core, result, states, targets, dts = identified_oscillator
    loss = one_step_mse(core, states, targets, dts)
    assert torch.isfinite(loss)
    assert loss < 1e-12
    assert result.final_loss == pytest.approx(float(loss.detach()), rel=1e-12, abs=1e-30)


def test_koopman_checkpoint_roundtrip(tmp_path, identified_oscillator) -> None:
    config, records, spec, core, result, _, _, _ = identified_oscillator
    assert isinstance(config, ProjectConfig)
    assert isinstance(result, IdentificationResult)
    probe_state = records[0].states_raw[0]
    probe_dt = records[0].dts[0]
    before = core.step(probe_state, probe_dt)
    before_spectrum = core.spectrum()
    saved_rng = capture_rng_state()
    destination = tmp_path / "koopman.pt"
    save_checkpoint(
        Checkpoint(
            train_stage=config.training.stage,
            epoch=result.epochs,
            global_step=result.global_step,
            online_model_state=core.state_dict(),
            optimizer_state=result.optimizer_state,
            rng_state=saved_rng,
            problem_spec=spec,
            config=config,
            data_fingerprint="sha256:v0.3-test",
            split_manifest={"train": [record.trajectory_id for record in records]},
            physics_constraint_spec=[],
        ),
        destination,
    )
    restored_checkpoint = load_checkpoint(destination)
    restored = ContinuousKoopmanCore(2, trainable=True, dtype=torch.float64)
    assert restored_checkpoint.online_model_state is not None
    restored.load_state_dict(restored_checkpoint.online_model_state)
    assert restored_checkpoint.optimizer_state is not None
    expected_optimizer = result.optimizer_state
    assert restored_checkpoint.optimizer_state["param_groups"] == expected_optimizer["param_groups"]
    restored_optimizer_state = restored_checkpoint.optimizer_state["state"]
    for parameter_id, expected_state in expected_optimizer["state"].items():
        actual_state = restored_optimizer_state[parameter_id]
        for name, expected_value in expected_state.items():
            actual_value = actual_state[name]
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                assert actual_value == expected_value
    assert restored_checkpoint.epoch == result.epochs
    assert restored_checkpoint.global_step == result.global_step
    assert restored_checkpoint.config == config
    assert restored_checkpoint.architecture_revision == "2.2"
    assert restored_checkpoint.project_version == "0.5.0"
    assert restored_checkpoint.rng_state is not None
    assert restored_checkpoint.rng_state.python == saved_rng.python
    np.testing.assert_array_equal(
        restored_checkpoint.rng_state.numpy[1], saved_rng.numpy[1]
    )
    torch.testing.assert_close(restored_checkpoint.rng_state.torch_cpu, saved_rng.torch_cpu)
    torch.testing.assert_close(restored.step(probe_state, probe_dt), before)
    restored_spectrum = restored.spectrum()
    torch.testing.assert_close(
        restored_spectrum.growth_rates.sort().values,
        before_spectrum.growth_rates.sort().values,
    )
    torch.testing.assert_close(
        restored_spectrum.angular_frequencies.sort().values,
        before_spectrum.angular_frequencies.sort().values,
    )


def test_duffing_pipeline_runs() -> None:
    data_config = DuffingConfig(num_steps=100, num_trajectories=6, rk4_substeps=2)
    records, _ = generate_duffing_trajectories(data_config, seed=23, dtype=torch.float64)
    states, targets, dts = trajectory_transition_tensors(records)
    fit_config = DirectIdentificationConfig(epochs=350, learning_rate=0.025, init_scale=0.05)
    core = initialize_direct_koopman(
        2, seed=23, init_scale=fit_config.init_scale, dtype=torch.float64
    )
    result = train_direct_koopman(core, states, targets, dts, fit_config)
    prediction = core.rollout(records[0].states_raw[0], records[0].dts[:80])
    metrics = evaluate_rollout(prediction, records[0].states_raw[:81])
    assert math.isfinite(result.initial_loss) and math.isfinite(result.final_loss)
    assert result.final_loss < result.initial_loss
    assert prediction.shape == (81, 2)
    assert metrics.finite and math.isfinite(metrics.rollout_mse)
