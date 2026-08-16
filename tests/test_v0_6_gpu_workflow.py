from __future__ import annotations

import sys

from gpu_validation.v0_6.scripts.gpu_compare import compare
from gpu_validation.v0_6.scripts.gpu_report import _absolute_gates
from gpu_validation.v0_6.scripts.gpu_validate_all import _run_checked


def _metrics(rmse: float, mass: float, operator: float) -> dict:
    return {
        "rollout": {"long": {"rmse": rmse, "mass_drift": mass, "operator": operator}},
        "relative_mass_drift_threshold": 0.01,
        "operator_mse_threshold": 1.0e-4,
        "collapse_gate": True,
        "target_used_for_rollout": False,
    }


def test_physics_comparison_uses_absolute_floor_near_zero() -> None:
    control = _metrics(1.0, 1.0e-12, 1.0e-12)
    jepa = _metrics(1.04, 0.005, 5.0e-5)
    result = compare(control, jepa, rollout_margin=0.05, physics_margin=0.10)
    assert result["pass"]
    assert result["limits"]["mass_drift"] == 0.01
    assert result["limits"]["operator"] == 1.0e-4


def test_long_rollout_gate_remains_relative() -> None:
    result = compare(
        _metrics(1.0, 0.001, 1.0e-5),
        _metrics(1.06, 0.001, 1.0e-5),
        rollout_margin=0.05,
        physics_margin=0.10,
    )
    assert not result["pass"]
    assert not result["gates"]["long_rollout_noninferiority"]


def test_validation_step_tees_output_to_terminal_and_log(tmp_path, capsys) -> None:
    log = tmp_path / "step.log"

    _run_checked(
        [sys.executable, "-c", "print('visible validation output', flush=True)"],
        log,
        label="test step",
    )

    terminal = capsys.readouterr().out
    assert "test step: START" in terminal
    assert "visible validation output" in terminal
    assert "test step: PASS" in terminal
    assert log.read_text(encoding="utf-8") == "visible validation output\n"


def test_v0_6_review_checks_inherited_absolute_gates() -> None:
    metrics = {
        "finite": True,
        "frequency_relative_error": 0.01,
        "frequency_threshold": 0.05,
        "decay_relative_error": 0.10,
        "decay_threshold": 0.20,
        "spectral_abscissa": -0.001,
        "spectral_abscissa_threshold": 0.001,
        "relative_mass_drift_threshold": 0.01,
        "operator_mse_threshold": 1.0e-4,
        "collapse_gate": True,
        "target_used_for_rollout": False,
        "rollout": {
            name: {
                "rmse": 0.1,
                "persistence_rmse": 1.0,
                "mass_drift": 0.001,
                "operator": 1.0e-5,
            }
            for name in ("short", "medium", "long")
        },
    }
    training = {
        "finite": True,
        "target_in_optimizer": False,
        "optimizer_ema_counts_match": True,
    }

    gates = _absolute_gates(metrics, training)

    assert all(gates.values())
    metrics["rollout"]["medium"]["mass_drift"] = 0.02
    assert not _absolute_gates(metrics, training)["mass_all_horizons"]
