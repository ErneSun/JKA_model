from __future__ import annotations

from gpu_validation.v0_6.scripts.gpu_compare import compare


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
