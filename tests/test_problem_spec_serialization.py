from __future__ import annotations

import pytest

from jka_model.contracts import DtMode, ProblemSpec


def test_problem_spec_serialization(toy_problem_spec: ProblemSpec) -> None:
    restored = ProblemSpec.from_dict(toy_problem_spec.to_dict())
    assert restored == toy_problem_spec
    assert restored.channel_names == ("rho", "u")
    assert restored.units == ("kg/m^3", "m/s")


def test_problem_spec_is_frozen(toy_problem_spec: ProblemSpec) -> None:
    with pytest.raises((AttributeError, TypeError)):
        toy_problem_spec.metadata["new"] = "not allowed"  # type: ignore[index]


def test_problem_spec_rejects_unknown_fields(toy_problem_spec: ProblemSpec) -> None:
    payload = toy_problem_spec.to_dict()
    payload["mystery"] = 1
    with pytest.raises(ValueError, match="unknown ProblemSpec"):
        ProblemSpec.from_dict(payload)


def test_variable_dt_rejects_hidden_constant(toy_problem_spec: ProblemSpec) -> None:
    payload = toy_problem_spec.to_dict()
    payload["dt_mode"] = DtMode.VARIABLE.value
    with pytest.raises(ValueError, match="must not set constant_dt"):
        ProblemSpec.from_dict(payload)

