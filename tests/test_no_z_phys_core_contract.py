from __future__ import annotations

import inspect
from dataclasses import fields

import torch

from jka_model.contracts import LatentState


def test_no_z_phys_core_contract() -> None:
    field_names = {field.name for field in fields(LatentState)}
    assert field_names == {"z_k", "z_r"}
    assert "z_phys" not in inspect.signature(LatentState).parameters

    state = LatentState(z_k=torch.zeros(2, 4))
    assert not hasattr(state, "z_phys")

