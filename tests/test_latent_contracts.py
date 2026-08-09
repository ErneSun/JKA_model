from __future__ import annotations

import pytest
import torch

from jka_model.contracts import LatentState, TransitionOutput


def test_frozen_public_latent_names() -> None:
    state = LatentState(z_k=torch.zeros(2, 4), z_r=torch.zeros(2, 3))
    assert state.z_k.shape == (2, 4)
    assert state.z_r is not None
    output = TransitionOutput(
        z_k_base=torch.zeros(2, 4),
        z_r=torch.zeros(2, 2),
        delta_z_k=torch.ones(2, 4),
        gate=torch.full((2, 1), 0.1),
        z_k_next=torch.full((2, 4), 0.1),
    )
    assert output.delta_z_k.shape == output.z_k_next.shape


def test_transition_output_requires_scalar_gate() -> None:
    with pytest.raises(ValueError, match="gate"):
        TransitionOutput(
            z_k_base=torch.zeros(2, 4),
            z_r=None,
            delta_z_k=torch.zeros(2, 4),
            gate=torch.zeros(2, 4),
            z_k_next=torch.zeros(2, 4),
        )
