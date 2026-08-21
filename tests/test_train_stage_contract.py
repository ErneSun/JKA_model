from __future__ import annotations

import pytest
import torch
from torch import nn

from jka_model.training import (
    TrainStage,
    assert_optimizer_matches_trainable_params,
    configure_train_stage,
)


class ToyStagedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.online_encoder = nn.Linear(2, 2)
        self.koopman_core = nn.Linear(2, 2)
        self.target_encoder = nn.Linear(2, 2)
        self.residual_memory = nn.Linear(2, 2)
        self.residual_head = nn.Linear(2, 2)
        self.training_decoder = nn.Linear(2, 1)


def test_train_stage_members_match_current_contract() -> None:
    assert {stage.value for stage in TrainStage} == {
        "koopman",
        "jepa",
        "residual",
        "joint",
        "context",
        "adaptive",
    }


@pytest.mark.parametrize("stage", list(TrainStage))
def test_train_stage_contract_is_deterministic(stage: TrainStage) -> None:
    model = ToyStagedModel()
    first = configure_train_stage(model, stage)
    first_flags = [parameter.requires_grad for parameter in model.parameters()]
    second = configure_train_stage(model, stage)
    assert first == second
    assert first_flags == [parameter.requires_grad for parameter in model.parameters()]
    assert not any(parameter.requires_grad for parameter in model.target_encoder.parameters())


def test_residual_stage_freezes_structure_and_optimizer_matches() -> None:
    model = ToyStagedModel()
    groups = configure_train_stage(model, TrainStage.RESIDUAL)
    assert groups["residual_memory"]
    assert groups["residual_head"]
    assert not groups["online_encoder"]
    assert not groups["koopman_core"]
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.1
    )
    assert_optimizer_matches_trainable_params(model, optimizer)

    frozen_before = [parameter.detach().clone() for parameter in model.koopman_core.parameters()]
    loss = model.residual_head(model.residual_memory(torch.ones(2, 2))).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for before, after in zip(frozen_before, model.koopman_core.parameters(), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_optimizer_ownership_rejects_frozen_parameter() -> None:
    model = ToyStagedModel()
    configure_train_stage(model, TrainStage.RESIDUAL)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="do not exactly match"):
        assert_optimizer_matches_trainable_params(model, optimizer)


def test_unowned_parameter_fails_loudly() -> None:
    model = ToyStagedModel()
    model.unregistered = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="not owned"):
        configure_train_stage(model, TrainStage.JOINT)
