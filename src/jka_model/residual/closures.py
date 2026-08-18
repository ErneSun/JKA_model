"""Minimal direct-delta latent closure models for V0.7."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from jka_model.models import FieldJEPAKoopmanModel


def _validate_inputs(
    history_z: Tensor,
    history_dts: Tensor,
    next_dt: Tensor,
    parameters: Tensor,
    *,
    latent_dim: int,
    history: int,
    parameter_dim: int,
) -> None:
    if history_z.ndim != 3 or history_z.shape[1:] != (history, latent_dim):
        raise ValueError(f"history_z must have shape [B,{history},{latent_dim}]")
    if history_dts.shape != (history_z.shape[0], history - 1):
        raise ValueError("history_dts must contain the intervals between history states")
    if next_dt.shape != (history_z.shape[0], 1):
        raise ValueError("next_dt must have shape [B,1]")
    if parameters.shape != (history_z.shape[0], parameter_dim):
        raise ValueError("parameters shape does not match closure contract")
    if torch.any(next_dt <= 0) or not all(
        torch.isfinite(item).all() for item in (history_z, history_dts, next_dt, parameters)
    ):
        raise ValueError("closure inputs must be finite and next_dt positive")


class BaseClosure(nn.Module):
    variant = "base"

    def __init__(self, latent_dim: int, history: int, parameter_dim: int) -> None:
        super().__init__()
        if latent_dim < 1 or history < 1 or parameter_dim < 0:
            raise ValueError("invalid closure dimensions")
        self.latent_dim = latent_dim
        self.history = history
        self.parameter_dim = parameter_dim

    def validate(
        self, history_z: Tensor, history_dts: Tensor, next_dt: Tensor, parameters: Tensor
    ) -> None:
        _validate_inputs(
            history_z,
            history_dts,
            next_dt,
            parameters,
            latent_dim=self.latent_dim,
            history=self.history,
            parameter_dim=self.parameter_dim,
        )


class ZeroClosure(BaseClosure):
    variant = "zero"

    def forward(
        self, history_z: Tensor, history_dts: Tensor, next_dt: Tensor, parameters: Tensor
    ) -> Tensor:
        self.validate(history_z, history_dts, next_dt, parameters)
        return torch.zeros_like(history_z[:, -1])


class LinearClosure(BaseClosure):
    variant = "linear"

    def __init__(self, latent_dim: int, history: int, parameter_dim: int) -> None:
        super().__init__(latent_dim, history, parameter_dim)
        self.linear = nn.Linear(latent_dim + 1 + parameter_dim, latent_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self, history_z: Tensor, history_dts: Tensor, next_dt: Tensor, parameters: Tensor
    ) -> Tensor:
        self.validate(history_z, history_dts, next_dt, parameters)
        return self.linear(torch.cat((history_z[:, -1], next_dt, parameters), dim=-1))


def _mlp(input_dim: int, hidden_dim: int, depth: int, output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(current, hidden_dim), nn.SiLU()))
        current = hidden_dim
    output = nn.Linear(current, output_dim)
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


def analytical_mlp_parameter_count(
    input_dim: int, hidden_dim: int, depth: int, output_dim: int
) -> int:
    """Count dense MLP parameters without allocating modules or consuming RNG."""
    if min(input_dim, hidden_dim, depth, output_dim) < 1:
        raise ValueError("MLP dimensions and depth must be positive")
    first = input_dim * hidden_dim + hidden_dim
    hidden = max(depth - 1, 0) * (hidden_dim * hidden_dim + hidden_dim)
    output = hidden_dim * output_dim + output_dim
    return first + hidden + output


def solve_parameter_matched_width(
    *,
    instantaneous_input_dim: int,
    history_input_dim: int,
    history_hidden_dim: int,
    depth: int,
    output_dim: int,
) -> int:
    """Find the closest integer control width using analytical counts only."""
    target_count = analytical_mlp_parameter_count(
        history_input_dim, history_hidden_dim, depth, output_dim
    )
    candidate_widths = range(1, max(history_hidden_dim * 4, 2))
    return min(
        candidate_widths,
        key=lambda width: abs(
            analytical_mlp_parameter_count(instantaneous_input_dim, width, depth, output_dim)
            - target_count
        ),
    )


class InstantaneousMLPClosure(BaseClosure):
    variant = "instantaneous"

    def __init__(
        self, latent_dim: int, history: int, parameter_dim: int, hidden_dim: int, depth: int
    ) -> None:
        super().__init__(latent_dim, history, parameter_dim)
        self.network = _mlp(latent_dim + 1 + parameter_dim, hidden_dim, depth, latent_dim)

    def forward(
        self, history_z: Tensor, history_dts: Tensor, next_dt: Tensor, parameters: Tensor
    ) -> Tensor:
        self.validate(history_z, history_dts, next_dt, parameters)
        return self.network(torch.cat((history_z[:, -1], next_dt, parameters), dim=-1))


class HistoryMLPClosure(BaseClosure):
    variant = "history"

    def __init__(
        self, latent_dim: int, history: int, parameter_dim: int, hidden_dim: int, depth: int
    ) -> None:
        super().__init__(latent_dim, history, parameter_dim)
        input_dim = history * latent_dim + (history - 1) + 1 + parameter_dim
        self.network = _mlp(input_dim, hidden_dim, depth, latent_dim)

    def forward(
        self, history_z: Tensor, history_dts: Tensor, next_dt: Tensor, parameters: Tensor
    ) -> Tensor:
        self.validate(history_z, history_dts, next_dt, parameters)
        flattened = history_z.flatten(start_dim=1)
        return self.network(torch.cat((flattened, history_dts, next_dt, parameters), dim=-1))


def build_closure(
    variant: str,
    *,
    latent_dim: int,
    history: int,
    parameter_dim: int,
    hidden_dim: int,
    depth: int,
) -> BaseClosure:
    if variant == "zero":
        return ZeroClosure(latent_dim, history, parameter_dim)
    if variant == "linear":
        return LinearClosure(latent_dim, history, parameter_dim)
    if variant == "instantaneous":
        # Parameter-match the instantaneous control to the primary history MLP as
        # closely as integer widths allow. This prevents capacity from masquerading
        # as temporal-memory evidence.
        history_input_dim = history * latent_dim + history + parameter_dim
        instantaneous_input_dim = latent_dim + 1 + parameter_dim
        matched_width = solve_parameter_matched_width(
            instantaneous_input_dim=instantaneous_input_dim,
            history_input_dim=history_input_dim,
            history_hidden_dim=hidden_dim,
            depth=depth,
            output_dim=latent_dim,
        )
        return InstantaneousMLPClosure(latent_dim, history, parameter_dim, matched_width, depth)
    if variant in {"history", "shuffled_history"}:
        closure = HistoryMLPClosure(latent_dim, history, parameter_dim, hidden_dim, depth)
        closure.variant = variant
        return closure
    raise ValueError(f"unknown closure variant: {variant}")


class ResidualKoopmanModel(nn.Module):
    """Frozen V0.6 backbone plus one trainable correction after each Koopman step."""

    def __init__(self, backbone: FieldJEPAKoopmanModel, residual_head: BaseClosure) -> None:
        super().__init__()
        if backbone.koopman_core.state_dim != residual_head.latent_dim:
            raise ValueError("backbone and closure latent dimensions differ")
        self.online_encoder = backbone.online_encoder
        self.koopman_core = backbone.koopman_core
        self.training_decoder = backbone.training_decoder
        self.target_encoder = backbone.target_encoder
        self.residual_head = residual_head
        self.online_encoder.requires_grad_(False)
        self.koopman_core.requires_grad_(False)
        self.training_decoder.requires_grad_(False)
        self.target_encoder.requires_grad_(False)

    def train_stage_modules(self) -> Mapping[str, nn.Module]:
        return {
            "online_encoder": self.online_encoder,
            "koopman_core": self.koopman_core,
            "training_decoder": self.training_decoder,
            "target_encoder": self.target_encoder,
            "residual_head": self.residual_head,
        }

    def train(self, mode: bool = True) -> ResidualKoopmanModel:
        super().train(mode)
        for module in (
            self.online_encoder,
            self.koopman_core,
            self.training_decoder,
            self.target_encoder,
        ):
            module.eval()
        return self

    def encode(self, field: Tensor) -> Tensor:
        return self.online_encoder(field).to(dtype=self.koopman_core.A.dtype)

    def decode(self, latent: Tensor) -> Tensor:
        return self.training_decoder(latent)

    def backbone_state_dict(self) -> dict[str, Any]:
        return {
            "online_encoder": self.online_encoder.state_dict(),
            "koopman_core": self.koopman_core.state_dict(),
            "training_decoder": self.training_decoder.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
        }

    def load_backbone_state_dict(self, state: Mapping[str, Any]) -> None:
        self.online_encoder.load_state_dict(state["online_encoder"], strict=True)
        self.koopman_core.load_state_dict(state["koopman_core"], strict=True)
        self.training_decoder.load_state_dict(state["training_decoder"], strict=True)
        self.target_encoder.load_state_dict(state["target_encoder"], strict=True)

    def corrected_step(
        self,
        history_z: Tensor,
        history_dts: Tensor,
        next_dt: Tensor,
        parameters: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        base = self.koopman_core.step(history_z[:, -1], next_dt[:, 0])
        correction = self.residual_head(history_z, history_dts, next_dt, parameters)
        return base + correction, base, correction
