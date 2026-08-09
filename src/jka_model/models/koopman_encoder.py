"""Small online encoder from normalized observations to Koopman coordinates."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    raise ValueError("activation must be tanh or silu")


class KoopmanEncoder(nn.Module):
    """Encode model-space vectors ``[...,input_dim]`` into ``z_k`` coordinates."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        hidden_dim: int = 32,
        hidden_layers: int = 2,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if input_dim < 1 or latent_dim < 1 or hidden_dim < 1:
            raise ValueError("encoder dimensions must be positive")
        if hidden_layers not in {0, 1, 2}:
            raise ValueError("encoder hidden_layers must be 0, 1, or 2")
        if hidden_layers == 0:
            layers: list[nn.Module] = [
                nn.Linear(input_dim, latent_dim, dtype=dtype, device=device)
            ]
        else:
            layers = [
                nn.Linear(input_dim, hidden_dim, dtype=dtype, device=device),
                _activation(activation),
            ]
            if hidden_layers == 2:
                layers.extend(
                    (
                        nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device),
                        _activation(activation),
                    )
                )
            layers.append(nn.Linear(hidden_dim, latent_dim, dtype=dtype, device=device))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def forward(self, state_model: Tensor) -> Tensor:
        if state_model.ndim < 2 or state_model.shape[-1] != self.input_dim:
            raise ValueError("encoder input must have shape [...,input_dim] with a batch axis")
        parameter = next(self.parameters())
        if state_model.dtype != parameter.dtype or state_model.device != parameter.device:
            raise ValueError("encoder input dtype/device must match model parameters")
        if not torch.isfinite(state_model).all():
            raise ValueError("encoder input must contain only finite values")
        latent = self.network(state_model)
        if not torch.isfinite(latent).all():
            raise FloatingPointError("encoder output became non-finite")
        return latent
