"""Small training-only decoder from Koopman coordinates to model-space state."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from jka_model.models.koopman_encoder import _activation


class TrainingDecoder(nn.Module):
    """Decode ``z_k`` only; this module has no time or propagation interface."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 32,
        hidden_layers: int = 2,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or output_dim < 1 or hidden_dim < 1:
            raise ValueError("decoder dimensions must be positive")
        if hidden_layers not in {1, 2}:
            raise ValueError("decoder hidden_layers must be 1 or 2")
        layers: list[nn.Module] = [
            nn.Linear(latent_dim, hidden_dim, dtype=dtype, device=device),
            _activation(activation),
        ]
        if hidden_layers == 2:
            layers.extend(
                (
                    nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device),
                    _activation(activation),
                )
            )
        layers.append(nn.Linear(hidden_dim, output_dim, dtype=dtype, device=device))
        self.network = nn.Sequential(*layers)
        self.latent_dim = latent_dim
        self.output_dim = output_dim

    def forward(self, z_k: Tensor) -> Tensor:
        if z_k.ndim < 2 or z_k.shape[-1] != self.latent_dim:
            raise ValueError("decoder input must have shape [...,latent_dim] with a batch axis")
        parameter = next(self.parameters())
        if z_k.dtype != parameter.dtype or z_k.device != parameter.device:
            raise ValueError("decoder input dtype/device must match model parameters")
        if not torch.isfinite(z_k).all():
            raise ValueError("decoder input must contain only finite values")
        decoded = self.network(z_k)
        if not torch.isfinite(decoded).all():
            raise FloatingPointError("decoder output became non-finite")
        return decoded
