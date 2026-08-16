"""V0.6 JEPA shell around the unchanged V0.5 field Koopman model."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from jka_model.models.field_koopman_autoencoder import (
    FieldKoopmanAutoencoder,
    KoopmanEncoder2D,
    TrainingDecoder2D,
)
from jka_model.models.koopman_core import ContinuousKoopmanCore


class FieldJEPAKoopmanModel(nn.Module):
    """Online V0.5 model plus a frozen, EMA-tracked encoder.

    The continuous Koopman core is the only predictor.  The target encoder is not
    used by :meth:`rollout` and therefore cannot change inference capacity.
    """

    def __init__(
        self,
        online_encoder: KoopmanEncoder2D,
        koopman_core: ContinuousKoopmanCore,
        training_decoder: TrainingDecoder2D,
        target_encoder: KoopmanEncoder2D | None = None,
    ) -> None:
        super().__init__()
        if (
            online_encoder.latent_dim != koopman_core.state_dim
            or training_decoder.latent_dim != koopman_core.state_dim
        ):
            raise ValueError("online encoder/core/decoder latent dimensions must match")
        self.online_encoder = online_encoder
        self.koopman_core = koopman_core
        self.training_decoder = training_decoder
        self.target_encoder = (
            copy.deepcopy(online_encoder) if target_encoder is None else target_encoder
        )
        if self.target_encoder.latent_dim != online_encoder.latent_dim:
            raise ValueError("online and target encoder architectures must match")
        self.hard_sync_target()

    @property
    def encoder(self) -> KoopmanEncoder2D:
        return self.online_encoder

    @property
    def core(self) -> ContinuousKoopmanCore:
        return self.koopman_core

    @property
    def decoder(self) -> TrainingDecoder2D:
        return self.training_decoder

    def encode(self, field: Tensor) -> Tensor:
        return self.online_encoder(field).to(dtype=self.koopman_core.A.dtype)

    def encode_target(self, field: Tensor) -> Tensor:
        """Return a graph-free target; latent normalization is deliberately absent."""
        with torch.no_grad():
            return self.target_encoder(field).to(dtype=self.koopman_core.A.dtype).detach()

    def decode(self, latent: Tensor) -> Tensor:
        return self.training_decoder(latent)

    def rollout(self, current_field: Tensor, future_dts: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(current_field)
        latent_rollout = self.koopman_core.rollout(latent, future_dts)
        return self.decode(latent_rollout[:, 1:]), latent_rollout

    def hard_sync_target(self) -> None:
        self.target_encoder.load_state_dict(self.online_encoder.state_dict(), strict=True)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> FieldJEPAKoopmanModel:
        super().train(mode)
        # Parent train() toggles every child, but the target is a deterministic teacher.
        self.target_encoder.eval()
        return self

    def train_stage_modules(self) -> dict[str, nn.Module]:
        return {
            "online_encoder": self.online_encoder,
            "koopman_core": self.koopman_core,
            "training_decoder": self.training_decoder,
            "target_encoder": self.target_encoder,
        }

    def online_state_dict(self) -> dict[str, Any]:
        """V0.5-compatible flat state for online inference components only."""
        model = FieldKoopmanAutoencoder(
            self.online_encoder, self.koopman_core, self.training_decoder
        )
        return model.state_dict()

    def load_online_state_dict(self, state: Mapping[str, Any]) -> None:
        model = FieldKoopmanAutoencoder(
            self.online_encoder, self.koopman_core, self.training_decoder
        )
        model.load_state_dict(state, strict=True)


@torch.no_grad()
def ema_update_target(model: FieldJEPAKoopmanModel, tau: float) -> None:
    """EMA-update parameters and floating buffers, copy integral buffers exactly."""
    if not 0 <= tau <= 1:
        raise ValueError("EMA tau must lie in [0,1]")
    online_parameters = dict(model.online_encoder.named_parameters())
    target_parameters = dict(model.target_encoder.named_parameters())
    if online_parameters.keys() != target_parameters.keys():
        raise ValueError("online/target parameter structures differ")
    for name, target in target_parameters.items():
        target.mul_(tau).add_(online_parameters[name].detach(), alpha=1.0 - tau)
    online_buffers = dict(model.online_encoder.named_buffers())
    target_buffers = dict(model.target_encoder.named_buffers())
    if online_buffers.keys() != target_buffers.keys():
        raise ValueError("online/target buffer structures differ")
    for name, target in target_buffers.items():
        source = online_buffers[name].detach()
        if target.is_floating_point() or target.is_complex():
            target.mul_(tau).add_(source, alpha=1.0 - tau)
        else:
            target.copy_(source)


def normalized_parameter_distance(model: FieldJEPAKoopmanModel, eps: float = 1e-12) -> float:
    online = torch.cat(
        [parameter.detach().flatten() for parameter in model.online_encoder.parameters()]
    )
    target = torch.cat(
        [parameter.detach().flatten() for parameter in model.target_encoder.parameters()]
    )
    return float((online - target).norm() / online.norm().clamp_min(eps))
