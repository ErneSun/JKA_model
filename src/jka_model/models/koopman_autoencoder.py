"""Thin V0.4 orchestration over encoder, V0.3 core, and training decoder."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

from torch import Tensor, nn

from jka_model.models.koopman_core import ContinuousKoopmanCore
from jka_model.models.koopman_encoder import KoopmanEncoder
from jka_model.models.training_decoder import TrainingDecoder


class KoopmanAutoencoder(nn.Module):
    """Compose learned coordinates with the unchanged continuous-time core."""

    def __init__(
        self,
        encoder: KoopmanEncoder,
        core: ContinuousKoopmanCore,
        decoder: TrainingDecoder,
    ) -> None:
        super().__init__()
        if encoder.latent_dim != core.state_dim or decoder.latent_dim != core.state_dim:
            raise ValueError("encoder/core/decoder latent dimensions must match")
        if encoder.input_dim != decoder.output_dim:
            raise ValueError("encoder input_dim must equal decoder output_dim")
        self.encoder = encoder
        self.core = core
        self.decoder = decoder

    def encode(self, state_model: Tensor) -> Tensor:
        return self.encoder(state_model)

    def train_stage_modules(self) -> Mapping[str, nn.Module]:
        """Expose the three and only three V0.4 optimizer ownership groups."""
        return {
            "koopman_encoder": self.encoder,
            "koopman_core": self.core,
            "training_decoder": self.decoder,
        }

    def decode(self, z_k: Tensor) -> Tensor:
        return self.decoder(z_k)

    def reconstruct(self, state_model: Tensor) -> Tensor:
        return self.decode(self.encode(state_model))

    def step(self, state_model: Tensor, dt: Tensor | Real) -> Tensor:
        """Encode one observation, propagate its latent, and decode the prediction."""
        return self.decode(self.core.step(self.encode(state_model), dt))

    def rollout_latent(
        self,
        state_model_0: Tensor,
        dts: Tensor | Real,
        horizon: int | None = None,
    ) -> Tensor:
        return self.core.rollout(self.encode(state_model_0), dts, horizon=horizon)

    def rollout_decoded(
        self,
        state_model_0: Tensor,
        dts: Tensor | Real,
        horizon: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        latent = self.rollout_latent(state_model_0, dts, horizon=horizon)
        return latent, self.decode(latent)
