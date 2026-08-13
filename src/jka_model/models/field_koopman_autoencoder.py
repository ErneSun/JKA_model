"""Small circular-CNN encoder/decoder around the continuous Koopman core."""

from __future__ import annotations

from torch import Tensor, nn

from jka_model.models.koopman_core import ContinuousKoopmanCore


class KoopmanEncoder2D(nn.Module):
    """Encode ``[...,C,Nx,Ny]`` fields with circular spatial padding."""

    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        width: int,
        nx: int,
        ny: int,
    ) -> None:
        super().__init__()
        if min(input_channels, latent_dim, width, nx, ny) < 1:
            raise ValueError("encoder dimensions must be positive")
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.nx, self.ny = nx, ny
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1, padding_mode="circular"),
            nn.SiLU(),
        )
        # Keep the coarse spatial feature map. Global average pooling would make the
        # code translation invariant and therefore erase the phase of travelling waves.
        coarse_nx, coarse_ny = (nx + 3) // 4, (ny + 3) // 4
        self.projection = nn.Linear(2 * width * coarse_nx * coarse_ny, latent_dim)

    def forward(self, field: Tensor) -> Tensor:
        if field.ndim < 4 or field.shape[-3] != self.input_channels:
            raise ValueError("field must have shape [...,C,Nx,Ny]")
        if field.shape[-2:] != (self.nx, self.ny):
            raise ValueError(f"field grid must be {(self.nx, self.ny)}")
        leading = field.shape[:-3]
        flattened = field.reshape(-1, *field.shape[-3:])
        features = self.network(flattened).flatten(1)
        return self.projection(features).reshape(*leading, self.latent_dim)


class TrainingDecoder2D(nn.Module):
    """Decode latents into a fixed-resolution scalar/vector field."""

    def __init__(
        self,
        latent_dim: int,
        output_channels: int,
        nx: int,
        ny: int,
        width: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if min(latent_dim, output_channels, nx, ny, width, hidden_dim) < 1:
            raise ValueError("decoder dimensions must be positive")
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.nx, self.ny, self.width = nx, ny, width
        self.lift = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, width * nx * ny)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv2d(width, output_channels, 3, padding=1, padding_mode="circular"),
        )

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim < 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError("latent must have shape [...,d]")
        leading = latent.shape[:-1]
        flat = latent.reshape(-1, self.latent_dim)
        field = self.lift(flat).reshape(-1, self.width, self.nx, self.ny)
        decoded = self.refine(field)
        return decoded.reshape(*leading, self.output_channels, self.nx, self.ny)


class FieldKoopmanAutoencoder(nn.Module):
    """Canonical V0.5 field model: CNN encoder, continuous core, train decoder."""

    def __init__(
        self,
        encoder: KoopmanEncoder2D,
        core: ContinuousKoopmanCore,
        decoder: TrainingDecoder2D,
    ) -> None:
        super().__init__()
        if encoder.latent_dim != core.state_dim or decoder.latent_dim != core.state_dim:
            raise ValueError("encoder/core/decoder latent dimensions must match")
        self.encoder, self.core, self.decoder = encoder, core, decoder

    def encode(self, field: Tensor) -> Tensor:
        # The core and matrix exponential are deliberately kept in FP32 under AMP.
        return self.encoder(field).to(dtype=self.core.A.dtype)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)

    def rollout(self, current_field: Tensor, future_dts: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(current_field)
        latent_rollout = self.core.rollout(latent, future_dts)
        predicted_fields = self.decode(latent_rollout[:, 1:])
        return predicted_fields, latent_rollout

    def train_stage_modules(self) -> dict[str, nn.Module]:
        return {
            "koopman_encoder": self.encoder,
            "koopman_core": self.core,
            "training_decoder": self.decoder,
        }
