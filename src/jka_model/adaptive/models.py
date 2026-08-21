"""V0.9 context-conditioned low-rank continuous Koopman adaptation."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from jka_model.config import V09AdaptiveConfig
from jka_model.context.models import BaseContextEncoder


class LowRankAdaptiveOperator(nn.Module):
    """Map a validated context to ``A0 + U diag(eta) V^T``.

    The final eta layer is zero initialized, so every input exactly recovers A0 at
    initialization.  U/V column normalization removes the scale gauge from the factors;
    sign and permutation remain diagnostics rather than coordinate-level physics claims.
    """

    def __init__(
        self,
        nominal_generator: Tensor,
        context_dim: int,
        config: V09AdaptiveConfig,
    ) -> None:
        super().__init__()
        if nominal_generator.ndim != 2 or nominal_generator.shape[0] != nominal_generator.shape[1]:
            raise ValueError("nominal generator must be square")
        if not torch.isfinite(nominal_generator).all():
            raise ValueError("nominal generator must be finite")
        latent_dim = nominal_generator.shape[0]
        if config.rank >= latent_dim:
            raise ValueError("adaptive operator requires rank < latent dimension")
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.rank = config.rank
        self.condition_mode = config.condition_mode
        self.normalize_factors = config.normalize_factors
        self.register_buffer("nominal_generator", nominal_generator.detach().float().clone())

        generator = torch.Generator(device="cpu").manual_seed(1729 + config.rank)
        left = torch.randn(latent_dim, config.rank, generator=generator)
        right = torch.randn(latent_dim, config.rank, generator=generator)
        self.left_factor = nn.Parameter(torch.linalg.qr(left, mode="reduced").Q)
        self.right_factor = nn.Parameter(torch.linalg.qr(right, mode="reduced").Q)

        if config.condition_mode == "known":
            self.condition_embedding: nn.Module = nn.Sequential(
                nn.Linear(2, config.condition_embedding_dim),
                nn.SiLU(),
            )
            head_input = context_dim + config.condition_embedding_dim
        else:
            self.condition_embedding = nn.Identity()
            head_input = context_dim
        output = nn.Linear(config.width, config.rank)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.operator_coordinate_head = nn.Sequential(
            nn.Linear(head_input, config.width),
            nn.SiLU(),
            output,
        )

    def factors(self) -> tuple[Tensor, Tensor]:
        if not self.normalize_factors:
            return self.left_factor, self.right_factor
        return F.normalize(self.left_factor, dim=0), F.normalize(self.right_factor, dim=0)

    def coordinates(self, context: Tensor, condition: Tensor | None = None) -> Tensor:
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context must have shape [B,d_c]")
        if self.condition_mode == "known":
            if condition is None or condition.shape != (context.shape[0], 2):
                raise ValueError("known-condition mode requires current [Re,U] with shape [B,2]")
            inputs = torch.cat((context, self.condition_embedding(condition)), dim=-1)
        else:
            if condition is not None:
                raise ValueError("latent-inferred mode forbids condition input")
            inputs = context
        if not torch.isfinite(inputs).all():
            raise ValueError("adaptive-operator inputs must be finite")
        return self.operator_coordinate_head(inputs)

    def generator(
        self, context: Tensor, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        eta = self.coordinates(context, condition)
        left, right = self.factors()
        delta = torch.einsum("ir,br,jr->bij", left, eta, right)
        adapted = self.nominal_generator.unsqueeze(0) + delta
        return eta, delta, adapted

    def step(
        self,
        z: Tensor,
        context: Tensor,
        dt: Tensor,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError("z must have shape [B,d_K]")
        if dt.shape not in {(z.shape[0],), (z.shape[0], 1)} or torch.any(dt <= 0):
            raise ValueError("dt must be positive with shape [B] or [B,1]")
        eta, delta, adapted = self.generator(context, condition)
        with torch.autocast(device_type=z.device.type, enabled=False):
            transition = torch.linalg.matrix_exp(adapted.float() * dt.reshape(-1, 1, 1).float())
            prediction = torch.einsum("bij,bj->bi", transition, z.float())
        return prediction, eta, delta, adapted

    def orthogonality_loss(self) -> Tensor:
        left, right = self.factors()
        identity = torch.eye(self.rank, device=left.device, dtype=left.dtype)
        return (left.T @ left - identity).square().mean() + (
            right.T @ right - identity
        ).square().mean()


class AdaptiveKoopmanModel(nn.Module):
    """Frozen V0.8 context encoder followed by the sole trainable V0.9 adapter."""

    def __init__(
        self,
        context_encoder: BaseContextEncoder,
        operator_adapter: LowRankAdaptiveOperator,
    ) -> None:
        super().__init__()
        if context_encoder.context_dim != operator_adapter.context_dim:
            raise ValueError("context/operator dimensions disagree")
        if context_encoder.latent_dim != operator_adapter.latent_dim:
            raise ValueError("context/operator latent dimensions disagree")
        self.context_encoder = context_encoder
        self.operator_adapter = operator_adapter
        self.context_encoder.requires_grad_(False)
        self.context_encoder.eval()

    def train(self, mode: bool = True) -> AdaptiveKoopmanModel:
        super().train(mode)
        self.context_encoder.eval()
        return self

    def train_stage_modules(self) -> Mapping[str, nn.Module]:
        return {
            "context_encoder": self.context_encoder,
            "operator_adapter": self.operator_adapter,
        }

    def forward(
        self,
        history_z: Tensor,
        history_dts: Tensor,
        next_dt: Tensor,
        context_parameters: Tensor,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            context = self.context_encoder(
                history_z, history_dts, next_dt, context_parameters
            )
        prediction, eta, delta, adapted = self.operator_adapter.step(
            history_z[:, -1], context, next_dt, condition
        )
        return prediction, context, eta, delta, adapted


def operator_burden(delta: Tensor, nominal_generator: Tensor) -> Tensor:
    if delta.ndim != 3 or nominal_generator.ndim != 2:
        raise ValueError("operator burden expects delta[B,d,d] and nominal[d,d]")
    return torch.linalg.matrix_norm(delta, ord="fro", dim=(-2, -1)) / (
        torch.linalg.matrix_norm(nominal_generator, ord="fro") + 1e-12
    )


def symmetric_abscissa_proxy(generator: Tensor) -> Tensor:
    """Largest eigenvalue of the symmetric part; a growth-bound diagnostic, not causality."""
    if generator.ndim not in {2, 3} or generator.shape[-1] != generator.shape[-2]:
        raise ValueError("generator must be square or batched square")
    symmetric = 0.5 * (generator + generator.transpose(-1, -2))
    return torch.linalg.eigvalsh(symmetric)[..., -1]
