"""Compact R2/R3 context encoders with a unified V0.8 downstream interface."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from jka_model.config import V08ContextConfig


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
    if history_z.shape[1:] != (history, latent_dim):
        raise ValueError(f"history_z must have shape [B,{history},{latent_dim}]")
    if history_dts.shape != (history_z.shape[0], history - 1):
        raise ValueError("history_dts alignment mismatch")
    if next_dt.shape != (history_z.shape[0], 1) or torch.any(next_dt <= 0):
        raise ValueError("next_dt must be positive with shape [B,1]")
    if parameters.shape != (history_z.shape[0], parameter_dim):
        raise ValueError("parameter shape mismatch")
    if not all(
        torch.isfinite(value).all() for value in (history_z, history_dts, next_dt, parameters)
    ):
        raise ValueError("context inputs must be finite")


class BaseContextEncoder(nn.Module):
    family = "base"

    def __init__(self, latent_dim: int, context_dim: int, history: int, parameter_dim: int) -> None:
        super().__init__()
        if min(latent_dim, context_dim, history) < 1 or parameter_dim < 0:
            raise ValueError("invalid context dimensions")
        if context_dim >= latent_dim:
            raise ValueError("V0.8 requires a strict context bottleneck d_c < d_K")
        self.latent_dim = latent_dim
        self.context_dim = context_dim
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


class InstantaneousContextEncoder(BaseContextEncoder):
    family = "instantaneous"

    def __init__(
        self, latent_dim: int, context_dim: int, history: int, parameter_dim: int, width: int
    ) -> None:
        super().__init__(latent_dim, context_dim, history, parameter_dim)
        self.network = nn.Sequential(
            nn.Linear(latent_dim + 1 + parameter_dim, width),
            nn.SiLU(),
            nn.Linear(width, context_dim),
        )

    def forward(
        self, history_z: Tensor, history_dts: Tensor, next_dt: Tensor, parameters: Tensor
    ) -> Tensor:
        self.validate(history_z, history_dts, next_dt, parameters)
        return self.network(torch.cat((history_z[:, -1], next_dt, parameters), dim=-1))


class ParameterMatchedInstantaneousContextEncoder(BaseContextEncoder):
    """Widened current-state control with capacity comparable to small Attention."""

    family = "instantaneous_matched"

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        history: int,
        parameter_dim: int,
        width: int,
    ) -> None:
        super().__init__(latent_dim, context_dim, history, parameter_dim)
        matched_width = 4 * width
        self.network = nn.Sequential(
            nn.Linear(latent_dim + 1 + parameter_dim, matched_width),
            nn.SiLU(),
            nn.Linear(matched_width, matched_width),
            nn.SiLU(),
            nn.Linear(matched_width, context_dim),
        )

    def forward(
        self,
        history_z: Tensor,
        history_dts: Tensor,
        next_dt: Tensor,
        parameters: Tensor,
    ) -> Tensor:
        self.validate(history_z, history_dts, next_dt, parameters)
        return self.network(torch.cat((history_z[:, -1], next_dt, parameters), dim=-1))


class HistoryMLPContextEncoder(BaseContextEncoder):
    family = "history_mlp"

    def __init__(
        self, latent_dim: int, context_dim: int, history: int, parameter_dim: int, width: int
    ) -> None:
        super().__init__(latent_dim, context_dim, history, parameter_dim)
        input_dim = history * latent_dim + history + parameter_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, context_dim),
        )

    def forward(
        self, history_z: Tensor, history_dts: Tensor, next_dt: Tensor, parameters: Tensor
    ) -> Tensor:
        self.validate(history_z, history_dts, next_dt, parameters)
        return self.network(
            torch.cat((history_z.flatten(1), history_dts, next_dt, parameters), dim=-1)
        )


class _CausalBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_multiplier * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_multiplier * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.norm1(values)
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=True,
            average_attn_weights=False,
        )
        values = values + self.dropout(attended)
        return values + self.dropout(self.ffn(self.norm2(values))), weights


class CausalAttentionContextEncoder(BaseContextEncoder):
    family = "attention"

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        history: int,
        parameter_dim: int,
        width: int,
        layers: int,
        heads: int,
        ffn_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__(latent_dim, context_dim, history, parameter_dim)
        if history < 2:
            raise ValueError("causal Attention requires history >= 2")
        self.token_projection = nn.Linear(latent_dim + 1 + parameter_dim, width)
        self.blocks = nn.ModuleList(
            [_CausalBlock(width, heads, ffn_multiplier, dropout) for _ in range(layers)]
        )
        self.output_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, context_dim))
        self.last_attention_weights: Tensor | None = None

    def encode_sequence(
        self, history_z: Tensor, history_dts: Tensor, next_dt: Tensor, parameters: Tensor
    ) -> Tensor:
        self.validate(history_z, history_dts, next_dt, parameters)
        token_dts = torch.cat((history_dts, next_dt), dim=1).unsqueeze(-1)
        conditions = parameters.unsqueeze(1).expand(-1, self.history, -1)
        values = self.token_projection(torch.cat((history_z, token_dts, conditions), dim=-1))
        mask = torch.triu(
            torch.ones(self.history, self.history, device=values.device, dtype=torch.bool),
            diagonal=1,
        )
        weights: Tensor | None = None
        for block in self.blocks:
            values, weights = block(values, mask)
        self.last_attention_weights = None if weights is None else weights.detach()
        return self.output_projection(values)

    def forward(
        self, history_z: Tensor, history_dts: Tensor, next_dt: Tensor, parameters: Tensor
    ) -> Tensor:
        return self.encode_sequence(history_z, history_dts, next_dt, parameters)[:, -1]


def _zero_output_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    output = nn.Linear(hidden_dim, output_dim)
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), output)


class DynamicContextModel(nn.Module):
    """Frozen-A0-compatible context, residual teacher head, and adequacy head."""

    def __init__(self, context_encoder: BaseContextEncoder, width: int) -> None:
        super().__init__()
        self.context_encoder = context_encoder
        self.residual_head = _zero_output_mlp(
            context_encoder.context_dim
            + context_encoder.latent_dim
            + 1
            + context_encoder.parameter_dim,
            width,
            context_encoder.latent_dim,
        )
        self.adequacy_head = _zero_output_mlp(context_encoder.context_dim, width, 1)

    def train_stage_modules(self) -> Mapping[str, nn.Module]:
        return {
            "context_encoder": self.context_encoder,
            "residual_head": self.residual_head,
            "adequacy_head": self.adequacy_head,
        }

    def forward(
        self,
        history_z: Tensor,
        history_dts: Tensor,
        next_dt: Tensor,
        parameters: Tensor,
        *,
        ablate_context: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        context = self.context_encoder(history_z, history_dts, next_dt, parameters)
        used_context = torch.zeros_like(context) if ablate_context else context
        residual = self.residual_head(
            torch.cat((used_context, history_z[:, -1], next_dt, parameters), dim=-1)
        )
        adequacy = self.adequacy_head(used_context)
        return context, residual, adequacy


def build_dynamic_context_model(
    config: V08ContextConfig,
    *,
    family: str,
    latent_dim: int,
    parameter_dim: int,
    history: int,
) -> DynamicContextModel:
    common = (latent_dim, config.context_dim, history, parameter_dim, config.width)
    if family == "instantaneous":
        encoder: BaseContextEncoder = InstantaneousContextEncoder(*common)
    elif family == "instantaneous_matched":
        encoder = ParameterMatchedInstantaneousContextEncoder(*common)
    elif family == "history_mlp":
        encoder = HistoryMLPContextEncoder(*common)
    elif family == "attention":
        encoder = CausalAttentionContextEncoder(
            latent_dim,
            config.context_dim,
            history,
            parameter_dim,
            config.width,
            config.layers,
            config.heads,
            config.ffn_multiplier,
            config.dropout,
        )
    else:
        raise ValueError(f"unknown V0.8 context family: {family!r}")
    return DynamicContextModel(encoder, config.width)
