"""V0.9 context-conditioned low-rank continuous Koopman adaptation."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from jka_model.config import V09AdaptiveConfig, V09Phase2Config
from jka_model.context.models import BaseContextEncoder


def causal_observer_features(history_z: Tensor, history_dts: Tensor) -> Tensor:
    """Causal state, history mean, and finite-time trend for condition observation.

    The frozen V0.8 context is optimized for residual prediction rather than physical
    parameter recovery.  Keeping these three summaries separate gives the observer
    direct access to the present state, its recent operating point, and its causal
    change rate without exposing any condition label.
    """
    if history_z.ndim != 3:
        raise ValueError("observer history must have shape [B,H,d_K]")
    batch, history, _ = history_z.shape
    if history_dts.shape != (batch, max(history - 1, 0)):
        raise ValueError("observer history/dt alignment mismatch")
    if not torch.isfinite(history_z).all() or not torch.isfinite(history_dts).all():
        raise ValueError("observer history inputs must be finite")
    current = history_z[:, -1]
    mean = history_z.mean(dim=1)
    if history > 1:
        elapsed = history_dts.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        trend = (history_z[:, -1] - history_z[:, 0]) / elapsed
    else:
        trend = torch.zeros_like(current)
    return torch.cat((current, mean, trend), dim=-1)


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
        self.bounded_coordinates = config.bounded_coordinates
        self.eta_max = config.eta_max
        self.trust_gate_enabled = config.trust_gate
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
        if config.trust_gate:
            gate = nn.Linear(head_input, 1)
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, config.trust_gate_bias)
            self.trust_gate_head: nn.Module | None = gate
        else:
            self.trust_gate_head = None

    def factors(self) -> tuple[Tensor, Tensor]:
        if not self.normalize_factors:
            return self.left_factor, self.right_factor
        return F.normalize(self.left_factor, dim=0), F.normalize(self.right_factor, dim=0)

    def _features(self, context: Tensor, condition: Tensor | None = None) -> Tensor:
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
        return inputs

    def adaptation_parameters(
        self, context: Tensor, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        features = self._features(context, condition)
        raw = self.operator_coordinate_head(features)
        coordinates = self.eta_max * torch.tanh(raw) if self.bounded_coordinates else raw
        gate = (
            torch.sigmoid(self.trust_gate_head(features))
            if self.trust_gate_head is not None
            else torch.ones((context.shape[0], 1), device=context.device, dtype=context.dtype)
        )
        return coordinates * gate, gate

    def coordinates(self, context: Tensor, condition: Tensor | None = None) -> Tensor:
        coordinates, _ = self.adaptation_parameters(context, condition)
        return coordinates

    def adaptation_gate(self, context: Tensor, condition: Tensor | None = None) -> Tensor:
        _, gate = self.adaptation_parameters(context, condition)
        return gate

    def generator_with_gate(
        self, context: Tensor, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        eta, gate = self.adaptation_parameters(context, condition)
        left, right = self.factors()
        delta = torch.einsum("ir,br,jr->bij", left, eta, right)
        adapted = self.nominal_generator.unsqueeze(0) + delta
        return eta, gate, delta, adapted

    def generator(
        self, context: Tensor, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        eta, _, delta, adapted = self.generator_with_gate(context, condition)
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
        prediction, eta, _, delta, adapted = self.step_with_gate(z, context, dt, condition)
        return prediction, eta, delta, adapted

    def step_with_gate(
        self,
        z: Tensor,
        context: Tensor,
        dt: Tensor,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError("z must have shape [B,d_K]")
        if dt.shape not in {(z.shape[0],), (z.shape[0], 1)} or torch.any(dt <= 0):
            raise ValueError("dt must be positive with shape [B] or [B,1]")
        eta, gate, delta, adapted = self.generator_with_gate(context, condition)
        with torch.autocast(device_type=z.device.type, enabled=False):
            transition = torch.linalg.matrix_exp(adapted.float() * dt.reshape(-1, 1, 1).float())
            prediction = torch.einsum("bij,bj->bi", transition, z.float())
        return prediction, eta, gate, delta, adapted

    def orthogonality_loss(self) -> Tensor:
        left, right = self.factors()
        identity = torch.eye(self.rank, device=left.device, dtype=left.dtype)
        return (left.T @ left - identity).square().mean() + (
            right.T @ right - identity
        ).square().mean()


class FactorizedAdaptiveOperator(nn.Module):
    """Condition branch plus condition-centered history innovation.

    The operator bases remain dyadic and low-rank:
    ``B_j = u^s_j (v^s_j)^T`` and ``C_k = u^d_k (v^d_k)^T``.  No
    additive state-space residual is introduced.
    """

    condition_dim = 3

    def __init__(
        self,
        nominal_generator: Tensor,
        context_dim: int,
        adaptive: V09AdaptiveConfig,
        phase2: V09Phase2Config,
    ) -> None:
        super().__init__()
        if nominal_generator.ndim != 2 or nominal_generator.shape[0] != nominal_generator.shape[1]:
            raise ValueError("nominal generator must be square")
        if not torch.isfinite(nominal_generator).all():
            raise ValueError("nominal generator must be finite")
        if phase2.static_rank + phase2.dynamic_rank != adaptive.rank:
            raise ValueError("Phase-2 ranks must sum to the adaptive rank")
        latent_dim = nominal_generator.shape[0]
        if adaptive.rank >= latent_dim:
            raise ValueError("factorized adaptive operator requires total rank < latent dimension")
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.rank = adaptive.rank
        self.static_rank = phase2.static_rank
        self.dynamic_rank = phase2.dynamic_rank
        self.observer_output_limit = phase2.observer_output_limit
        self.symmetric_delta_budget = phase2.symmetric_delta_budget
        self.condition_mode = adaptive.condition_mode
        self.normalize_factors = adaptive.normalize_factors
        self.bounded_coordinates = adaptive.bounded_coordinates
        self.eta_max = adaptive.eta_max
        self.trust_gate_enabled = adaptive.trust_gate
        self.register_buffer("nominal_generator", nominal_generator.detach().float().clone())

        generator = torch.Generator(device="cpu").manual_seed(2718 + adaptive.rank)
        left = torch.linalg.qr(
            torch.randn(latent_dim, adaptive.rank, generator=generator), mode="reduced"
        ).Q
        right = torch.linalg.qr(
            torch.randn(latent_dim, adaptive.rank, generator=generator), mode="reduced"
        ).Q
        split = self.static_rank
        self.static_left_factor = nn.Parameter(left[:, :split].clone())
        self.static_right_factor = nn.Parameter(right[:, :split].clone())
        self.dynamic_left_factor = nn.Parameter(left[:, split:].clone())
        self.dynamic_right_factor = nn.Parameter(right[:, split:].clone())

        observer_layers: list[nn.Module] = [
            nn.LayerNorm(context_dim + 3 * latent_dim)
        ]
        width_in = context_dim + 3 * latent_dim
        for _ in range(phase2.observer_depth):
            observer_layers.extend((nn.Linear(width_in, phase2.observer_width), nn.SiLU()))
            width_in = phase2.observer_width
        observer_layers.append(nn.Linear(width_in, self.condition_dim))
        self.condition_observer = nn.Sequential(*observer_layers)

        self.static_coordinate_head = self._zero_head(
            self.condition_dim, adaptive.width, self.static_rank
        )
        # The dynamic branch receives only the part of history context that
        # cannot be predicted from the current condition.  This implements the
        # factorization h = E[h|q] + h_perp explicitly instead of relying on a
        # subtractive coordinate head that can retain direct condition leakage.
        self.dynamic_context_mean_head = self._zero_head(
            self.condition_dim, adaptive.width, context_dim
        )
        dynamic_input = context_dim
        self.dynamic_coordinate_head = self._zero_head(
            dynamic_input, adaptive.width, self.dynamic_rank
        )
        if adaptive.trust_gate:
            static_gate = nn.Linear(self.condition_dim, 1)
            dynamic_gate = nn.Linear(dynamic_input, 1)
            for gate in (static_gate, dynamic_gate):
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, adaptive.trust_gate_bias)
            self.static_trust_gate_head: nn.Module | None = static_gate
            self.dynamic_trust_gate_head: nn.Module | None = dynamic_gate
        else:
            self.static_trust_gate_head = None
            self.dynamic_trust_gate_head = None

    @staticmethod
    def _zero_head(input_dim: int, width: int, output_dim: int) -> nn.Sequential:
        output = nn.Linear(width, output_dim)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        return nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(), output)

    def _factor_pair(self, kind: str) -> tuple[Tensor, Tensor]:
        if kind == "static":
            left, right = self.static_left_factor, self.static_right_factor
        elif kind == "dynamic":
            left, right = self.dynamic_left_factor, self.dynamic_right_factor
        else:
            raise ValueError("unknown Phase-2 operator basis")
        if self.normalize_factors:
            return F.normalize(left, dim=0), F.normalize(right, dim=0)
        return left, right

    def factors(self) -> tuple[Tensor, Tensor]:
        static = self._factor_pair("static")
        dynamic = self._factor_pair("dynamic")
        return torch.cat((static[0], dynamic[0]), dim=1), torch.cat((static[1], dynamic[1]), dim=1)

    def condition_prediction(
        self, context: Tensor, observer_features: Tensor | None = None
    ) -> Tensor:
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context must have shape [B,d_c]")
        if observer_features is None:
            observer_features = context.new_zeros((context.shape[0], 3 * self.latent_dim))
        if observer_features.shape != (context.shape[0], 3 * self.latent_dim):
            raise ValueError("observer features must have shape [B,3*d_K]")
        raw = self.condition_observer(torch.cat((context, observer_features), dim=-1))
        return self.observer_output_limit * torch.tanh(raw / self.observer_output_limit)

    @staticmethod
    def _limit_symmetric_growth(delta: Tensor, budget: float) -> tuple[Tensor, Tensor]:
        """Scale a low-rank increment so its logarithmic-norm burden is bounded."""
        symmetric = 0.5 * (delta + delta.transpose(-1, -2))
        # Frobenius norm upper-bounds the spectral norm and has a stable,
        # epsilon-regularized derivative at the exact zero initialization.
        burden = symmetric.square().sum(dim=(-2, -1)).add(1.0e-12).sqrt()
        scale = torch.clamp(budget / burden, max=1.0)
        return delta * scale[:, None, None], scale[:, None]

    def phase2_components(
        self,
        context: Tensor,
        condition: Tensor | None = None,
        *,
        dynamic_context: Tensor | None = None,
        observer_features: Tensor | None = None,
        condition_override: Tensor | None = None,
        active_components: str = "full",
        detach_static: bool = False,
        delta_budget: float | None = None,
    ) -> dict[str, Tensor]:
        if active_components not in {"static", "full"}:
            raise ValueError("Phase-2 active_components must be static or full")
        budget = self.symmetric_delta_budget if delta_budget is None else delta_budget
        if not math.isfinite(float(budget)) or budget <= 0:
            raise ValueError("Phase-2 total delta budget must be finite and positive")
        q_hat = self.condition_prediction(context, observer_features)
        if condition_override is not None:
            if condition_override.shape != q_hat.shape:
                raise ValueError("Phase-2 condition override shape mismatch")
            q_used = condition_override
        elif self.condition_mode == "known":
            if condition is None or condition.shape != q_hat.shape:
                raise ValueError("known Phase-2 mode requires normalized [Re,U,dRe/dt]")
            q_used = condition
        else:
            if condition is not None:
                raise ValueError("latent-inferred Phase-2 mode forbids condition input")
            # Q is a supervised physical observer, not a free latent coordinate.
            # Stop operator gradients from distorting its physical semantics.
            q_used = q_hat.detach()
        history_context = context if dynamic_context is None else dynamic_context
        if history_context.shape != context.shape:
            raise ValueError("Phase-2 dynamic context shape mismatch")
        static_raw = self.static_coordinate_head(q_used)
        condition_context = self.dynamic_context_mean_head(q_used)
        innovation_context = history_context - condition_context
        dynamic_raw = self.dynamic_coordinate_head(innovation_context)
        if self.bounded_coordinates:
            static_coordinates = self.eta_max * torch.tanh(static_raw)
            dynamic_coordinates = self.eta_max * torch.tanh(dynamic_raw)
        else:
            static_coordinates, dynamic_coordinates = static_raw, dynamic_raw
        dynamic_gate_inputs = innovation_context
        static_gate = (
            torch.sigmoid(self.static_trust_gate_head(q_used))
            if self.static_trust_gate_head is not None
            else torch.ones((context.shape[0], 1), device=context.device, dtype=context.dtype)
        )
        dynamic_gate = (
            torch.sigmoid(self.dynamic_trust_gate_head(dynamic_gate_inputs))
            if self.dynamic_trust_gate_head is not None
            else torch.ones((context.shape[0], 1), device=context.device, dtype=context.dtype)
        )
        static_coordinates = static_coordinates * static_gate
        dynamic_coordinates = dynamic_coordinates * dynamic_gate
        if active_components == "static":
            dynamic_coordinates = torch.zeros_like(dynamic_coordinates)
            dynamic_gate = torch.zeros_like(dynamic_gate)
        static_left, static_right = self._factor_pair("static")
        dynamic_left, dynamic_right = self._factor_pair("dynamic")
        static_delta = torch.einsum("ir,br,jr->bij", static_left, static_coordinates, static_right)
        dynamic_delta = torch.einsum(
            "ir,br,jr->bij", dynamic_left, dynamic_coordinates, dynamic_right
        )
        if detach_static:
            # Stage 2 is a residual fit: the dynamic branch sees the forecast
            # error left by the already identified condition branch, but cannot
            # rewrite that branch or its dyadic basis.
            static_coordinates = static_coordinates.detach()
            static_delta = static_delta.detach()
        raw_delta = static_delta + dynamic_delta
        _, total_stability_scale = self._limit_symmetric_growth(raw_delta, budget)
        static_delta = static_delta * total_stability_scale[:, None]
        dynamic_delta = dynamic_delta * total_stability_scale[:, None]
        delta = static_delta + dynamic_delta
        return {
            "q_hat": q_hat,
            "q_used": q_used,
            "static_coordinates": static_coordinates,
            "dynamic_coordinates": dynamic_coordinates,
            "history_context": history_context,
            "condition_context_prediction": condition_context,
            "innovation_context": innovation_context,
            "static_delta": static_delta,
            "dynamic_delta": dynamic_delta,
            "delta": delta,
            "gate": torch.maximum(static_gate, dynamic_gate),
            "static_gate": static_gate,
            "dynamic_gate": dynamic_gate,
            "static_stability_scale": total_stability_scale,
            "dynamic_stability_scale": total_stability_scale,
            "total_stability_scale": total_stability_scale,
            "delta_budget": delta.new_full((delta.shape[0], 1), budget),
        }

    def adaptation_parameters(
        self, context: Tensor, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        components = self.phase2_components(context, condition)
        coordinates = torch.cat(
            (components["static_coordinates"], components["dynamic_coordinates"]), dim=-1
        )
        return coordinates, components["gate"]

    def coordinates(self, context: Tensor, condition: Tensor | None = None) -> Tensor:
        return self.adaptation_parameters(context, condition)[0]

    def adaptation_gate(self, context: Tensor, condition: Tensor | None = None) -> Tensor:
        return self.adaptation_parameters(context, condition)[1]

    def generator_with_gate(
        self,
        context: Tensor,
        condition: Tensor | None = None,
        *,
        condition_override: Tensor | None = None,
        observer_features: Tensor | None = None,
        active_components: str = "full",
        detach_static: bool = False,
        delta_budget: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        components = self.phase2_components(
            context,
            condition,
            condition_override=condition_override,
            observer_features=observer_features,
            active_components=active_components,
            detach_static=detach_static,
            delta_budget=delta_budget,
        )
        coordinates = torch.cat(
            (components["static_coordinates"], components["dynamic_coordinates"]), dim=-1
        )
        delta = components["delta"]
        return (
            coordinates,
            components["gate"],
            delta,
            self.nominal_generator.unsqueeze(0) + delta,
        )

    def generator(
        self, context: Tensor, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        coordinates, _, delta, adapted = self.generator_with_gate(context, condition)
        return coordinates, delta, adapted

    def step(
        self,
        z: Tensor,
        context: Tensor,
        dt: Tensor,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        prediction, coordinates, _, delta, adapted = self.step_with_gate(z, context, dt, condition)
        return prediction, coordinates, delta, adapted

    def step_with_gate(
        self,
        z: Tensor,
        context: Tensor,
        dt: Tensor,
        condition: Tensor | None = None,
        *,
        condition_override: Tensor | None = None,
        observer_features: Tensor | None = None,
        active_components: str = "full",
        detach_static: bool = False,
        delta_budget: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError("z must have shape [B,d_K]")
        if dt.shape not in {(z.shape[0],), (z.shape[0], 1)} or torch.any(dt <= 0):
            raise ValueError("dt must be positive with shape [B] or [B,1]")
        coordinates, gate, delta, adapted = self.generator_with_gate(
            context,
            condition,
            condition_override=condition_override,
            observer_features=observer_features,
            active_components=active_components,
            detach_static=detach_static,
            delta_budget=delta_budget,
        )
        with torch.autocast(device_type=z.device.type, enabled=False):
            transition = torch.linalg.matrix_exp(adapted.float() * dt.reshape(-1, 1, 1).float())
            prediction = torch.einsum("bij,bj->bi", transition, z.float())
        return prediction, coordinates, gate, delta, adapted

    def orthogonality_loss(self, component: str = "full") -> Tensor:
        if component not in {"static", "dynamic", "full"}:
            raise ValueError("Phase-2 orthogonality component is invalid")
        result = self.nominal_generator.new_zeros(())
        for kind, rank in (("static", self.static_rank), ("dynamic", self.dynamic_rank)):
            if component != "full" and kind != component:
                continue
            left, right = self._factor_pair(kind)
            identity = torch.eye(rank, device=left.device, dtype=left.dtype)
            result = result + (left.T @ left - identity).square().mean()
            result = result + (right.T @ right - identity).square().mean()
        return result

    def cross_basis_orthogonality_loss(self, *, detach_static: bool = False) -> Tensor:
        static_left, static_right = self._factor_pair("static")
        dynamic_left, dynamic_right = self._factor_pair("dynamic")
        if detach_static:
            static_left = static_left.detach()
            static_right = static_right.detach()
        frobenius_inner = (static_left.T @ dynamic_left) * (static_right.T @ dynamic_right)
        return frobenius_inner.square().mean()


class AdaptiveKoopmanModel(nn.Module):
    """Frozen V0.8 context encoder followed by the sole trainable V0.9 adapter."""

    def __init__(
        self,
        context_encoder: BaseContextEncoder,
        operator_adapter: LowRankAdaptiveOperator | FactorizedAdaptiveOperator,
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
            context = self.context_encoder(history_z, history_dts, next_dt, context_parameters)
        if isinstance(self.operator_adapter, FactorizedAdaptiveOperator):
            prediction, eta, _, delta, adapted = self.operator_adapter.step_with_gate(
                history_z[:, -1],
                context,
                next_dt,
                condition,
                observer_features=causal_observer_features(history_z, history_dts),
            )
        else:
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
    if generator.ndim < 2 or generator.shape[-1] != generator.shape[-2]:
        raise ValueError("generator must have square trailing dimensions")
    symmetric = 0.5 * (generator + generator.transpose(-1, -2))
    return torch.linalg.eigvalsh(symmetric)[..., -1]
