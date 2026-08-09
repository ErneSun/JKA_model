"""Tensor contracts with explicit state/action/time alignment."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor


def validate_trajectory_alignment(
    states: Tensor,
    actions: Tensor | None,
    dts: Tensor,
) -> None:
    """Validate the sole legal unbatched trajectory convention.

    Shapes:
        ``states``: ``[T + 1, ...]`` holding ``U_0 ... U_T``.
        ``actions``: optional ``[T, d_a]`` where ``a_i`` drives ``U_i -> U_{i+1}``.
        ``dts``: ``[T]`` where ``dt_i`` belongs to ``U_i -> U_{i+1}``.

    No tensor is reshaped, truncated, shifted, or otherwise corrected.
    """
    if states.ndim < 1 or dts.ndim != 1:
        raise ValueError("states must have a time axis and dts must have shape [T]")
    transitions = states.shape[0] - 1
    if transitions < 1:
        raise ValueError("a trajectory must contain at least two states")
    if dts.shape[0] != transitions:
        raise ValueError("states length must equal dts length + 1")
    if actions is not None:
        if actions.ndim != 2:
            raise ValueError("actions must have shape [T, d_a]")
        if actions.shape[0] != transitions:
            raise ValueError("states length must equal actions length + 1")
    if not torch.isfinite(dts).all() or not torch.all(dts > 0):
        raise ValueError("all dts must be finite and positive")


def _require_tensor(name: str, value: Any) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


@dataclass(slots=True)
class ProblemBatch:
    """One model-ready trajectory window with raw/model states kept separate.

    Shapes for grid or vector states (``*state`` includes channel and spatial axes):

    - ``context_states_raw/model``: ``[B, H, *state]``, ending at ``U_t``.
    - ``future_states_raw/model``: ``[B, K, *state]``, ``U_{t+1} ... U_{t+K}``.
    - ``history_actions``: optional ``[B, H-1, d_a]``.
    - ``future_actions``: optional ``[B, K, d_a]``; index 0 drives ``U_t -> U_{t+1}``.
    - ``history_dts``: ``[B, H-1]``.
    - ``future_dts``: ``[B, K]``; index 0 belongs to ``U_t -> U_{t+1}``.
    - ``mu_static``: optional ``[B, d_mu]``.

    ``states_raw`` are physical-unit values for future physical anchoring and metrics.
    ``states_model`` are normalized/preprocessed neural-network values. They are never
    substituted for one another by this contract.
    """

    context_states_raw: Tensor
    future_states_raw: Tensor
    context_states_model: Tensor
    future_states_model: Tensor
    history_dts: Tensor
    future_dts: Tensor
    history_actions: Tensor | None = None
    future_actions: Tensor | None = None
    mu_static: Tensor | None = None
    coordinates: Tensor | None = None
    cell_weights: Tensor | None = None
    valid_mask: Tensor | None = None
    trajectory_id: object | None = None

    def __post_init__(self) -> None:
        tensor_names = (
            "context_states_raw",
            "future_states_raw",
            "context_states_model",
            "future_states_model",
            "history_dts",
            "future_dts",
        )
        for name in tensor_names:
            _require_tensor(name, getattr(self, name))
        optional_tensors = (
            "history_actions",
            "future_actions",
            "mu_static",
            "coordinates",
            "cell_weights",
            "valid_mask",
        )
        for name in optional_tensors:
            value = getattr(self, name)
            if value is not None:
                _require_tensor(name, value)
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` on any alignment or shape contract violation."""
        state_tensors = (
            self.context_states_raw,
            self.future_states_raw,
            self.context_states_model,
            self.future_states_model,
        )
        if any(tensor.ndim < 3 for tensor in state_tensors):
            raise ValueError("state tensors must have shape [B, time, *state]")
        if self.context_states_raw.shape != self.context_states_model.shape:
            raise ValueError("raw and model context states must have identical shapes")
        if self.future_states_raw.shape != self.future_states_model.shape:
            raise ValueError("raw and model future states must have identical shapes")
        if self.context_states_raw.shape[0] != self.future_states_raw.shape[0]:
            raise ValueError("context and future batch sizes must match")
        if self.context_states_raw.shape[2:] != self.future_states_raw.shape[2:]:
            raise ValueError("context and future state shapes must match after the time axis")

        batch_size = self.context_states_raw.shape[0]
        history = self.context_states_raw.shape[1]
        horizon = self.future_states_raw.shape[1]
        if history < 1 or horizon < 1:
            raise ValueError("context history and future horizon must both be positive")
        if self.history_dts.shape != (batch_size, history - 1):
            raise ValueError("history_dts must have shape [B, H-1]")
        if self.future_dts.shape != (batch_size, horizon):
            raise ValueError("future_dts must have shape [B, K]")
        all_dts = torch.cat((self.history_dts, self.future_dts), dim=1)
        if not torch.isfinite(all_dts).all() or not torch.all(all_dts > 0):
            raise ValueError("all history and future dts must be finite and positive")

        if (self.history_actions is None) != (self.future_actions is None):
            raise ValueError(
                "history_actions and future_actions must both be present or both be None"
            )
        if self.history_actions is not None and self.future_actions is not None:
            if self.history_actions.ndim != 3 or self.future_actions.ndim != 3:
                raise ValueError("actions must have shape [B, time, d_a]")
            if self.history_actions.shape[:2] != (batch_size, history - 1):
                raise ValueError("history_actions must have shape [B, H-1, d_a]")
            if self.future_actions.shape[:2] != (batch_size, horizon):
                raise ValueError("future_actions must have shape [B, K, d_a]")
            if self.history_actions.shape[2] != self.future_actions.shape[2]:
                raise ValueError("history and future action dimensions must match")

        if self.mu_static is not None and (
            self.mu_static.ndim != 2 or self.mu_static.shape[0] != batch_size
        ):
            raise ValueError("mu_static must have shape [B, d_mu]")
        if isinstance(self.trajectory_id, (list, tuple)) and len(self.trajectory_id) != batch_size:
            raise ValueError("trajectory_id sequence length must equal batch size")

        # The concatenated view must always satisfy [T+1] states / [T] transitions.
        if self.states_raw.shape[1] != self.dts.shape[1] + 1:
            raise ValueError("states length must equal dts length + 1")
        if self.actions is not None and self.states_raw.shape[1] != self.actions.shape[1] + 1:
            raise ValueError("states length must equal actions length + 1")

    @property
    def states_raw(self) -> Tensor:
        """Concatenated physical-unit window, shape ``[B, H+K, *state]``."""
        return torch.cat((self.context_states_raw, self.future_states_raw), dim=1)

    @property
    def states_model(self) -> Tensor:
        """Concatenated model-input window, shape ``[B, H+K, *state]``."""
        return torch.cat((self.context_states_model, self.future_states_model), dim=1)

    @property
    def actions(self) -> Tensor | None:
        """Concatenated aligned transitions, shape ``[B, H-1+K, d_a]``."""
        if self.history_actions is None or self.future_actions is None:
            return None
        return torch.cat((self.history_actions, self.future_actions), dim=1)

    @property
    def dts(self) -> Tensor:
        """Concatenated aligned transition intervals, shape ``[B, H-1+K]``."""
        return torch.cat((self.history_dts, self.future_dts), dim=1)

    @property
    def parameters(self) -> Tensor | None:
        """Alias for static physical parameters ``mu_static``."""
        return self.mu_static

    @property
    def mask(self) -> Tensor | None:
        """Alias for the geometry/domain validity mask ``valid_mask``."""
        return self.valid_mask

    def to(self, *args: Any, **kwargs: Any) -> ProblemBatch:
        """Return a new batch with every tensor moved via ``Tensor.to``.

        Non-tensor trajectory identifiers are preserved unchanged. This method performs
        no reshape, normalization, or unit conversion.
        """
        values: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(*args, **kwargs) if isinstance(value, Tensor) else value
        return ProblemBatch(**values)


@dataclass(slots=True)
class LatentState:
    """Public latent naming contract; no encoder is implemented in V0.1.

    Shapes: ``z_k [B,d_k]``, optional ``z_phys [B,d_phys]`` and ``z_r [B,d_r]``.
    """

    z_k: Tensor
    z_phys: Tensor | None = None
    z_r: Tensor | None = None

    def __post_init__(self) -> None:
        if self.z_k.ndim != 2:
            raise ValueError("z_k must have shape [B, d_k]")
        for name in ("z_phys", "z_r"):
            value = getattr(self, name)
            if value is not None and (value.ndim != 2 or value.shape[0] != self.z_k.shape[0]):
                raise ValueError(f"{name} must have shape [B, d_{name[2:]}]")


@dataclass(slots=True)
class TransitionOutput:
    """Auditable transition decomposition for future model versions.

    ``z_k_base``, ``delta_z_k`` and ``z_k_next`` have shape ``[B,d_k]``;
    ``gate`` has shape ``[B,1]``; ``z_r`` is optional ``[B,d_r]``.
    """

    z_k_base: Tensor
    z_r: Tensor | None
    delta_z_k: Tensor
    gate: Tensor
    z_k_next: Tensor

    def __post_init__(self) -> None:
        if self.z_k_base.ndim != 2:
            raise ValueError("z_k_base must have shape [B, d_k]")
        expected = self.z_k_base.shape
        if self.delta_z_k.shape != expected or self.z_k_next.shape != expected:
            raise ValueError("z_k_base, delta_z_k, and z_k_next must have identical shapes")
        if self.gate.shape != (expected[0], 1):
            raise ValueError("gate must have shape [B, 1]")
        if self.z_r is not None and (self.z_r.ndim != 2 or self.z_r.shape[0] != expected[0]):
            raise ValueError("z_r must have shape [B, d_r]")
