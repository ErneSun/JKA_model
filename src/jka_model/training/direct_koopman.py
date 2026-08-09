"""Minimal full-batch system identification for the V0.3 generator matrix only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from jka_model.models import ContinuousKoopmanCore

if TYPE_CHECKING:
    from jka_model.config import DirectIdentificationConfig


@dataclass(frozen=True, slots=True)
class IdentificationResult:
    """Finite loss trace and optimizer state from one direct-state fit."""

    initial_loss: float
    final_loss: float
    epochs: int
    global_step: int
    optimizer_state: dict[str, Any]


def initialize_direct_koopman(
    state_dim: int,
    *,
    seed: int,
    init_scale: float,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> ContinuousKoopmanCore:
    """Deterministically initialize away from the unknown true generator."""
    if seed < 0 or init_scale < 0:
        raise ValueError("seed and init_scale must be non-negative")
    random = torch.Generator(device="cpu").manual_seed(seed)
    matrix = init_scale * torch.randn((state_dim, state_dim), generator=random, dtype=dtype)
    matrix = matrix.to(device=device)
    return ContinuousKoopmanCore(
        state_dim,
        generator=matrix,
        trainable=True,
        dtype=dtype,
        device=device,
    )


def one_step_mse(
    core: ContinuousKoopmanCore,
    states: Tensor,
    targets: Tensor,
    dts: Tensor,
) -> Tensor:
    """The sole V0.3 training loss: matrix-exponential one-step state MSE."""
    if states.ndim != 2 or targets.shape != states.shape:
        raise ValueError("states and targets must share shape [N,d]")
    if dts.shape != (states.shape[0],):
        raise ValueError("dts must have shape [N]")
    return (core.step(states, dts) - targets).square().mean()


def train_direct_koopman(
    core: ContinuousKoopmanCore,
    states: Tensor,
    targets: Tensor,
    dts: Tensor,
    config: DirectIdentificationConfig,
) -> IdentificationResult:
    """Run ``zero_grad -> matrix_exp loss -> backward -> Adam.step``."""
    if not core.trainable:
        raise ValueError("direct identification requires a trainable KoopmanCore")
    if states.device != core.A.device or states.dtype != core.A.dtype:
        raise ValueError("training tensors must match core device/dtype")
    if targets.device != states.device or targets.dtype != states.dtype:
        raise ValueError("targets must match states device/dtype")
    if dts.device != states.device or dts.dtype != states.dtype:
        raise ValueError("dts must match states device/dtype")
    optimizer = torch.optim.Adam(
        core.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    initial_loss = float(one_step_mse(core, states, targets, dts).detach().item())
    final_loss = initial_loss
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = one_step_mse(core, states, targets, dts)
        if not torch.isfinite(loss):
            raise FloatingPointError("direct Koopman loss became non-finite")
        loss.backward()
        if core.A.grad is None or not torch.isfinite(core.A.grad).all():
            raise FloatingPointError("direct Koopman generator gradient is missing/non-finite")
        optimizer.step()
    with torch.no_grad():
        final_loss_tensor = one_step_mse(core, states, targets, dts)
    if not torch.isfinite(final_loss_tensor):
        raise FloatingPointError("final direct Koopman loss is non-finite")
    final_loss = float(final_loss_tensor.item())
    return IdentificationResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        epochs=config.epochs,
        global_step=config.epochs,
        optimizer_state=optimizer.state_dict(),
    )
