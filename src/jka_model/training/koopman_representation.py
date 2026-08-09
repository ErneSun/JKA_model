"""Minimal V0.4 training loop for encoder, decoder, and continuous generator only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torch.optim import Adam

from jka_model.contracts import ProblemBatch
from jka_model.data import TrajectoryWindowDataset, collate_problem_batches
from jka_model.losses import compute_representation_loss
from jka_model.models import (
    ContinuousKoopmanCore,
    KoopmanAutoencoder,
    KoopmanEncoder,
    TrainingDecoder,
)
from jka_model.training.stages import (
    TrainStage,
    assert_optimizer_matches_trainable_params,
    configure_train_stage,
)

if TYPE_CHECKING:
    from jka_model.config import (
        KoopmanAutoencoderConfig,
        RepresentationLossConfig,
        RepresentationTrainingConfig,
    )


@dataclass(frozen=True, slots=True)
class RepresentationTrainingSnapshot:
    epoch: int
    losses: dict[str, float]
    latent_mean: tuple[float, ...]
    latent_std: tuple[float, ...]
    minimum_latent_std: float
    maximum_latent_std: float


@dataclass(frozen=True, slots=True)
class RepresentationTrainingResult:
    initial_losses: dict[str, float]
    final_losses: dict[str, float]
    epochs: int
    global_step: int
    optimizer_state: dict[str, Any]
    gradient_norms: dict[str, float]
    diagnostic_history: tuple[RepresentationTrainingSnapshot, ...]


def initialize_koopman_autoencoder(
    architecture: KoopmanAutoencoderConfig,
    *,
    seed: int,
    init_scale: float,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> KoopmanAutoencoder:
    """Randomly initialize all learned components without oracle coordinates or A."""
    if seed < 0 or init_scale < 0:
        raise ValueError("seed and init_scale must be non-negative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = KoopmanEncoder(
            architecture.observation_dim,
            architecture.latent_dim,
            hidden_dim=architecture.hidden_dim,
            hidden_layers=architecture.encoder_hidden_layers,
            activation=architecture.activation,
            dtype=dtype,
            device=device,
        )
        decoder = TrainingDecoder(
            architecture.latent_dim,
            architecture.observation_dim,
            hidden_dim=architecture.hidden_dim,
            hidden_layers=architecture.decoder_hidden_layers,
            activation=architecture.activation,
            dtype=dtype,
            device=device,
        )
        for module in (encoder, decoder):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=init_scale)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    random = torch.Generator(device="cpu").manual_seed(seed + 1)
    generator = init_scale * torch.randn(
        (architecture.latent_dim, architecture.latent_dim),
        generator=random,
        dtype=dtype,
    )
    core = ContinuousKoopmanCore(
        architecture.latent_dim,
        generator=generator.to(device=device),
        trainable=True,
        dtype=dtype,
        device=device,
    )
    return KoopmanAutoencoder(encoder, core, decoder)


def make_window_batches(
    windows: TrajectoryWindowDataset,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
    shuffle: bool,
) -> list[ProblemBatch]:
    """Collate canonical windows without introducing a parallel dataset abstraction."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if shuffle:
        indices = torch.randperm(len(windows), generator=generator).tolist()
    else:
        indices = list(range(len(windows)))
    return [
        collate_problem_batches([windows[index] for index in indices[start : start + batch_size]])
        for start in range(0, len(indices), batch_size)
    ]


def _mean_losses(
    model: KoopmanAutoencoder,
    batches: list[ProblemBatch],
    loss_config: RepresentationLossConfig,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    with torch.no_grad():
        for batch in batches:
            values = compute_representation_loss(model, batch, loss_config).as_scalars()
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value * batch.future_dts.shape[0]
    count = sum(batch.future_dts.shape[0] for batch in batches)
    return {name: value / count for name, value in totals.items()}


def _gradient_norm(parameters: list[Tensor]) -> float:
    norms = [
        parameter.grad.detach().norm()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm().item())


def _training_snapshot(
    model: KoopmanAutoencoder,
    batches: list[ProblemBatch],
    loss_config: RepresentationLossConfig,
    *,
    epoch: int,
) -> RepresentationTrainingSnapshot:
    latent_values: list[Tensor] = []
    with torch.no_grad():
        for batch in batches:
            sequence = torch.cat(
                (
                    batch.context_states_model[:, -1:].detach(),
                    batch.future_states_model.detach(),
                ),
                dim=1,
            )
            latent_values.append(model.encode(sequence).reshape(-1, model.core.state_dim))
    values = torch.cat(latent_values, dim=0)
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    return RepresentationTrainingSnapshot(
        epoch=epoch,
        losses=_mean_losses(model, batches, loss_config),
        latent_mean=tuple(float(value) for value in mean.tolist()),
        latent_std=tuple(float(value) for value in std.tolist()),
        minimum_latent_std=float(std.min().item()),
        maximum_latent_std=float(std.max().item()),
    )


def train_koopman_representation(
    model: KoopmanAutoencoder,
    windows: TrajectoryWindowDataset,
    loss_config: RepresentationLossConfig,
    training_config: RepresentationTrainingConfig,
    *,
    seed: int,
    epochs: int | None = None,
) -> RepresentationTrainingResult:
    """Train exactly ``E_K + D_train + A`` on model-space ProblemBatch windows."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    effective_epochs = training_config.epochs if epochs is None else epochs
    if effective_epochs < 1:
        raise ValueError("epochs must be positive")
    configure_train_stage(model, TrainStage.KOOPMAN)
    optimizer = Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    assert_optimizer_matches_trainable_params(model, optimizer)
    fixed_batches = make_window_batches(
        windows,
        batch_size=training_config.batch_size,
        shuffle=False,
    )
    initial_snapshot = _training_snapshot(
        model, fixed_batches, loss_config, epoch=0
    )
    initial_losses = initial_snapshot.losses
    diagnostic_history = [initial_snapshot]
    random = torch.Generator(device="cpu").manual_seed(seed)
    global_step = 0
    model.train()
    for epoch_index in range(effective_epochs):
        batches = make_window_batches(
            windows,
            batch_size=training_config.batch_size,
            generator=random,
            shuffle=True,
        )
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            losses = compute_representation_loss(model, batch, loss_config)
            if not torch.isfinite(losses.total):
                raise FloatingPointError("V0.4 representation loss became non-finite")
            losses.total.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError("V0.4 gradients are missing or non-finite")
            optimizer.step()
            global_step += 1
        completed_epoch = epoch_index + 1
        if (
            completed_epoch % training_config.diagnostic_interval == 0
            or completed_epoch == effective_epochs
        ):
            diagnostic_history.append(
                _training_snapshot(
                    model,
                    fixed_batches,
                    loss_config,
                    epoch=completed_epoch,
                )
            )
    final_losses = diagnostic_history[-1].losses
    gradient_norms = {
        "encoder": _gradient_norm(list(model.encoder.parameters())),
        "decoder": _gradient_norm(list(model.decoder.parameters())),
        "generator": _gradient_norm([model.core.A]),
    }
    return RepresentationTrainingResult(
        initial_losses=initial_losses,
        final_losses=final_losses,
        epochs=effective_epochs,
        global_step=global_step,
        optimizer_state=optimizer.state_dict(),
        gradient_norms=gradient_norms,
        diagnostic_history=tuple(diagnostic_history),
    )
