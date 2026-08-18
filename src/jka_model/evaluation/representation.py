"""Held-out reconstruction, alignment-pair, and closed-loop decoded evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from jka_model.data import ChannelStandardizer, TrajectoryRecord
from jka_model.evaluation.dynamics import evaluate_rollout
from jka_model.models import KoopmanAutoencoder


@dataclass(frozen=True, slots=True)
class LearnedRolloutMetrics:
    reconstruction_model_mse: float
    reconstruction_raw_mse: float
    one_step_latent_mse: float
    multi_step_latent_mse: float
    decoded_model_mse: float
    decoded_raw_mse: float
    persistence_model_mse: float
    persistence_raw_mse: float
    finite: bool


def encode_records_for_alignment(
    model: KoopmanAutoencoder,
    records: Sequence[TrajectoryRecord],
    normalizer: ChannelStandardizer,
    true_latents: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Create post-training alignment pairs; hidden state never enters model forward."""
    learned: list[Tensor] = []
    hidden: list[Tensor] = []
    with torch.no_grad():
        for record in records:
            if record.trajectory_id not in true_latents:
                raise ValueError("missing evaluation-only true latent")
            model_state = normalizer.transform(record.states_raw)
            z_k = model.encode(model_state)
            true_state = true_latents[record.trajectory_id].to(dtype=z_k.dtype, device=z_k.device)
            if true_state.shape[0] != z_k.shape[0]:
                raise ValueError("true and learned latent trajectory lengths differ")
            learned.append(z_k)
            hidden.append(true_state)
    return torch.cat(learned), torch.cat(hidden)


def evaluate_learned_trajectory(
    model: KoopmanAutoencoder,
    record: TrajectoryRecord,
    normalizer: ChannelStandardizer,
    *,
    horizon: int,
) -> tuple[LearnedRolloutMetrics, Tensor, Tensor, Tensor]:
    """Evaluate a closed-loop latent rollout and decode it in model/raw spaces."""
    if horizon < 1 or horizon > record.num_steps:
        raise ValueError("evaluation horizon is outside the trajectory")
    raw_truth = record.states_raw[: horizon + 1]
    model_truth = normalizer.transform(raw_truth)
    with torch.no_grad():
        reconstruction_model = model.reconstruct(model_truth)
        reconstruction_raw = normalizer.inverse_transform(reconstruction_model)
        latent, decoded_model_batch = model.rollout_decoded(
            model_truth[0].unsqueeze(0), record.dts[:horizon]
        )
        latent = latent[0]
        encoded_truth = model.encode(model_truth)
        one_step_latent = model.core.step(encoded_truth[:-1], record.dts[:horizon])
        decoded_model = decoded_model_batch[0]
        decoded_raw = normalizer.inverse_transform(decoded_model)
    model_metrics = evaluate_rollout(decoded_model, model_truth)
    raw_metrics = evaluate_rollout(decoded_raw, raw_truth)
    reconstruction_model_mse = (reconstruction_model - model_truth).square().mean()
    reconstruction_raw_mse = (reconstruction_raw - raw_truth).square().mean()
    one_step_latent_mse = (one_step_latent - encoded_truth[1:]).square().mean()
    multi_step_latent_mse = (latent - encoded_truth).square().mean()
    finite = bool(
        model_metrics.finite
        and raw_metrics.finite
        and torch.isfinite(latent).all()
        and torch.isfinite(reconstruction_model_mse)
        and torch.isfinite(reconstruction_raw_mse)
        and torch.isfinite(one_step_latent_mse)
        and torch.isfinite(multi_step_latent_mse)
    )
    return (
        LearnedRolloutMetrics(
            reconstruction_model_mse=float(reconstruction_model_mse.item()),
            reconstruction_raw_mse=float(reconstruction_raw_mse.item()),
            one_step_latent_mse=float(one_step_latent_mse.item()),
            multi_step_latent_mse=float(multi_step_latent_mse.item()),
            decoded_model_mse=model_metrics.rollout_mse,
            decoded_raw_mse=raw_metrics.rollout_mse,
            persistence_model_mse=model_metrics.persistence_mse,
            persistence_raw_mse=raw_metrics.persistence_mse,
            finite=finite,
        ),
        latent,
        decoded_model,
        decoded_raw,
    )
