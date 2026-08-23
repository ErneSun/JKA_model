"""Auditable gradient-scale and direction diagnostics for multi-objective training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class GradientGeometry:
    names: tuple[str, ...]
    cosine: Tensor
    norms: Tensor

    @property
    def minimum_off_diagonal_cosine(self) -> float:
        if len(self.names) < 2:
            return 1.0
        mask = ~torch.eye(len(self.names), dtype=torch.bool)
        return float(self.cosine[mask].min())

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "cosine": self.cosine.tolist(),
            "gradient_norms": {
                name: float(value) for name, value in zip(self.names, self.norms, strict=True)
            },
            "minimum_off_diagonal_cosine": self.minimum_off_diagonal_cosine,
        }


def gradient_cosine_matrix(
    objectives: Mapping[str, Tensor],
    parameters: Sequence[nn.Parameter],
    *,
    epsilon: float = 1.0e-12,
) -> GradientGeometry:
    """Measure objective gradient geometry without modifying ``parameter.grad``."""
    active = [(name, value) for name, value in objectives.items() if value.requires_grad]
    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not active or not trainable:
        raise ValueError("gradient geometry requires objectives and trainable parameters")
    flattened: list[Tensor] = []
    for _, objective in active:
        gradients = torch.autograd.grad(
            objective,
            trainable,
            retain_graph=True,
            allow_unused=True,
        )
        flattened.append(
            torch.cat(
                [
                    torch.zeros_like(parameter).reshape(-1)
                    if gradient is None
                    else gradient.detach().float().reshape(-1)
                    for parameter, gradient in zip(trainable, gradients, strict=True)
                ]
            )
        )
    matrix = torch.stack(flattened)
    norms = matrix.norm(dim=1)
    cosine = matrix @ matrix.T
    cosine = cosine / (norms[:, None] * norms[None, :] + epsilon)
    cosine = cosine.clamp(-1.0, 1.0).cpu()
    return GradientGeometry(tuple(name for name, _ in active), cosine, norms.cpu())


@dataclass(frozen=True, slots=True)
class PCGradResult:
    task_names: tuple[str, ...]
    projected_conflicts: int
    compared_pairs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_names": list(self.task_names),
            "projected_conflicts": self.projected_conflicts,
            "compared_pairs": self.compared_pairs,
        }


def pcgrad_backward(
    objectives: Mapping[str, Tensor],
    parameters: Sequence[nn.Parameter],
    *,
    epsilon: float = 1.0e-12,
) -> PCGradResult:
    """Populate ``parameter.grad`` with deterministic-resume PCGrad projections.

    Projection order is sampled from PyTorch's captured RNG state, so exact resume
    remains reproducible while avoiding a fixed task-order bias.
    """
    active = [(name, value) for name, value in objectives.items() if value.requires_grad]
    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if len(active) < 2 or not trainable:
        raise ValueError("PCGrad requires at least two objectives and trainable parameters")
    task_gradients: list[Tensor] = []
    sizes = tuple(parameter.numel() for parameter in trainable)
    for _, objective in active:
        gradients = torch.autograd.grad(
            objective,
            trainable,
            retain_graph=True,
            allow_unused=True,
        )
        task_gradients.append(
            torch.cat(
                [
                    torch.zeros_like(parameter).reshape(-1)
                    if gradient is None
                    else gradient.reshape(-1)
                    for parameter, gradient in zip(trainable, gradients, strict=True)
                ]
            )
        )
    originals = tuple(task_gradients)
    projected = [gradient.clone() for gradient in originals]
    conflicts = 0
    comparisons = 0
    for index, gradient in enumerate(projected):
        order = torch.randperm(len(projected)).tolist()
        for other_index in order:
            if other_index == index:
                continue
            other = originals[other_index]
            denominator = other.square().sum()
            if float(denominator.detach()) <= epsilon:
                continue
            dot = torch.dot(gradient, other)
            comparisons += 1
            if float(dot.detach()) < 0:
                gradient = gradient - dot / denominator * other
                conflicts += 1
        projected[index] = gradient
    merged = torch.stack(projected).sum(dim=0)
    offset = 0
    for parameter, size in zip(trainable, sizes, strict=True):
        parameter.grad = merged[offset : offset + size].reshape_as(parameter).detach()
        offset += size
    return PCGradResult(
        tuple(name for name, _ in active),
        conflicts,
        comparisons,
    )
