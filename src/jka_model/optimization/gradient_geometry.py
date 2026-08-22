"""Auditable gradient-scale and direction diagnostics for multi-objective training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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
