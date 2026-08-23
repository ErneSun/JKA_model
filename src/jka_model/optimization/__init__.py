"""Generic optimization utilities used by versioned scientific trainers."""

from jka_model.optimization.augmented_lagrangian import InequalityAugmentedLagrangian
from jka_model.optimization.gradient_geometry import (
    GradientGeometry,
    PCGradResult,
    gradient_cosine_matrix,
    pcgrad_backward,
)

__all__ = [
    "GradientGeometry",
    "InequalityAugmentedLagrangian",
    "PCGradResult",
    "gradient_cosine_matrix",
    "pcgrad_backward",
]
