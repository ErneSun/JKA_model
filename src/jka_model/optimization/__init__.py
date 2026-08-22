"""Generic optimization utilities used by versioned scientific trainers."""

from jka_model.optimization.augmented_lagrangian import InequalityAugmentedLagrangian
from jka_model.optimization.gradient_geometry import GradientGeometry, gradient_cosine_matrix

__all__ = [
    "GradientGeometry",
    "InequalityAugmentedLagrangian",
    "gradient_cosine_matrix",
]
