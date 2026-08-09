"""Small numerical metrics for V0.3 direct-state dynamics."""

from jka_model.metrics.representation import (
    AffineLatentAlignment,
    LatentAlignmentMetrics,
    LatentDiagnostics,
    evaluate_affine_latent_alignment,
    fit_affine_latent_alignment,
    latent_diagnostics,
)
from jka_model.metrics.spectral import (
    SpectrumDiagnostics,
    continuous_spectrum,
    dominant_oscillatory_mode,
    relative_frequency_error,
    spectral_growth_rate,
)

__all__ = [
    "SpectrumDiagnostics",
    "AffineLatentAlignment",
    "LatentAlignmentMetrics",
    "LatentDiagnostics",
    "continuous_spectrum",
    "dominant_oscillatory_mode",
    "evaluate_affine_latent_alignment",
    "fit_affine_latent_alignment",
    "latent_diagnostics",
    "relative_frequency_error",
    "spectral_growth_rate",
]
