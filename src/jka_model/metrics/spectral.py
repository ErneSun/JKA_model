"""Detached continuous-time eigenspectrum diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class SpectrumDiagnostics:
    """Continuous eigenvalues and their physical rate/frequency interpretations."""

    eigenvalues: Tensor
    growth_rates: Tensor
    angular_frequencies: Tensor
    frequencies_hz: Tensor

    def __post_init__(self) -> None:
        shapes = {
            tuple(self.eigenvalues.shape),
            tuple(self.growth_rates.shape),
            tuple(self.angular_frequencies.shape),
            tuple(self.frequencies_hz.shape),
        }
        if len(shapes) != 1 or self.eigenvalues.ndim != 1:
            raise ValueError("spectrum diagnostics must be aligned one-dimensional tensors")
        if any(tensor.requires_grad for tensor in self.as_tuple()):
            raise ValueError("spectrum diagnostics must be detached")

    def as_tuple(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.eigenvalues,
            self.growth_rates,
            self.angular_frequencies,
            self.frequencies_hz,
        )


def continuous_spectrum(generator: Tensor) -> SpectrumDiagnostics:
    """Extract detached continuous eigenvalues without imposing an arbitrary order."""
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError("generator must have shape [d,d]")
    if not torch.isfinite(generator).all():
        raise ValueError("generator must contain only finite values")
    eigenvalues = torch.linalg.eigvals(generator.detach())
    growth_rates = eigenvalues.real
    angular_frequencies = eigenvalues.imag.abs()
    frequencies_hz = angular_frequencies / (2.0 * math.pi)
    return SpectrumDiagnostics(
        eigenvalues=eigenvalues.detach(),
        growth_rates=growth_rates.detach(),
        angular_frequencies=angular_frequencies.detach(),
        frequencies_hz=frequencies_hz.detach(),
    )


def dominant_oscillatory_mode(spectrum: SpectrumDiagnostics) -> tuple[float, float]:
    """Return ``(growth_rate, angular_frequency)`` of the largest-|imag| mode."""
    if spectrum.eigenvalues.numel() == 0:
        raise ValueError("spectrum must contain at least one eigenvalue")
    index = int(torch.argmax(spectrum.angular_frequencies).item())
    return (
        float(spectrum.growth_rates[index].item()),
        float(spectrum.angular_frequencies[index].item()),
    )


def relative_frequency_error(estimated: float, reference: float) -> float:
    """Return the relative error between positive finite angular frequencies."""
    if not math.isfinite(estimated) or estimated < 0:
        raise ValueError("estimated frequency must be finite and non-negative")
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("reference frequency must be finite and positive")
    return abs(estimated - reference) / reference


def spectral_growth_rate(spectrum: SpectrumDiagnostics) -> float:
    """Return the largest continuous growth rate as a stability diagnostic."""
    if spectrum.growth_rates.numel() == 0:
        raise ValueError("spectrum must contain at least one eigenvalue")
    return float(spectrum.growth_rates.max().item())
