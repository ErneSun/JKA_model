from __future__ import annotations

import torch

from jka_model.metrics import (
    continuous_spectrum,
    evaluate_affine_latent_alignment,
    fit_affine_latent_alignment,
    latent_diagnostics,
)


def test_latent_alignment_is_linear_transform_invariant() -> None:
    random = torch.Generator().manual_seed(9)
    hidden = torch.randn((100, 2), generator=random, dtype=torch.float64)
    transform = torch.tensor([[1.5, -0.3], [0.4, 0.8]], dtype=torch.float64)
    offset = torch.tensor([0.2, -0.1], dtype=torch.float64)
    learned = hidden @ transform.T + offset
    alignment = fit_affine_latent_alignment(learned[:70], hidden[:70])
    metrics = evaluate_affine_latent_alignment(alignment, learned[70:], hidden[70:])
    assert metrics.r2 > 1.0 - 1e-12
    assert metrics.mse < 1e-24


def test_spectrum_is_similarity_invariant() -> None:
    generator = torch.tensor([[-0.1, -1.3], [1.3, -0.1]], dtype=torch.float64)
    transform = torch.tensor([[1.2, 0.4], [-0.3, 0.9]], dtype=torch.float64)
    similar = transform @ generator @ torch.linalg.inv(transform)
    original_spectrum = continuous_spectrum(generator)
    similar_spectrum = continuous_spectrum(similar)
    torch.testing.assert_close(
        original_spectrum.growth_rates.sort().values,
        similar_spectrum.growth_rates.sort().values,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        original_spectrum.angular_frequencies.sort().values,
        similar_spectrum.angular_frequencies.sort().values,
        atol=1e-12,
        rtol=1e-12,
    )


def test_latent_diagnostics_population_statistics() -> None:
    latent = torch.tensor(
        [[[-1.0, 0.0], [1.0, 2.0]], [[-1.0, 2.0], [1.0, 0.0]]],
        dtype=torch.float64,
    )
    diagnostics = latent_diagnostics(latent)
    torch.testing.assert_close(diagnostics.mean, torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(diagnostics.std, torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert diagnostics.minimum_std == 1.0
