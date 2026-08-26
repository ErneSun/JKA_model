from __future__ import annotations

import torch
from torch import nn

from jka_model.config import ProjectConfig, V09Phase3Config, load_config
from jka_model.manifold import (
    MatchedRouteContract,
    StreamFunctionPhysicalDecoder2D,
    assert_online_reencoding_required,
    central_difference_2d,
    configure_phase3_route,
    physical_manifold_metrics,
)


def test_phase3_config_roundtrip_and_frozen_phase2_dependency() -> None:
    source = load_config("gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml")
    assert source.v0_9_phase3 == V09Phase3Config(
        enabled=True,
        source_phase2_result="v09-added-p2-physical-20260824T105209Z",
    )
    assert ProjectConfig.from_dict(source.to_dict()).stable_hash == source.stable_hash
    payload = source.to_dict()
    payload["v0_9_phase2"]["enabled"] = False
    try:
        ProjectConfig.from_dict(payload)
    except ValueError as error:
        assert "Phase-3 requires" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Phase-3 must not detach from its Phase-2 evidence contract")


def test_streamfunction_decoder_is_differentiable_and_interior_divergence_free() -> None:
    torch.manual_seed(7)
    model = StreamFunctionPhysicalDecoder2D(4, 12, 10, hidden_dim=8, dx=0.2, dy=0.3)
    latent = torch.randn(2, 4, requires_grad=True)
    valid = torch.ones(12, 10, dtype=torch.bool)
    field = model(latent, valid_mask=valid, inlet_velocity=torch.ones(2))
    divergence = central_difference_2d(field[:, 0], 0.2, -2) + central_difference_2d(
        field[:, 1], 0.3, -1
    )
    # Boundary lifting changes boundary stencils; the curl identity is audited in the interior.
    assert float(divergence[:, 2:-2, 2:-2].detach().abs().max()) < 2.0e-5
    assert torch.equal(field[:, 0, 0], torch.ones_like(field[:, 0, 0]))
    assert torch.equal(field[:, 1, 0], torch.zeros_like(field[:, 1, 0]))
    field.square().mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()


def test_physical_metrics_enforce_cylinder_no_slip() -> None:
    field = torch.zeros(1, 3, 8, 6)
    valid = torch.ones(8, 6, dtype=torch.bool)
    valid[3:5, 2:4] = False
    clean = physical_manifold_metrics(field, valid_mask=valid, dx=1.0, dy=1.0)
    assert clean["divergence_rms"] == 0
    assert clean["boundary_no_slip_mse"] == 0
    field[:, 0, 3:5, 2:4] = 1.0
    dirty = physical_manifold_metrics(field, valid_mask=valid, dx=1.0, dy=1.0)
    assert dirty["boundary_no_slip_mse"] > 0


def test_trainable_phase3_routes_reject_frozen_latent_cache() -> None:
    assert_online_reencoding_required("frozen", uses_frozen_latent_cache=True)
    for route in ("joint", "from_scratch"):
        try:
            assert_online_reencoding_required(route, uses_frozen_latent_cache=True)
        except ValueError as error:
            assert "online re-encoding" in str(error)
        else:  # pragma: no cover
            raise AssertionError("trainable representation route accepted stale latents")
        assert_online_reencoding_required(route, uses_frozen_latent_cache=False)


def test_phase3_route_ownership_and_matched_contract() -> None:
    modules = [nn.Linear(2, 2) for _ in range(4)]
    declaration = configure_phase3_route(
        "frozen",
        backbone=modules[0],
        context_encoder=modules[1],
        operator=modules[2],
        physical_decoder=modules[3],
    )
    assert declaration["backbone_frozen"]
    assert not any(
        parameter.requires_grad for module in modules for parameter in module.parameters()
    )
    reference = MatchedRouteContract("split", 47, 701, 80, ("a", "b"), "gates")
    reference.assert_matched(MatchedRouteContract("split", 47, 701, 80, ("a", "b"), "gates"))
    try:
        reference.assert_matched(
            MatchedRouteContract("split", 47, 809, 80, ("a", "b"), "gates")
        )
    except ValueError as error:
        assert "matched" in str(error)
    else:  # pragma: no cover
        raise AssertionError("mismatched operator seed was accepted")
