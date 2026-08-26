"""Matched Phase-3 route contracts and trainability boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

PHASE3_ROUTES = ("frozen", "joint", "from_scratch")


@dataclass(frozen=True, slots=True)
class MatchedRouteContract:
    split_fingerprint: str
    backbone_seed: int
    operator_seed: int
    epochs: int
    trajectory_ids: tuple[str, ...]
    evaluation_gates_hash: str

    def __post_init__(self) -> None:
        if not self.split_fingerprint or not self.evaluation_gates_hash:
            raise ValueError("Phase-3 matched route requires split and gate provenance")
        if min(self.backbone_seed, self.operator_seed, self.epochs) < 0 or not self.trajectory_ids:
            raise ValueError("Phase-3 matched route metadata is incomplete")

    def assert_matched(self, other: MatchedRouteContract) -> None:
        if self != other:
            raise ValueError("Phase-3 routes do not share the matched scientific contract")


def assert_online_reencoding_required(route: str, *, uses_frozen_latent_cache: bool) -> None:
    """Representation-changing routes may never train against stale frozen latents."""
    if route not in PHASE3_ROUTES:
        raise ValueError("unknown Phase-3 route")
    if route != "frozen" and uses_frozen_latent_cache:
        raise ValueError(
            "joint/from_scratch Phase-3 routes require raw-field online re-encoding; "
            "a frozen V0.9 latent cache is scientifically invalid"
        )


def configure_phase3_route(
    route: str,
    *,
    backbone: nn.Module,
    context_encoder: nn.Module,
    operator: nn.Module,
    physical_decoder: nn.Module | None,
    joint_backbone_allowlist: tuple[str, ...] = (
        "online_encoder.projection",
        "training_decoder.refine.2",
    ),
) -> dict[str, Any]:
    """Apply explicit route ownership without silently replacing the frozen reference."""
    if route not in PHASE3_ROUTES:
        raise ValueError("unknown Phase-3 route")
    modules = {
        "backbone": backbone,
        "context_encoder": context_encoder,
        "operator": operator,
        "physical_decoder": physical_decoder,
    }
    for module in modules.values():
        if module is not None:
            module.requires_grad_(False)
    if route == "joint":
        selected = []
        for name, parameter in backbone.named_parameters():
            if any(name.startswith(prefix) for prefix in joint_backbone_allowlist):
                parameter.requires_grad_(True)
                selected.append(name)
        if not selected:
            raise ValueError("Phase-3 joint backbone allow-list selected no parameters")
        context_encoder.requires_grad_(True)
        operator.requires_grad_(True)
        if physical_decoder is not None:
            physical_decoder.requires_grad_(True)
    elif route == "from_scratch":
        for module in modules.values():
            if module is not None:
                reset = getattr(module, "reset_parameters", None)
                if callable(reset):
                    reset()
                else:
                    for child in module.modules():
                        if child is module:
                            continue
                        child_reset = getattr(child, "reset_parameters", None)
                        if callable(child_reset):
                            child_reset()
                module.requires_grad_(True)
    return {
        "route": route,
        "backbone_frozen": route == "frozen",
        "context_frozen": route == "frozen",
        "operator_trainable": route != "frozen",
        "physical_decoder_trainable": route != "frozen" and physical_decoder is not None,
        "requires_online_reencoding": route != "frozen",
        "inherits_v0_8_validation": route in {"frozen", "joint"},
        "joint_backbone_allowlist": list(joint_backbone_allowlist),
    }
