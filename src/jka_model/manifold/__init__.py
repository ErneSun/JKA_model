"""V0.9 Phase-3 physical-manifold and representation audits."""

from jka_model.manifold.audit import (
    Phase3RepresentationAudit,
    audit_representation_checkpoint,
)
from jka_model.manifold.physical import (
    StreamFunctionPhysicalDecoder2D,
    central_difference_2d,
    physical_manifold_metrics,
)
from jka_model.manifold.routes import (
    PHASE3_ROUTES,
    MatchedRouteContract,
    assert_online_reencoding_required,
    configure_phase3_route,
)

__all__ = [
    "PHASE3_ROUTES",
    "MatchedRouteContract",
    "Phase3RepresentationAudit",
    "StreamFunctionPhysicalDecoder2D",
    "assert_online_reencoding_required",
    "audit_representation_checkpoint",
    "central_difference_2d",
    "configure_phase3_route",
    "physical_manifold_metrics",
]
