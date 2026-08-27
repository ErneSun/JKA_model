"""V0.9 Phase-3 physical-manifold and representation audits."""

from jka_model.manifold.audit import (
    Phase3RepresentationAudit,
    audit_representation_checkpoint,
    classify_phase3_metrics,
    classify_phase3_route,
)
from jka_model.manifold.joint import (
    JointMarkovObjectiveResult,
    MatureCheckpointTracker,
    RawFieldAdaptiveRolloutDataset,
    classify_phase3_joint_run,
    joint_markov_objective,
    move_raw_field_batch,
    phase3_checkpoint_key,
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
    "JointMarkovObjectiveResult",
    "MatureCheckpointTracker",
    "Phase3RepresentationAudit",
    "RawFieldAdaptiveRolloutDataset",
    "StreamFunctionPhysicalDecoder2D",
    "assert_online_reencoding_required",
    "audit_representation_checkpoint",
    "classify_phase3_metrics",
    "classify_phase3_joint_run",
    "classify_phase3_route",
    "central_difference_2d",
    "configure_phase3_route",
    "joint_markov_objective",
    "move_raw_field_batch",
    "phase3_checkpoint_key",
    "physical_manifold_metrics",
]
