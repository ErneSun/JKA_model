"""V0.9 Phase-3 physical-manifold and representation audits."""

from jka_model.manifold.audit import (
    Phase3RepresentationAudit,
    audit_representation_checkpoint,
    classify_phase3_metrics,
    classify_phase3_route,
)
from jka_model.manifold.comparison import (
    classify_matched_phase3_run,
    matched_decoded_gains,
    nested_route_support,
)
from jka_model.manifold.joint import (
    JointMarkovObjectiveResult,
    MatureCheckpointTracker,
    RawFieldAdaptiveRolloutDataset,
    centered_linear_cka,
    classify_phase3_joint_run,
    decoded_physical_supervision,
    joint_markov_objective,
    move_raw_field_batch,
    orthogonal_procrustes_nrmse,
    phase3_checkpoint_key,
    representation_effective_rank,
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
    "centered_linear_cka",
    "decoded_physical_supervision",
    "classify_phase3_metrics",
    "classify_phase3_joint_run",
    "classify_matched_phase3_run",
    "classify_phase3_route",
    "central_difference_2d",
    "configure_phase3_route",
    "joint_markov_objective",
    "matched_decoded_gains",
    "move_raw_field_batch",
    "orthogonal_procrustes_nrmse",
    "nested_route_support",
    "phase3_checkpoint_key",
    "physical_manifold_metrics",
    "representation_effective_rank",
]
