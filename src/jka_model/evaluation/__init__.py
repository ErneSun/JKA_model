"""V0.3 direct-state and V0.4 learned-coordinate evaluation."""

from jka_model.evaluation.duffing_lifting import (
    DuffingLiftingDiagnostic,
    run_duffing_lifting_diagnostic,
)
from jka_model.evaluation.dynamics import (
    RolloutMetrics,
    evaluate_rollout,
    persistence_rollout,
)
from jka_model.evaluation.gates import (
    GateResult,
    GateStatus,
    MetricDirection,
    MetricGateSpec,
    aggregate_gate_results,
    evaluate_metric_gate,
)
from jka_model.evaluation.known_latent_experiment import (
    KnownLatentExperimentResult,
    run_known_latent_experiment,
    without_multi_step,
    without_reconstruction,
)
from jka_model.evaluation.representation import (
    LearnedRolloutMetrics,
    encode_records_for_alignment,
    evaluate_learned_trajectory,
)

__all__ = [
    "LearnedRolloutMetrics",
    "KnownLatentExperimentResult",
    "DuffingLiftingDiagnostic",
    "RolloutMetrics",
    "GateResult",
    "GateStatus",
    "MetricDirection",
    "MetricGateSpec",
    "aggregate_gate_results",
    "encode_records_for_alignment",
    "evaluate_learned_trajectory",
    "evaluate_rollout",
    "evaluate_metric_gate",
    "persistence_rollout",
    "run_known_latent_experiment",
    "run_duffing_lifting_diagnostic",
    "without_multi_step",
    "without_reconstruction",
]
from jka_model.evaluation.v0_6_diagnostics import (
    latent_statistics,
    latent_tracking_distance,
    model_tracking_diagnostics,
    near_identity_diagnostic,
)

__all__ += [
    "latent_statistics",
    "latent_tracking_distance",
    "model_tracking_diagnostics",
    "near_identity_diagnostic",
]
