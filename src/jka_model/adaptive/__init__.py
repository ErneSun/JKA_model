"""V0.9 adaptive Koopman public API."""

from jka_model.adaptive.cache import (
    AdaptiveCache,
    AdaptiveTrajectory,
    adaptive_training_scales,
    build_adaptive_cache,
    load_adaptive_cache,
    save_adaptive_cache,
)
from jka_model.adaptive.checkpoint import (
    load_adaptive_checkpoint,
    save_adaptive_checkpoint,
    validate_adaptive_checkpoint,
)
from jka_model.adaptive.dataset import AdaptiveRolloutDataset, AdaptiveWindowDataset
from jka_model.adaptive.error_attribution import observable_error_attribution
from jka_model.adaptive.handoff import V08Handoff, V08SeedHandoff, audit_v0_8_handoff
from jka_model.adaptive.identifiability import (
    MatchedHistoryPair,
    condition_observer_metrics,
    condition_targets,
    conditional_centering_loss,
    matched_history_pairs,
    phase2_condition_scales,
)
from jka_model.adaptive.metrics import (
    latent_prediction_metrics,
    operator_explained_fraction,
    residual_decomposition,
)
from jka_model.adaptive.models import (
    AdaptiveKoopmanModel,
    FactorizedAdaptiveOperator,
    LowRankAdaptiveOperator,
    causal_observer_features,
    operator_burden,
    symmetric_abscissa_proxy,
)
from jka_model.adaptive.objectives import (
    AdaptiveObjectiveResult,
    CurriculumState,
    Phase2TrainingState,
    adaptive_stabilization_objective,
    curriculum_state,
    differentiable_adaptive_rollout,
    phase2_training_state,
    relative_propagator_growth_loss,
)
from jka_model.adaptive.observer_admission import (
    OBSERVER_VARIANTS,
    classify_observer_admission,
    observer_history_variant,
)
from jka_model.adaptive.physics import (
    FrozenCylinderPhysics,
    FrozenDecoderObservables,
    PhysicalLossResult,
)
from jka_model.adaptive.reporting import aggregate_v0_9_results
from jka_model.adaptive.rollout import adaptive_latent_rollout

__all__ = [
    "AdaptiveKoopmanModel",
    "AdaptiveCache",
    "AdaptiveTrajectory",
    "AdaptiveRolloutDataset",
    "AdaptiveWindowDataset",
    "V08Handoff",
    "V08SeedHandoff",
    "LowRankAdaptiveOperator",
    "MatchedHistoryPair",
    "AdaptiveObjectiveResult",
    "CurriculumState",
    "Phase2TrainingState",
    "FrozenCylinderPhysics",
    "FrozenDecoderObservables",
    "FactorizedAdaptiveOperator",
    "PhysicalLossResult",
    "OBSERVER_VARIANTS",
    "latent_prediction_metrics",
    "operator_burden",
    "operator_explained_fraction",
    "observable_error_attribution",
    "residual_decomposition",
    "symmetric_abscissa_proxy",
    "adaptive_training_scales",
    "condition_observer_metrics",
    "causal_observer_features",
    "condition_targets",
    "classify_observer_admission",
    "conditional_centering_loss",
    "adaptive_stabilization_objective",
    "adaptive_latent_rollout",
    "audit_v0_8_handoff",
    "curriculum_state",
    "differentiable_adaptive_rollout",
    "phase2_training_state",
    "aggregate_v0_9_results",
    "build_adaptive_cache",
    "load_adaptive_cache",
    "load_adaptive_checkpoint",
    "matched_history_pairs",
    "observer_history_variant",
    "phase2_condition_scales",
    "save_adaptive_cache",
    "save_adaptive_checkpoint",
    "relative_propagator_growth_loss",
    "validate_adaptive_checkpoint",
]
