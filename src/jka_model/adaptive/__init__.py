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
from jka_model.adaptive.handoff import V08Handoff, V08SeedHandoff, audit_v0_8_handoff
from jka_model.adaptive.metrics import (
    latent_prediction_metrics,
    operator_explained_fraction,
    residual_decomposition,
)
from jka_model.adaptive.models import (
    AdaptiveKoopmanModel,
    LowRankAdaptiveOperator,
    operator_burden,
    symmetric_abscissa_proxy,
)
from jka_model.adaptive.objectives import (
    AdaptiveObjectiveResult,
    CurriculumState,
    adaptive_stabilization_objective,
    curriculum_state,
    differentiable_adaptive_rollout,
    relative_propagator_growth_loss,
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
    "AdaptiveObjectiveResult",
    "CurriculumState",
    "FrozenCylinderPhysics",
    "FrozenDecoderObservables",
    "PhysicalLossResult",
    "latent_prediction_metrics",
    "operator_burden",
    "operator_explained_fraction",
    "residual_decomposition",
    "symmetric_abscissa_proxy",
    "adaptive_training_scales",
    "adaptive_stabilization_objective",
    "adaptive_latent_rollout",
    "audit_v0_8_handoff",
    "curriculum_state",
    "differentiable_adaptive_rollout",
    "aggregate_v0_9_results",
    "build_adaptive_cache",
    "load_adaptive_cache",
    "load_adaptive_checkpoint",
    "save_adaptive_cache",
    "save_adaptive_checkpoint",
    "relative_propagator_growth_loss",
    "validate_adaptive_checkpoint",
]
