"""Public API through V0.9 context-conditioned adaptive Koopman learning."""

from jka_model.constants import (
    ARCHITECTURE_REVISION,
    CHECKPOINT_SCHEMA_VERSION,
    PROJECT_VERSION,
)
from jka_model.context import DynamicContextModel, build_dynamic_context_model
from jka_model.adaptive import AdaptiveKoopmanModel, LowRankAdaptiveOperator
from jka_model.contracts import (
    BoundarySpec,
    ChannelSpec,
    DtMode,
    GeometrySpec,
    GridSpec,
    LatentState,
    NormalizationSpec,
    ProblemBatch,
    ProblemSpec,
    TransitionOutput,
    validate_trajectory_alignment,
)
from jka_model.models import (
    ContinuousKoopmanCore,
    KoopmanAutoencoder,
    KoopmanEncoder,
    TrainingDecoder,
)
from jka_model.physics import PhysicsConstraint
from jka_model.residual import ResidualKoopmanModel, ResidualWindowDataset, build_closure
from jka_model.training import TrainStage, configure_train_stage

__all__ = [
    "ARCHITECTURE_REVISION",
    "CHECKPOINT_SCHEMA_VERSION",
    "PROJECT_VERSION",
    "BoundarySpec",
    "AdaptiveKoopmanModel",
    "ChannelSpec",
    "ContinuousKoopmanCore",
    "DynamicContextModel",
    "DtMode",
    "GeometrySpec",
    "GridSpec",
    "LatentState",
    "LowRankAdaptiveOperator",
    "KoopmanAutoencoder",
    "KoopmanEncoder",
    "NormalizationSpec",
    "ProblemBatch",
    "ProblemSpec",
    "PhysicsConstraint",
    "ResidualKoopmanModel",
    "ResidualWindowDataset",
    "TrainStage",
    "TrainingDecoder",
    "TransitionOutput",
    "build_closure",
    "build_dynamic_context_model",
    "configure_train_stage",
    "validate_trajectory_alignment",
]
