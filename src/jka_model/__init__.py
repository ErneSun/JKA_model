"""Public API through V0.4 learned continuous-time Koopman coordinates."""

from jka_model.constants import (
    ARCHITECTURE_REVISION,
    CHECKPOINT_SCHEMA_VERSION,
    PROJECT_VERSION,
)
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
from jka_model.training import TrainStage, configure_train_stage

__all__ = [
    "ARCHITECTURE_REVISION",
    "CHECKPOINT_SCHEMA_VERSION",
    "PROJECT_VERSION",
    "BoundarySpec",
    "ChannelSpec",
    "ContinuousKoopmanCore",
    "DtMode",
    "GeometrySpec",
    "GridSpec",
    "LatentState",
    "KoopmanAutoencoder",
    "KoopmanEncoder",
    "NormalizationSpec",
    "ProblemBatch",
    "ProblemSpec",
    "PhysicsConstraint",
    "TrainStage",
    "TrainingDecoder",
    "TransitionOutput",
    "configure_train_stage",
    "validate_trajectory_alignment",
]
