"""Public API for the V0.1 project skeleton and contracts."""

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
from jka_model.training import TrainStage, configure_train_stage

__all__ = [
    "ARCHITECTURE_REVISION",
    "CHECKPOINT_SCHEMA_VERSION",
    "PROJECT_VERSION",
    "BoundarySpec",
    "ChannelSpec",
    "DtMode",
    "GeometrySpec",
    "GridSpec",
    "LatentState",
    "NormalizationSpec",
    "ProblemBatch",
    "ProblemSpec",
    "TrainStage",
    "TransitionOutput",
    "configure_train_stage",
    "validate_trajectory_alignment",
]

