"""Stage-aware parameter ownership contracts."""

from jka_model.training.stages import (
    TrainStage,
    assert_optimizer_matches_trainable_params,
    configure_train_stage,
    configure_trainable,
)

__all__ = [
    "TrainStage",
    "assert_optimizer_matches_trainable_params",
    "configure_train_stage",
    "configure_trainable",
]

