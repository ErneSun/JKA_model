"""Stage contracts and minimal V0.3/V0.4 Koopman training loops."""

from jka_model.training.direct_koopman import (
    IdentificationResult,
    initialize_direct_koopman,
    one_step_mse,
    train_direct_koopman,
)
from jka_model.training.stages import (
    TrainStage,
    assert_optimizer_matches_trainable_params,
    configure_train_stage,
    configure_trainable,
)

__all__ = [
    "IdentificationResult",
    "TrainStage",
    "assert_optimizer_matches_trainable_params",
    "configure_train_stage",
    "configure_trainable",
    "initialize_direct_koopman",
    "one_step_mse",
    "train_direct_koopman",
]
