"""V0.3 continuous core and V0.4 learned-coordinate vector models."""

from jka_model.models.field_jepa_koopman import (
    FieldJEPAKoopmanModel,
    ema_update_target,
    normalized_parameter_distance,
)
from jka_model.models.field_koopman_autoencoder import (
    FieldKoopmanAutoencoder,
    KoopmanEncoder2D,
    TrainingDecoder2D,
)
from jka_model.models.koopman_autoencoder import KoopmanAutoencoder
from jka_model.models.koopman_core import ContinuousKoopmanCore
from jka_model.models.koopman_encoder import KoopmanEncoder
from jka_model.models.training_decoder import TrainingDecoder

__all__ = [
    "ContinuousKoopmanCore",
    "KoopmanAutoencoder",
    "KoopmanEncoder",
    "TrainingDecoder",
    "FieldKoopmanAutoencoder",
    "FieldJEPAKoopmanModel",
    "KoopmanEncoder2D",
    "TrainingDecoder2D",
    "ema_update_target",
    "normalized_parameter_distance",
]
