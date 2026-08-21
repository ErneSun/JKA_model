"""Versioned canonical training entry points."""

from train.train_v0_5 import V05TrainingResult, initialize_v0_5_model, train_v0_5
from train.train_v0_8 import V08TrainingResult, train_v0_8
from train.train_v0_9 import V09TrainingResult, train_v0_9

__all__ = [
    "V05TrainingResult",
    "V08TrainingResult",
    "V09TrainingResult",
    "initialize_v0_5_model",
    "train_v0_5",
    "train_v0_8",
    "train_v0_9",
]
