"""Versioned canonical training entry points."""

from train.train_v0_5 import V05TrainingResult, initialize_v0_5_model, train_v0_5
from train.train_v0_8 import V08TrainingResult, train_v0_8

__all__ = [
    "V05TrainingResult",
    "V08TrainingResult",
    "initialize_v0_5_model",
    "train_v0_5",
    "train_v0_8",
]
