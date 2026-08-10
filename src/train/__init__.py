"""Versioned canonical training entry points."""

from train.train_v0_5 import V05TrainingResult, initialize_v0_5_model, train_v0_5

__all__ = ["V05TrainingResult", "initialize_v0_5_model", "train_v0_5"]
