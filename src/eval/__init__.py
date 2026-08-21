"""Versioned canonical evaluation entry points."""

from eval.evaluate_v0_5 import evaluate_v0_5
from eval.evaluate_v0_8 import evaluate_v0_8
from eval.evaluate_v0_9 import evaluate_v0_9

__all__ = ["evaluate_v0_5", "evaluate_v0_8", "evaluate_v0_9"]
