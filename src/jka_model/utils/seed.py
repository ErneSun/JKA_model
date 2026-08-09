"""Unified random seeding and exact RNG state capture/restore."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(slots=True)
class RNGState:
    """All RNG streams required for epoch-boundary exact resume."""

    python: tuple[Any, ...]
    numpy: tuple[Any, ...]
    torch_cpu: Tensor
    torch_cuda: tuple[Tensor, ...] | None

    def to_checkpoint_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "numpy": self.numpy,
            "torch_cpu": self.torch_cpu,
            "torch_cuda": self.torch_cuda,
        }

    @classmethod
    def from_checkpoint_dict(cls, data: Mapping[str, Any]) -> RNGState:
        required = {"python", "numpy", "torch_cpu", "torch_cuda"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"RNG state is missing field(s): {', '.join(sorted(missing))}")
        numpy_state = data["numpy"]
        python_state = data["python"]
        if not isinstance(python_state, tuple):
            raise ValueError("Python RNG state must be a tuple")
        if not isinstance(numpy_state, tuple):
            raise ValueError("NumPy RNG state must be a tuple")
        return cls(
            python=python_state,
            numpy=numpy_state,
            torch_cpu=data["torch_cpu"],
            torch_cuda=data["torch_cuda"],
        )


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch CPU, and every available CUDA device."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def capture_rng_state() -> RNGState:
    """Capture all supported RNG streams without modifying them."""
    cuda_state = tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None
    return RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=cuda_state,
    )


def restore_rng_state(state: RNGState) -> None:
    """Restore a previously captured RNG state exactly."""
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(state.torch_cuda) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count does not match captured RNG state")
        torch.cuda.set_rng_state_all(list(state.torch_cuda))
