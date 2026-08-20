"""Scientific problem adapters."""

from jka_model.problems.advection_diffusion_2d import AdvectionDiffusion2DProblemAdapter
from jka_model.problems.base import ProblemAdapter
from jka_model.problems.cylinder_wake_2d import CylinderWake2DProblemAdapter
from jka_model.problems.registry import create_problem_adapter, register_problem_adapter

__all__ = [
    "AdvectionDiffusion2DProblemAdapter",
    "CylinderWake2DProblemAdapter",
    "ProblemAdapter",
    "create_problem_adapter",
    "register_problem_adapter",
]
