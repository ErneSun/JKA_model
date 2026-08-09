"""Trajectory-safe V0.2 data pipeline."""

from jka_model.data.datasets import (
    TrajectoryDataset,
    TrajectoryRecord,
    validate_trajectories_against_spec,
)
from jka_model.data.fingerprint import data_fingerprint
from jka_model.data.normalization import ChannelStandardizer
from jka_model.data.splits import SplitManifest, make_split_manifest, select_split
from jka_model.data.toy_advection_diffusion import (
    generate_advection_diffusion_trajectories,
    make_advection_diffusion_problem_spec,
)
from jka_model.data.toy_oscillators import (
    damped_oscillator_analytic_state,
    damped_oscillator_analytic_transition,
    damped_oscillator_generator_matrix,
    generate_damped_oscillator_trajectories,
    generate_duffing_trajectories,
    make_damped_oscillator_problem_spec,
    make_duffing_problem_spec,
    rotation_decay_transition,
    trajectory_transition_tensors,
)
from jka_model.data.windows import TrajectoryWindowDataset, collate_problem_batches

__all__ = [
    "ChannelStandardizer",
    "SplitManifest",
    "TrajectoryDataset",
    "TrajectoryRecord",
    "TrajectoryWindowDataset",
    "collate_problem_batches",
    "data_fingerprint",
    "damped_oscillator_analytic_state",
    "damped_oscillator_analytic_transition",
    "damped_oscillator_generator_matrix",
    "generate_advection_diffusion_trajectories",
    "generate_damped_oscillator_trajectories",
    "generate_duffing_trajectories",
    "make_advection_diffusion_problem_spec",
    "make_damped_oscillator_problem_spec",
    "make_duffing_problem_spec",
    "make_split_manifest",
    "select_split",
    "rotation_decay_transition",
    "trajectory_transition_tensors",
    "validate_trajectories_against_spec",
]
