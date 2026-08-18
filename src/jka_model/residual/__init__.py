"""V0.7 Koopman residual identification and minimal closure."""

from jka_model.residual.assessment import assess_residual_structure
from jka_model.residual.cache import (
    ResidualCache,
    ResidualTrajectory,
    build_residual_cache,
    load_residual_cache,
    save_residual_cache,
)
from jka_model.residual.closures import (
    HistoryMLPClosure,
    InstantaneousMLPClosure,
    LinearClosure,
    ResidualKoopmanModel,
    ZeroClosure,
    analytical_mlp_parameter_count,
    build_closure,
    solve_parameter_matched_width,
)
from jka_model.residual.dataset import ResidualWindowDataset
from jka_model.residual.diagnostics import (
    classify_memory_evidence,
    closure_metrics,
    residual_statistics,
    residual_structure_metrics,
)
from jka_model.residual.memory import (
    classify_memory_sweep,
    compare_residual_memory_v0_7,
    load_evaluation_records,
    validate_sweep_provenance,
)
from jka_model.residual.rollout import corrected_latent_rollout
from jka_model.residual.synthetic import make_v0_7_synthetic_memory_cache

__all__ = [
    "HistoryMLPClosure",
    "InstantaneousMLPClosure",
    "LinearClosure",
    "ResidualCache",
    "ResidualKoopmanModel",
    "ResidualTrajectory",
    "ResidualWindowDataset",
    "ZeroClosure",
    "analytical_mlp_parameter_count",
    "assess_residual_structure",
    "build_closure",
    "build_residual_cache",
    "closure_metrics",
    "classify_memory_evidence",
    "classify_memory_sweep",
    "compare_residual_memory_v0_7",
    "corrected_latent_rollout",
    "load_residual_cache",
    "load_evaluation_records",
    "make_v0_7_synthetic_memory_cache",
    "residual_structure_metrics",
    "residual_statistics",
    "save_residual_cache",
    "solve_parameter_matched_width",
    "validate_sweep_provenance",
]
