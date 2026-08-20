"""V0.8 residual-supervised compact dynamic-context learning."""

from jka_model.context.dataset import ContextWindowDataset, residual_training_scales
from jka_model.context.diagnostics import context_diagnostics, context_prediction_metrics
from jka_model.context.models import (
    CausalAttentionContextEncoder,
    DynamicContextModel,
    HistoryMLPContextEncoder,
    InstantaneousContextEncoder,
    ParameterMatchedInstantaneousContextEncoder,
    build_dynamic_context_model,
)
from jka_model.context.reporting import aggregate_v0_8_results
from jka_model.context.rollout import context_corrected_latent_rollout
from jka_model.context.routing import ContextRoute, load_v0_7_route, select_context_family

__all__ = [
    "CausalAttentionContextEncoder",
    "ContextRoute",
    "ContextWindowDataset",
    "DynamicContextModel",
    "HistoryMLPContextEncoder",
    "InstantaneousContextEncoder",
    "ParameterMatchedInstantaneousContextEncoder",
    "build_dynamic_context_model",
    "aggregate_v0_8_results",
    "context_diagnostics",
    "context_prediction_metrics",
    "context_corrected_latent_rollout",
    "load_v0_7_route",
    "residual_training_scales",
    "select_context_family",
]
