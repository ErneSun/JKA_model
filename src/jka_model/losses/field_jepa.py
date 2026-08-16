"""V0.6 JEPA objective with strict online/target semantic separation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor

from jka_model.config import FieldLossConfig, JEPALossConfig
from jka_model.contracts import ProblemBatch, ProblemSpec
from jka_model.data import ChannelStandardizer
from jka_model.losses.field_koopman import FieldLossBreakdown, compute_field_koopman_loss
from jka_model.models import FieldJEPAKoopmanModel
from jka_model.physics import PhysicsConstraint


@dataclass(frozen=True, slots=True)
class JEPALossBreakdown:
    total: Tensor
    v0_5: FieldLossBreakdown
    jepa_one_step: Tensor
    jepa_multi_step: Tensor

    def as_scalars(self) -> dict[str, float]:
        return {
            **self.v0_5.as_scalars(),
            "v0_5_total_loss": float(self.v0_5.total.detach()),
            "jepa_one_step_loss": float(self.jepa_one_step.detach()),
            "jepa_multi_step_loss": float(self.jepa_multi_step.detach()),
            "total_loss": float(self.total.detach()),
        }


def compute_field_jepa_loss(
    model: FieldJEPAKoopmanModel,
    batch: ProblemBatch,
    normalizer: ChannelStandardizer,
    spec: ProblemSpec,
    field_config: FieldLossConfig,
    jepa_config: JEPALossConfig,
    constraints: Mapping[str, PhysicsConstraint],
    *,
    physics_scale: float,
) -> JEPALossBreakdown:
    """Add EMA-target JEPA terms without changing any V0.5 target or physics term."""
    # The complete V0.5 objective calls model.encode() for current and future fields,
    # so every Koopman target remains an ONLINE encoding.
    v0_5 = compute_field_koopman_loss(
        model,  # type: ignore[arg-type]
        batch,
        normalizer,
        spec,
        field_config,
        constraints,
        physics_scale=physics_scale,
    )
    zero = v0_5.total.new_zeros(())
    if not jepa_config.enabled:
        return JEPALossBreakdown(v0_5.total, v0_5, zero, zero)
    current = batch.context_states_model[:, -1]
    future = batch.future_states_model
    z_k_online_current = model.encode(current)
    z_k_pred = model.koopman_core.rollout(z_k_online_current, batch.future_dts)[:, 1:]
    z_k_jepa_target = model.encode_target(future)
    if z_k_pred.shape != z_k_jepa_target.shape:
        raise ValueError("JEPA prediction and target latent shapes differ")
    jepa_one_step = (z_k_pred[:, 0] - z_k_jepa_target[:, 0]).square().mean()
    # k>=2 is a distinct closed-loop term. No teacher-forced latent enters the rollout.
    jepa_multi_step = (
        (z_k_pred[:, 1:] - z_k_jepa_target[:, 1:]).square().mean()
        if z_k_pred.shape[1] > 1
        else zero
    )
    total = (
        v0_5.total
        + jepa_config.lambda_one_step * jepa_one_step
        + jepa_config.lambda_multi_step * jepa_multi_step
    )
    return JEPALossBreakdown(total, v0_5, jepa_one_step, jepa_multi_step)
