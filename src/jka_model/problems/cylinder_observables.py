"""Cylinder-wake observables kept outside the generic Koopman core."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor

from jka_model.config import CylinderWake2DConfig, V09EvaluationConfig, V09TrainingConfig
from jka_model.data import (
    cylinder_force_coefficients,
    shedding_frequency,
    velocity_vorticity_divergence,
)
from jka_model.evaluation import MetricDirection, MetricGateSpec
from jka_model.observables import ObservableLossResult


class CylinderWakeObservableObjective:
    """Local fields and global force observables for one cylinder-wake adapter."""

    name = "cylinder_wake_2d_observables"
    _KNOWN_COMPONENTS = {
        "velocity",
        "vorticity",
        "divergence",
        "boundary",
        "lift",
        "drag",
    }

    def __init__(
        self,
        config: CylinderWake2DConfig,
        training: V09TrainingConfig | None = None,
        evaluation: V09EvaluationConfig | None = None,
    ) -> None:
        self.config = config
        self.training = training or V09TrainingConfig()
        self.evaluation = evaluation or V09EvaluationConfig()
        if self.training.observable_names:
            unknown = set(self.training.observable_names) - self._KNOWN_COMPONENTS
            if unknown:
                raise ValueError(f"unknown cylinder observable component(s): {sorted(unknown)!r}")
            self.weights = dict(
                zip(
                    self.training.observable_names,
                    self.training.observable_component_weights,
                    strict=True,
                )
            )
        else:
            self.weights = {
                "velocity": self.training.physics_velocity_weight,
                "vorticity": self.training.physics_vorticity_weight,
                "divergence": self.training.physics_divergence_weight,
                "boundary": self.training.physics_boundary_weight,
                "lift": self.training.physics_lift_weight,
                "drag": self.training.physics_drag_weight,
            }

    @staticmethod
    def _relative_energy(prediction: Tensor, target: Tensor) -> Tensor:
        return (prediction - target).square().mean() / target.square().mean().clamp_min(1e-12)

    def training_loss(
        self,
        predicted_raw: Tensor,
        target_raw: Tensor,
        metadata: Mapping[str, Any],
    ) -> ObservableLossResult:
        valid_mask = metadata.get("valid_mask")
        expected_mask_shape = predicted_raw.shape[:1] + predicted_raw.shape[-2:]
        if not isinstance(valid_mask, Tensor) or valid_mask.shape != expected_mask_shape:
            raise ValueError("cylinder observable training requires valid_mask[B,Nx,Ny]")
        if predicted_raw.shape != target_raw.shape or predicted_raw.shape[-3:] != (
            3,
            self.config.nx,
            self.config.ny,
        ):
            raise ValueError("cylinder observable states must have shape [B,3,Nx,Ny]")
        fluid = valid_mask.unsqueeze(1).to(predicted_raw.dtype)
        solid = (~valid_mask).unsqueeze(1).to(predicted_raw.dtype)
        velocity = ((predicted_raw[:, :2] - target_raw[:, :2]).square() * fluid).sum() / (
            target_raw[:, :2].square() * fluid
        ).sum().clamp_min(1e-12)
        predicted_vorticity, predicted_divergence = velocity_vorticity_divergence(
            predicted_raw, self.config
        )
        target_vorticity, _ = velocity_vorticity_divergence(target_raw, self.config)
        fluid_scalar = valid_mask.to(predicted_raw.dtype)
        vorticity = (
            (predicted_vorticity - target_vorticity).square() * fluid_scalar
        ).sum() / (target_vorticity.square() * fluid_scalar).sum().clamp_min(1e-12)
        reference_gradient = (target_vorticity.square() * fluid_scalar).sum().div(
            fluid_scalar.sum().clamp_min(1.0)
        )
        divergence = (
            (predicted_divergence.square() * fluid_scalar)
            .sum()
            .div(fluid_scalar.sum().clamp_min(1.0))
            / reference_gradient.clamp_min(1e-12)
        )
        reference_velocity = (target_raw[:, :2].square() * fluid).sum().div(
            fluid.sum().clamp_min(1.0)
        )
        boundary = (
            (predicted_raw[:, :2].square() * solid)
            .sum()
            .div(solid.sum().clamp_min(1.0))
            / reference_velocity.clamp_min(1e-12)
        )
        predicted_drag, predicted_lift = cylinder_force_coefficients(
            predicted_raw, self.config
        )
        target_drag, target_lift = cylinder_force_coefficients(target_raw, self.config)
        components = {
            "velocity": velocity,
            "vorticity": vorticity,
            "divergence": divergence,
            "boundary": boundary,
            "lift": self._relative_energy(predicted_lift, target_lift),
            "drag": self._relative_energy(predicted_drag, target_drag),
        }
        total = predicted_raw.new_zeros(())
        terms: dict[str, Tensor] = {}
        for name, weight in self.weights.items():
            total = total + weight * components[name]
            terms[f"observable_{name}"] = components[name]
        return ObservableLossResult(total, terms)

    def evaluation_metrics(
        self,
        predicted_trajectory: Tensor,
        target_trajectory: Tensor,
        metadata: Mapping[str, Any],
    ) -> dict[str, float]:
        valid_mask = metadata.get("valid_mask")
        if (
            not isinstance(valid_mask, Tensor)
            or valid_mask.shape != predicted_trajectory.shape[-2:]
        ):
            raise ValueError("cylinder observable evaluation requires valid_mask[Nx,Ny]")
        vorticity, divergence = velocity_vorticity_divergence(
            predicted_trajectory, self.config
        )
        target_vorticity, _ = velocity_vorticity_divergence(
            target_trajectory, self.config
        )
        drag, lift = cylinder_force_coefficients(predicted_trajectory, self.config)
        target_drag, target_lift = cylinder_force_coefficients(
            target_trajectory, self.config
        )
        solid = ~valid_mask.to(predicted_trajectory.device)
        return {
            "velocity_relative_l2": float(
                (predicted_trajectory[:, :2] - target_trajectory[:, :2]).norm()
                / target_trajectory[:, :2].norm().clamp_min(1e-12)
            ),
            "vorticity_relative_l2": float(
                (vorticity - target_vorticity).norm()
                / target_vorticity.norm().clamp_min(1e-12)
            ),
            "divergence_rms": float(divergence.square().mean().sqrt()),
            "lift_rmse": float((lift - target_lift).square().mean().sqrt()),
            "drag_rmse": float((drag - target_drag).square().mean().sqrt()),
            "frequency_error": abs(
                shedding_frequency(lift, self.config.snapshot_dt)
                - shedding_frequency(target_lift, self.config.snapshot_dt)
            ),
            "boundary_no_slip_mse": float(
                predicted_trajectory[:, :2, solid].square().mean()
            ),
        }

    def evaluation_gate_specs(
        self,
        *,
        sequence_length: int,
        dt: float,
    ) -> Mapping[str, MetricGateSpec]:
        if sequence_length < 2 or dt <= 0:
            raise ValueError("observable gate resolution requires sequence_length>=2 and dt>0")
        relative = self.evaluation.max_physics_degradation
        lower = MetricDirection.LOWER_IS_BETTER
        specs = {
            name: MetricGateSpec(name, lower, relative_margin=relative)
            for name in (
                "velocity_relative_l2",
                "vorticity_relative_l2",
                "lift_rmse",
                "drag_rmse",
            )
        }
        specs["frequency_error"] = MetricGateSpec(
            "frequency_error",
            lower,
            relative_margin=relative,
            resolution_floor=(
                self.evaluation.frequency_resolution_bins / (sequence_length * dt)
            ),
        )
        specs["divergence_rms"] = MetricGateSpec(
            "divergence_rms",
            lower,
            threshold=self.evaluation.max_divergence_mse**0.5,
            relative_margin=relative,
        )
        specs["boundary_no_slip_mse"] = MetricGateSpec(
            "boundary_no_slip_mse",
            lower,
            threshold=self.evaluation.max_boundary_mse,
            relative_margin=relative,
        )
        return specs
