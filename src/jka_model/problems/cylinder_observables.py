"""Cylinder-wake observables kept outside the generic Koopman core."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from jka_model.config import CylinderWake2DConfig, V09EvaluationConfig, V09TrainingConfig
from jka_model.data import (
    cylinder_force_coefficients,
    shedding_frequency,
    velocity_vorticity_divergence,
)
from jka_model.evaluation import MetricDirection, MetricGateSpec
from jka_model.observables import (
    ObservableLossResult,
    RobustObservableScaleState,
    deterministic_subsample,
    fit_robust_observable_scales,
    standardized_huber,
)


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
        self.scale_state: RobustObservableScaleState | None = None

    def set_scale_state(self, state: RobustObservableScaleState) -> None:
        missing = self._KNOWN_COMPONENTS - set(state.scales)
        if missing:
            raise ValueError(f"cylinder observable scales miss {sorted(missing)!r}")
        self.scale_state = state

    def fit_training_scales(
        self,
        records: Mapping[str, Any],
        trajectory_ids: tuple[str, ...],
        *,
        split_fingerprint: str,
    ) -> RobustObservableScaleState:
        """Fit physical scales only from records owned by the training split."""
        if not trajectory_ids or any(identifier not in records for identifier in trajectory_ids):
            raise ValueError("cylinder scale fit requires complete training trajectory ids")
        per_record = max(
            1,
            self.training.observable_scale_max_samples // len(trajectory_ids),
        )
        collected: dict[str, list[Tensor]] = {name: [] for name in self._KNOWN_COMPONENTS}
        for identifier in trajectory_ids:
            record = records[identifier]
            raw = record.states_raw.detach().float().cpu()
            valid = record.valid_mask
            if not isinstance(valid, Tensor):
                raise ValueError("cylinder scale fitting requires a valid fluid mask")
            valid = valid.bool().cpu()
            vorticity, divergence = velocity_vorticity_divergence(raw, self.config)
            drag, lift = cylinder_force_coefficients(raw, self.config)
            solid = ~valid
            values = {
                "velocity": raw[:, :2, valid],
                "vorticity": vorticity[:, valid],
                "divergence": divergence[:, valid],
                "boundary": raw[:, :2, solid],
                "lift": lift,
                "drag": drag,
            }
            for name, value in values.items():
                collected[name].append(deterministic_subsample(value, per_record))
        state = fit_robust_observable_scales(
            {name: torch.cat(values) for name, values in collected.items()},
            method=self.training.observable_scale_method,
            epsilon=self.training.observable_scale_epsilon,
            split_fingerprint=split_fingerprint,
            maximum_samples=self.training.observable_scale_max_samples,
            relative_floors={
                "divergence": ("vorticity", 1.0e-3),
                "boundary": ("velocity", 1.0e-3),
            },
        )
        self.set_scale_state(state)
        return state

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
        if self.training.phase1_enabled:
            if self.scale_state is None:
                raise RuntimeError("phase-1 observable loss requires train-only scales")
            predicted_vorticity, predicted_divergence = velocity_vorticity_divergence(
                predicted_raw, self.config
            )
            target_vorticity, _ = velocity_vorticity_divergence(target_raw, self.config)
            predicted_drag, predicted_lift = cylinder_force_coefficients(
                predicted_raw, self.config
            )
            target_drag, target_lift = cylinder_force_coefficients(target_raw, self.config)
            delta = self.training.observable_huber_delta
            components = {
                "velocity": standardized_huber(
                    predicted_raw[:, :2] - target_raw[:, :2],
                    self.scale_state.scale("velocity", predicted_raw),
                    delta=delta,
                    mask=valid_mask,
                ),
                "vorticity": standardized_huber(
                    predicted_vorticity - target_vorticity,
                    self.scale_state.scale("vorticity", predicted_raw),
                    delta=delta,
                    mask=valid_mask,
                ),
                "divergence": standardized_huber(
                    predicted_divergence,
                    self.scale_state.scale("divergence", predicted_raw),
                    delta=delta,
                    mask=valid_mask,
                ),
                "boundary": standardized_huber(
                    predicted_raw[:, :2],
                    self.scale_state.scale("boundary", predicted_raw),
                    delta=delta,
                    mask=~valid_mask,
                ),
                "lift": standardized_huber(
                    predicted_lift - target_lift,
                    self.scale_state.scale("lift", predicted_raw),
                    delta=delta,
                ),
                "drag": standardized_huber(
                    predicted_drag - target_drag,
                    self.scale_state.scale("drag", predicted_raw),
                    delta=delta,
                ),
            }
            total = predicted_raw.new_zeros(())
            terms: dict[str, Tensor] = {}
            for name, weight in self.weights.items():
                total = total + weight * components[name]
                terms[f"observable_{name}"] = components[name]
            return ObservableLossResult(total, terms)
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

    def force_window_loss(
        self,
        predicted_raw: Tensor,
        target_raw: Tensor,
        metadata: Mapping[str, Any],
    ) -> ObservableLossResult:
        """Causal waveform/correlation/spectrum loss on decoded force windows."""
        del metadata
        if self.scale_state is None:
            raise RuntimeError("force-window loss requires train-only scales")
        if predicted_raw.shape != target_raw.shape or predicted_raw.ndim != 5:
            raise ValueError("force windows must have shape [B,T,3,Nx,Ny]")
        batch, steps = predicted_raw.shape[:2]
        predicted_drag, predicted_lift = cylinder_force_coefficients(
            predicted_raw.reshape(-1, *predicted_raw.shape[2:]), self.config
        )
        target_drag, target_lift = cylinder_force_coefficients(
            target_raw.reshape(-1, *target_raw.shape[2:]), self.config
        )
        predicted_force = torch.stack((predicted_lift, predicted_drag), dim=-1).reshape(
            batch, steps, 2
        )
        target_force = torch.stack((target_lift, target_drag), dim=-1).reshape(
            batch, steps, 2
        )
        scales = predicted_force.new_tensor(
            [self.scale_state.scales["lift"], self.scale_state.scales["drag"]]
        )
        component_weights = predicted_force.new_tensor(
            [self.weights.get("lift", 0.0), self.weights.get("drag", 0.0)]
        )
        weight_sum = component_weights.sum()
        if not bool(weight_sum > 0):
            raise ValueError("force-window objective requires lift or drag weight")
        waveform_by_component = torch.nn.functional.huber_loss(
            (predicted_force - target_force) / scales,
            torch.zeros_like(predicted_force),
            delta=self.training.observable_huber_delta,
            reduction="none",
        )
        waveform = (
            waveform_by_component.mean(dim=(0, 1)) * component_weights
        ).sum() / weight_sum
        correlation = waveform.new_zeros(())
        spectrum = waveform.new_zeros(())
        if steps >= 2:
            predicted_centered = predicted_force - predicted_force.mean(dim=1, keepdim=True)
            target_centered = target_force - target_force.mean(dim=1, keepdim=True)
            numerator = (predicted_centered * target_centered).sum(dim=1)
            denominator = (
                predicted_centered.square().sum(dim=1).sqrt()
                * target_centered.square().sum(dim=1).sqrt()
            ).clamp_min(self.training.observable_scale_epsilon)
            target_norm = target_centered.square().sum(dim=1).sqrt()
            valid_correlation = target_norm > self.training.observable_scale_epsilon
            if bool(valid_correlation.any()):
                correlation_by_component = 1.0 - numerator / denominator
                active_weights = component_weights.unsqueeze(0).expand_as(
                    correlation_by_component
                ) * valid_correlation
                correlation = (correlation_by_component * active_weights).sum() / (
                    active_weights.sum().clamp_min(self.training.observable_scale_epsilon)
                )
        if steps >= 4:
            predicted_power = torch.fft.rfft(predicted_force.float(), dim=1).abs().square()
            target_power = torch.fft.rfft(target_force.float(), dim=1).abs().square()
            predicted_power = predicted_power / predicted_power.sum(dim=1, keepdim=True).clamp_min(
                self.training.observable_scale_epsilon
            )
            target_power = target_power / target_power.sum(dim=1, keepdim=True).clamp_min(
                self.training.observable_scale_epsilon
            )
            spectrum_by_component = (predicted_power - target_power).square().mean(
                dim=(0, 1)
            )
            spectrum = (spectrum_by_component * component_weights).sum() / weight_sum
        total = (
            waveform
            + self.training.force_correlation_weight * correlation
            + self.training.force_spectrum_weight * spectrum
        )
        return ObservableLossResult(
            total,
            {
                "observable_force_waveform": waveform,
                "observable_force_correlation": correlation,
                "observable_force_spectrum": spectrum,
            },
        )

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
