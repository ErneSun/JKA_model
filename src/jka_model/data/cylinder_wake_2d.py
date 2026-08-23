"""Fixed-condition transient cylinder-wake trajectories from a compact D2Q9 solver.

The solver is an offline benchmark generator, not a production CFD framework.  It
uses a low-Mach BGK lattice-Boltzmann discretization whose hydrodynamic limit is the
two-dimensional incompressible Navier--Stokes system.  All learned models consume
only saved trajectories; CFD is never executed inside a training epoch.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from jka_model.config import CylinderWake2DConfig, V09ConditionConfig
from jka_model.contracts import (
    BoundarySpec,
    ChannelSpec,
    DtMode,
    GeometrySpec,
    GridSpec,
    NormalizationSpec,
    ProblemSpec,
)
from jka_model.data.datasets import (
    TrajectoryDataset,
    TrajectoryRecord,
    validate_trajectories_against_spec,
)

_C = torch.tensor(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=torch.int64,
)
_W = torch.tensor([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
_OPPOSITE = (0, 3, 4, 1, 2, 7, 8, 5, 6)


@dataclass(frozen=True, slots=True)
class CylinderWakeDataset:
    records: TrajectoryDataset
    problem_spec: ProblemSpec


def _generation_contract(config: CylinderWake2DConfig) -> dict[str, Any]:
    payload = config.to_dict()
    payload.pop("dataset_path", None)
    return payload


def cylinder_dataset_contract_fingerprint(config: CylinderWake2DConfig) -> str:
    encoded = json.dumps(
        _generation_contract(config), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def cylinder_coordinates(
    config: CylinderWake2DConfig, *, device: torch.device
) -> tuple[Tensor, Tensor]:
    x = config.x_min + (torch.arange(config.nx, device=device) + 0.5) * config.dx
    y = config.y_min + (torch.arange(config.ny, device=device) + 0.5) * config.dy
    return torch.meshgrid(x, y, indexing="ij")


def cylinder_solid_mask(
    config: CylinderWake2DConfig, *, device: torch.device | str = "cpu"
) -> Tensor:
    selected = torch.device(device)
    x, y = cylinder_coordinates(config, device=selected)
    radius = 0.5 * config.cylinder_diameter
    return (x - config.cylinder_x).square() + (y - config.cylinder_y).square() <= radius**2


def make_cylinder_wake_problem_spec(config: CylinderWake2DConfig) -> ProblemSpec:
    return ProblemSpec(
        name="cylinder_wake_2d",
        channels=(
            ChannelSpec("u", "U_inf"),
            ChannelSpec("v", "U_inf"),
            ChannelSpec("pressure_coefficient", "1"),
        ),
        spatial_dim=2,
        grid=GridSpec(
            layout="channels_first",
            shape=(config.nx, config.ny),
            spacing=(config.dx, config.dy),
            coordinates_required=True,
            cell_weights_required=True,
            metadata={"endpoint": False, "solver": "D2Q9-BGK"},
        ),
        boundary=BoundarySpec(
            "fixed_cylinder_wake",
            {
                "inlet": (
                    "controlled_uniform_velocity"
                    if config.time_varying_boundary
                    else "constant_uniform_velocity"
                ),
                "outlet": "zero_gradient",
                "top_bottom": "constant_far_field",
                "cylinder": "halfway_bounce_back_no_slip",
                "time_varying": config.time_varying_boundary,
            },
        ),
        action_dim=0,
        parameter_dim=3,
        dt_mode=DtMode.CONSTANT,
        constant_dt=config.snapshot_dt,
        normalization=NormalizationSpec("standard", {"fit_scope": "train_only"}),
        geometry=GeometrySpec(
            mask_required=True,
            metadata={
                "cylinder_center": [config.cylinder_x, config.cylinder_y],
                "cylinder_diameter": config.cylinder_diameter,
            },
        ),
        observable_requirements=("u", "v", "pressure_coefficient"),
        metadata={
            "equations": "2D incompressible Navier-Stokes",
            "reynolds_number": config.reynolds_number,
            "u_infinity": config.u_infinity,
            "time_varying_boundary": config.time_varying_boundary,
            "pressure_reason": "required for auditable lift/drag traction diagnostics",
        },
    )


def _equilibrium(rho: Tensor, u: Tensor, v: Tensor) -> Tensor:
    c = _C.to(device=rho.device, dtype=rho.dtype)
    weights = _W.to(device=rho.device, dtype=rho.dtype).view(9, 1, 1)
    cu = c[:, 0, None, None] * u + c[:, 1, None, None] * v
    speed2 = u.square() + v.square()
    return weights * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu.square() - 1.5 * speed2)


def velocity_vorticity_divergence(
    state: Tensor, config: CylinderWake2DConfig
) -> tuple[Tensor, Tensor]:
    """Return second-order interior vorticity and divergence with replicated edges."""
    if state.shape[-3] < 2:
        raise ValueError("cylinder state requires u and v channels")
    u, v = state[..., 0, :, :], state[..., 1, :, :]
    du_dx = torch.zeros_like(u)
    du_dy = torch.zeros_like(u)
    dv_dx = torch.zeros_like(v)
    dv_dy = torch.zeros_like(v)
    du_dx[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) / (2 * config.dx)
    dv_dx[..., 1:-1, :] = (v[..., 2:, :] - v[..., :-2, :]) / (2 * config.dx)
    du_dy[..., :, 1:-1] = (u[..., :, 2:] - u[..., :, :-2]) / (2 * config.dy)
    dv_dy[..., :, 1:-1] = (v[..., :, 2:] - v[..., :, :-2]) / (2 * config.dy)
    return dv_dx - du_dy, du_dx + dv_dy


def shedding_frequency(signal: Tensor, dt: float) -> float:
    """Dominant nonzero FFT frequency; returns zero for an uninformative signal."""
    values = signal.detach().double().flatten()
    if values.numel() < 8 or float(values.std()) <= 1e-12:
        return 0.0
    spectrum = torch.fft.rfft(values - values.mean()).abs()
    frequencies = torch.fft.rfftfreq(values.numel(), d=dt)
    if spectrum.numel() <= 1:
        return 0.0
    return float(frequencies[1:][spectrum[1:].argmax()])


def shedding_spectral_peak(signal: Tensor, dt: float) -> dict[str, float]:
    """Return dominant frequency and robust peak-to-background spectral prominence."""
    values = signal.detach().double().flatten()
    if values.numel() < 8 or float(values.std()) <= 1e-12:
        return {"frequency": 0.0, "prominence": 0.0}
    spectrum = torch.fft.rfft(values - values.mean()).abs()[1:]
    frequencies = torch.fft.rfftfreq(values.numel(), d=dt)[1:]
    peak_index = int(spectrum.argmax())
    background = torch.cat((spectrum[:peak_index], spectrum[peak_index + 1 :]))
    baseline = background.median() if background.numel() else spectrum.new_tensor(0.0)
    prominence = float(spectrum[peak_index] / baseline.clamp_min(1e-12))
    return {"frequency": float(frequencies[peak_index]), "prominence": prominence}


def cylinder_force_coefficients(
    state: Tensor, config: CylinderWake2DConfig
) -> tuple[Tensor, Tensor]:
    """Discrete pressure/shear traction diagnostic on fluid cells adjoining the cylinder.

    This is a consistent reduced-grid diagnostic for comparing true and decoded
    trajectories, not a replacement for a high-order CFD surface-force integral.
    """
    if state.shape[-3:] != (3, config.nx, config.ny):
        raise ValueError("force diagnostic requires [...,3,Nx,Ny] cylinder state")
    solid = cylinder_solid_mask(config, device=state.device)
    adjacent = torch.zeros_like(solid)
    for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        adjacent |= torch.roll(solid, shifts=shift, dims=(0, 1))
    boundary = adjacent & ~solid
    x, y = cylinder_coordinates(config, device=state.device)
    radius = torch.sqrt((x - config.cylinder_x).square() + (y - config.cylinder_y).square())
    nx = (x - config.cylinder_x) / radius.clamp_min(1e-12)
    ny = (y - config.cylinder_y) / radius.clamp_min(1e-12)
    tx, ty = -ny, nx
    pressure = state[..., 2, :, :]
    tangential_velocity = state[..., 0, :, :] * tx + state[..., 1, :, :] * ty
    shear = (2.0 / config.reynolds_number) * tangential_velocity / min(config.dx, config.dy)
    traction_x = -pressure * nx + shear * tx
    traction_y = -pressure * ny + shear * ty
    weight = torch.pi * config.cylinder_diameter / max(int(boundary.sum()), 1)
    drag = traction_x[..., boundary].sum(dim=-1) * weight / config.cylinder_diameter
    lift = traction_y[..., boundary].sum(dim=-1) * weight / config.cylinder_diameter
    return drag, lift


class D2Q9CylinderWakeSolver:
    """Small deterministic BGK solver with controlled inlet and cylinder bounce-back."""

    def __init__(
        self,
        config: CylinderWake2DConfig,
        *,
        seed: int,
        device: torch.device | str = "cpu",
        initial_lattice_inflow_velocity: float | None = None,
    ) -> None:
        if seed < 0:
            raise ValueError("flow seed must be non-negative")
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.solid = cylinder_solid_mask(config, device=self.device)
        self.fluid = ~self.solid
        x, y = cylinder_coordinates(config, device=self.device)
        phase = 2 * torch.pi * torch.rand((), generator=torch.Generator().manual_seed(seed))
        phase = phase.to(self.device)
        wake = torch.exp(-((x - config.cylinder_x - 2.0) / 3.0).square())
        downstream = (x > config.cylinder_x).to(self.dtype)
        initial_inlet = (
            config.lattice_inflow_velocity
            if initial_lattice_inflow_velocity is None
            else float(initial_lattice_inflow_velocity)
        )
        if not 0 < initial_inlet < 0.12:
            raise ValueError("D2Q9 controlled inlet must remain in the low-Mach interval (0,0.12)")
        self.current_lattice_inflow_velocity = initial_inlet
        u = torch.full_like(x, initial_inlet)
        v = (
            initial_inlet
            * config.perturbation_amplitude
            * torch.sin(2 * torch.pi * (y - config.y_min) / (config.y_max - config.y_min) + phase)
            * wake
            * downstream
        )
        u[self.solid] = 0.0
        v[self.solid] = 0.0
        self.f = _equilibrium(torch.ones_like(x), u, v)
        self.last_force = torch.zeros(2, device=self.device, dtype=self.dtype)

    def macroscopic(self) -> tuple[Tensor, Tensor, Tensor]:
        rho = self.f.sum(dim=0).clamp_min(1e-8)
        c = _C.to(device=self.device, dtype=self.dtype)
        u = (self.f * c[:, 0, None, None]).sum(dim=0) / rho
        v = (self.f * c[:, 1, None, None]).sum(dim=0) / rho
        u[self.solid] = 0.0
        v[self.solid] = 0.0
        return rho, u, v

    def step(self, inlet_velocity: float | None = None) -> None:
        rho, u, v = self.macroscopic()
        inlet = (
            self.current_lattice_inflow_velocity
            if inlet_velocity is None
            else float(inlet_velocity)
        )
        if not 0 < inlet < 0.12:
            raise ValueError("D2Q9 controlled inlet must remain in the low-Mach interval (0,0.12)")
        self.current_lattice_inflow_velocity = inlet
        u[0, :] = inlet
        v[0, :] = 0.0
        u[:, 0] = inlet
        u[:, -1] = inlet
        v[:, 0] = 0.0
        v[:, -1] = 0.0
        equilibrium = _equilibrium(rho, u, v)
        post = self.f - (self.f - equilibrium) / self.config.lattice_relaxation_time
        streamed = torch.empty_like(post)
        force = torch.zeros(2, device=self.device, dtype=self.dtype)
        c = _C.to(device=self.device)
        for index in range(9):
            shift = (int(c[index, 0]), int(c[index, 1]))
            pulled = torch.roll(post[index], shifts=shift, dims=(0, 1))
            source_solid = torch.roll(self.solid, shifts=shift, dims=(0, 1))
            reflected = post[_OPPOSITE[index]]
            streamed[index] = torch.where(source_solid, reflected, pulled)
            interface = source_solid & self.fluid
            force += 2.0 * reflected[interface].sum() * c[index].to(self.dtype)
        boundary_rho = torch.ones_like(rho)
        far = _equilibrium(
            boundary_rho,
            torch.full_like(u, inlet),
            torch.zeros_like(v),
        )
        streamed[:, 0, :] = far[:, 0, :]
        streamed[:, :, 0] = far[:, :, 0]
        streamed[:, :, -1] = far[:, :, -1]
        streamed[:, -1, :] = streamed[:, -2, :]
        streamed[:, self.solid] = equilibrium[:, self.solid]
        self.f = streamed
        self.last_force = force

    def state(self) -> Tensor:
        rho, u, v = self.macroscopic()
        scale = self.config.lattice_inflow_velocity
        pressure = (rho - rho[self.fluid].mean()) / (1.5 * scale * scale)
        state = torch.stack((u / scale, v / scale, pressure))
        state[:, self.solid] = 0.0
        return state

    def diagnostics(self) -> dict[str, float]:
        state = self.state()
        vorticity, divergence = velocity_vorticity_divergence(state, self.config)
        fluid = self.fluid
        dynamic = self.current_lattice_inflow_velocity**2 * self.config.cylinder_diameter_cells
        coefficients = 2.0 * self.last_force / max(dynamic, 1e-12)
        return {
            "lift_coefficient": float(coefficients[1]),
            "drag_coefficient": float(coefficients[0]),
            "kinetic_energy": float(0.5 * state[:2, fluid].square().sum(dim=0).mean()),
            "divergence_rms": float(divergence[fluid].square().mean().sqrt()),
            "vorticity_rms": float(vorticity[fluid].square().mean().sqrt()),
            "max_abs_state": float(state.abs().max()),
        }


def generate_cylinder_wake_2d_trajectories(
    config: CylinderWake2DConfig,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> CylinderWakeDataset:
    """Generate multiple fixed-condition transients with independent admissible perturbations."""
    selected = torch.device(device)
    xx, yy = cylinder_coordinates(config, device=selected)
    coordinates = torch.stack((xx, yy)).cpu()
    cell_weights = torch.full((config.nx, config.ny), config.dx * config.dy)
    valid_mask = (~cylinder_solid_mask(config)).to(torch.bool)
    records: list[TrajectoryRecord] = []
    for index in range(config.num_trajectories):
        solver = D2Q9CylinderWakeSolver(config, seed=seed * 1009 + index, device=selected)
        states = [solver.state().cpu()]
        diagnostics = [solver.diagnostics()]
        for _ in range(config.num_steps):
            for _ in range(config.solver_steps_per_snapshot):
                solver.step()
            state = solver.state()
            if not torch.isfinite(state).all():
                raise RuntimeError("cylinder solver produced non-finite state")
            states.append(state.cpu())
            diagnostics.append(solver.diagnostics())
        lift = torch.tensor([item["lift_coefficient"] for item in diagnostics])
        records.append(
            TrajectoryRecord(
                trajectory_id=f"cylinder-wake-2d-{seed:04d}-{index:04d}",
                states_raw=torch.stack(states),
                dts=torch.full((config.num_steps,), config.snapshot_dt),
                mu_static=torch.tensor(
                    [config.reynolds_number, config.u_infinity, config.cylinder_diameter]
                ),
                coordinates=coordinates,
                cell_weights=cell_weights,
                valid_mask=valid_mask,
                metadata={
                    "flow_data_seed": seed,
                    "trajectory_perturbation_index": index,
                    "diagnostics": diagnostics,
                    "dominant_lift_frequency": shedding_frequency(lift, config.snapshot_dt),
                    "strouhal_number": shedding_frequency(lift, config.snapshot_dt),
                    "solver": "D2Q9-BGK-low-Mach",
                },
            )
        )
    return CylinderWakeDataset(TrajectoryDataset(records), make_cylinder_wake_problem_spec(config))


def cylinder_condition_schedule(
    config: CylinderWake2DConfig,
    condition: V09ConditionConfig,
    schedule_type: str,
) -> dict[str, Any]:
    """Return the causal per-transition Re/U schedule and its audited transition metadata."""
    if schedule_type not in condition.schedule_types:
        raise ValueError(f"unregistered V0.9 schedule type: {schedule_type!r}")
    steps = config.num_steps
    transition_index = max(1, min(steps - 2, round(condition.transition_start_fraction * steps)))
    ramp_steps = max(1, round(condition.smooth_duration_fraction * steps))
    indices = torch.arange(steps, dtype=torch.float64)
    if "slow" in schedule_type:
        ramp_steps = max(1, round(1.35 * ramp_steps))
    elif "fast" in schedule_type:
        ramp_steps = max(1, round(0.55 * ramp_steps))
    ramp_steps = min(ramp_steps, max(1, steps - transition_index - 2))
    family = "cyclic" if schedule_type.startswith("cyclic_") else (
        "down" if "_down" in schedule_type else "up"
    )
    dwell_steps = 0
    if schedule_type in {"abrupt", "abrupt_up", "abrupt_down"}:
        blend = (indices >= transition_index).to(torch.float64)
        ramp_steps = 0
    elif family == "cyclic":
        dwell_fraction = 0.18 if schedule_type.endswith("_long") else 0.07
        dwell_steps = max(2, round(dwell_fraction * steps))
        ramp_steps = min(
            ramp_steps,
            max(1, (steps - transition_index - dwell_steps - 2) // 2),
        )
        up_tau = ((indices - transition_index) / ramp_steps).clamp(0.0, 1.0)
        down_start = transition_index + ramp_steps + dwell_steps
        down_tau = ((indices - down_start) / ramp_steps).clamp(0.0, 1.0)
        blend = 0.5 * (1.0 - torch.cos(torch.pi * up_tau))
        blend = blend * (1.0 - 0.5 * (1.0 - torch.cos(torch.pi * down_tau)))
    else:
        tau = ((indices - transition_index) / ramp_steps).clamp(0.0, 1.0)
        blend = 0.5 * (1.0 - torch.cos(torch.pi * tau))
    if family == "down":
        blend = 1.0 - blend
    reynolds = condition.reynolds_low + (condition.reynolds_high - condition.reynolds_low) * blend
    lattice_inlet = config.lattice_viscosity * reynolds / config.cylinder_diameter_cells
    physical_inlet = config.u_infinity * reynolds / config.reynolds_number
    if float(lattice_inlet.max()) >= 0.12:
        raise ValueError("V0.9 condition range violates the D2Q9 low-Mach inlet limit")
    return {
        "schedule_type": schedule_type,
        "transition_index": transition_index,
        "ramp_steps": ramp_steps,
        "schedule_family": family,
        "rate_class": (
            "abrupt"
            if "abrupt" in schedule_type
            else "slow"
            if "slow" in schedule_type
            else "fast"
            if "fast" in schedule_type
            else "medium"
        ),
        "dwell_steps": dwell_steps,
        "reynolds_number": reynolds.float(),
        "lattice_inflow_velocity": lattice_inlet.float(),
        "u_infinity": physical_inlet.float(),
    }


def generate_v0_9_cylinder_wake_trajectories(
    config: CylinderWake2DConfig,
    condition: V09ConditionConfig,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> CylinderWakeDataset:
    """Generate smooth/abrupt in-distribution trajectories with fixed viscosity and geometry."""
    if not config.time_varying_boundary:
        raise ValueError("V0.9 trajectory generation requires time_varying_boundary=true")
    selected = torch.device(device)
    xx, yy = cylinder_coordinates(config, device=selected)
    coordinates = torch.stack((xx, yy)).cpu()
    cell_weights = torch.full((config.nx, config.ny), config.dx * config.dy)
    valid_mask = (~cylinder_solid_mask(config)).to(torch.bool)
    records: list[TrajectoryRecord] = []
    for index in range(config.num_trajectories):
        schedule_type = condition.schedule_types[index % len(condition.schedule_types)]
        schedule = cylinder_condition_schedule(config, condition, schedule_type)
        lattice = torch.as_tensor(schedule["lattice_inflow_velocity"])
        solver = D2Q9CylinderWakeSolver(
            config,
            seed=seed * 1009 + index,
            device=selected,
            initial_lattice_inflow_velocity=float(lattice[0]),
        )
        states = [solver.state().cpu()]
        diagnostics = [solver.diagnostics()]
        for step_index in range(config.num_steps):
            inlet = float(lattice[step_index])
            for _ in range(config.solver_steps_per_snapshot):
                solver.step(inlet)
            state = solver.state()
            if not torch.isfinite(state).all():
                raise RuntimeError("V0.9 cylinder solver produced non-finite state")
            states.append(state.cpu())
            diagnostics.append(solver.diagnostics())
        lift = torch.tensor([item["lift_coefficient"] for item in diagnostics])
        condition_series = torch.stack(
            (
                torch.as_tensor(schedule["reynolds_number"]),
                torch.as_tensor(schedule["u_infinity"]),
            ),
            dim=-1,
        )
        records.append(
            TrajectoryRecord(
                trajectory_id=(
                    f"cylinder-wake-v09-{schedule_type}-{seed:04d}-{index:04d}"
                ),
                states_raw=torch.stack(states),
                dts=torch.full((config.num_steps,), config.snapshot_dt),
                mu_static=torch.tensor(
                    [config.reynolds_number, config.u_infinity, config.cylinder_diameter]
                ),
                coordinates=coordinates,
                cell_weights=cell_weights,
                valid_mask=valid_mask,
                metadata={
                    "flow_data_seed": seed,
                    "trajectory_perturbation_index": index,
                    "schedule_type": schedule_type,
                    "transition_index": int(schedule["transition_index"]),
                    "ramp_steps": int(schedule["ramp_steps"]),
                    "schedule_family": str(schedule["schedule_family"]),
                    "rate_class": str(schedule["rate_class"]),
                    "dwell_steps": int(schedule["dwell_steps"]),
                    "condition_series": condition_series.tolist(),
                    "diagnostics": diagnostics,
                    "dominant_lift_frequency": shedding_frequency(lift, config.snapshot_dt),
                    "solver": "D2Q9-BGK-low-Mach-fixed-viscosity-controlled-inlet",
                },
            )
        )
    result = CylinderWakeDataset(
        TrajectoryDataset(records), make_cylinder_wake_problem_spec(config)
    )
    validate_trajectories_against_spec(result.records, result.problem_spec)
    return result


def validate_v0_9_cylinder_wake_dataset(
    dataset: CylinderWakeDataset,
    config: CylinderWake2DConfig,
    condition: V09ConditionConfig,
) -> dict[str, Any]:
    """Pre-ML gate for controlled schedules without presuming a shedding peak per short phase."""
    base = validate_cylinder_wake_dataset(dataset, config, require_shedding=False)
    types = {str(record.metadata.get("schedule_type")) for record in dataset.records}
    schedule_complete = types == set(condition.schedule_types)
    aligned = True
    ranges_valid = True
    expected_extended = len(condition.schedule_types) > 2
    observed_families: set[str] = set()
    observed_rate_classes: set[str] = set()
    cyclic_dwell_steps: set[int] = set()
    positive_rate = False
    negative_rate = False
    cyclic_return = False
    for record in dataset.records:
        series = torch.as_tensor(record.metadata.get("condition_series"), dtype=torch.float64)
        transition = int(record.metadata.get("transition_index", -1))
        family = str(record.metadata.get("schedule_family", ""))
        rate_class = str(record.metadata.get("rate_class", ""))
        if family:
            observed_families.add(family)
        if rate_class:
            observed_rate_classes.add(rate_class)
        if family == "cyclic":
            cyclic_dwell_steps.add(int(record.metadata.get("dwell_steps", 0)))
        aligned = (
            aligned
            and series.shape == (record.num_steps, 2)
            and 0 < transition < record.num_steps
        )
        if series.shape == (record.num_steps, 2):
            increments = torch.diff(series[:, 0])
            positive_rate = positive_rate or bool(torch.any(increments > 1.0e-6))
            negative_rate = negative_rate or bool(torch.any(increments < -1.0e-6))
            if family == "cyclic":
                cyclic_return = cyclic_return or bool(
                    torch.max(series[:, 0]) - series[0, 0] > 1.0
                    and torch.abs(series[-1, 0] - series[0, 0]) <= 1.0e-5
                )
            ranges_valid = ranges_valid and bool(
                torch.all(series[:, 0] >= condition.reynolds_low - 1e-6)
                and torch.all(series[:, 0] <= condition.reynolds_high + 1e-6)
            )
    gates = {
        **base["gates"],
        "schedule_types_complete": schedule_complete,
        "condition_alignment": aligned,
        "condition_range": ranges_valid,
        "identifiability_schedule_families": (
            not expected_extended or observed_families == {"up", "down", "cyclic"}
        ),
        "bidirectional_condition_rate": (
            not expected_extended or (positive_rate and negative_rate)
        ),
        "cyclic_return": not expected_extended or cyclic_return,
        "rate_variants_complete": not expected_extended
        or observed_rate_classes == {"slow", "medium", "fast", "abrupt"},
        "dwell_variants_complete": not expected_extended
        or len({value for value in cyclic_dwell_steps if value > 0}) >= 2,
    }
    required = (
        "finite_fields",
        "bounded_solution",
        "reasonable_divergence",
        "nontrivial_transient",
        "schedule_types_complete",
        "condition_alignment",
        "condition_range",
        "identifiability_schedule_families",
        "bidirectional_condition_rate",
        "cyclic_return",
        "rate_variants_complete",
        "dwell_variants_complete",
    )
    return {
        **base,
        "status": "PASS" if all(gates[name] for name in required) else "FAIL",
        "gates": gates,
        "condition_contract": condition.to_dict(),
    }


def save_cylinder_wake_dataset(
    dataset: CylinderWakeDataset,
    config: CylinderWake2DConfig,
    path: str | Path,
) -> None:
    """Atomically persist one offline dataset for reuse by every learning stage."""
    validate_trajectories_against_spec(dataset.records, dataset.problem_spec)
    payload = {
        "schema_version": 1,
        "problem_name": "cylinder_wake_2d",
        "generation_contract": _generation_contract(config),
        "generation_contract_fingerprint": cylinder_dataset_contract_fingerprint(config),
        "records": [
            {
                "trajectory_id": record.trajectory_id,
                "states_raw": record.states_raw,
                "dts": record.dts,
                "actions": record.actions,
                "mu_static": record.mu_static,
                "coordinates": record.coordinates,
                "cell_weights": record.cell_weights,
                "valid_mask": record.valid_mask,
                "metadata": dict(record.metadata),
            }
            for record in dataset.records
        ],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_cylinder_wake_dataset(
    path: str | Path,
    config: CylinderWake2DConfig,
) -> CylinderWakeDataset:
    """Load only data produced under the exact current physical/numerical contract."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("invalid cylinder-wake dataset payload")
    expected = cylinder_dataset_contract_fingerprint(config)
    if payload.get("generation_contract_fingerprint") != expected:
        raise ValueError("cylinder-wake dataset/config contract mismatch")
    records = TrajectoryDataset([TrajectoryRecord(**item) for item in payload["records"]])
    result = CylinderWakeDataset(records, make_cylinder_wake_problem_spec(config))
    validate_trajectories_against_spec(result.records, result.problem_spec)
    return result


def validate_cylinder_wake_dataset(
    dataset: CylinderWakeDataset,
    config: CylinderWake2DConfig,
    *,
    require_shedding: bool = True,
) -> dict[str, Any]:
    """Run the pre-ML numerical, transient, and shedding acceptance gate."""
    validate_trajectories_against_spec(dataset.records, dataset.problem_spec)
    divergence_values: list[float] = []
    lift_values: list[Tensor] = []
    transient_scores: list[float] = []
    max_abs_state = 0.0
    for record in dataset.records:
        states = record.states_raw
        max_abs_state = max(max_abs_state, float(states.abs().max()))
        vorticity, divergence = velocity_vorticity_divergence(states, config)
        assert record.valid_mask is not None
        fluid = record.valid_mask
        divergence_values.append(float(divergence[..., fluid].square().mean().sqrt()))
        early = vorticity[: max(2, len(vorticity) // 4), ..., fluid].square().mean().sqrt()
        late = vorticity[-max(2, len(vorticity) // 4) :, ..., fluid].square().mean().sqrt()
        transient_scores.append(float((late - early).abs() / early.clamp_min(1e-8)))
        diagnostics = record.metadata.get("diagnostics", [])
        lift_values.append(torch.tensor([float(item["lift_coefficient"]) for item in diagnostics]))
    lift_std = min(float(values.std()) for values in lift_values)
    spectral = [shedding_spectral_peak(values, config.snapshot_dt) for values in lift_values]
    frequencies = [item["frequency"] for item in spectral]
    prominences = [item["prominence"] for item in spectral]
    gates = {
        "finite_fields": all(
            bool(torch.isfinite(record.states_raw).all()) for record in dataset.records
        ),
        "bounded_solution": max_abs_state < 50.0,
        "reasonable_divergence": max(divergence_values) < 0.25,
        "nontrivial_lift": lift_std > 1e-5,
        "identifiable_spectral_peak": min(prominences) >= 3.0,
        "nontrivial_transient": min(transient_scores) > 1e-3,
    }
    required = [
        "finite_fields",
        "bounded_solution",
        "reasonable_divergence",
        "nontrivial_transient",
    ]
    if require_shedding:
        required.extend(("nontrivial_lift", "identifiable_spectral_peak"))
    return {
        "status": "PASS" if all(gates[name] for name in required) else "FAIL",
        "require_shedding": require_shedding,
        "gates": gates,
        "metrics": {
            "max_abs_state": max_abs_state,
            "max_divergence_rms": max(divergence_values),
            "minimum_lift_std": lift_std,
            "dominant_lift_frequencies": frequencies,
            "minimum_spectral_peak_prominence": min(prominences),
            "minimum_transient_relative_change": min(transient_scores),
        },
        "dataset_contract_fingerprint": cylinder_dataset_contract_fingerprint(config),
    }
