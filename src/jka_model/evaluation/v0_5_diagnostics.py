"""Bounded V0.5 diagnostics shared by CPU tests and the GPU validation package."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from jka_model.config import FieldLossConfig, ProjectConfig, load_config
from jka_model.contracts import ProblemBatch, ProblemSpec
from jka_model.data import (
    ChannelStandardizer,
    TrajectoryWindowDataset,
    collate_problem_batches,
    make_split_manifest,
    select_split,
)
from jka_model.losses import compute_field_koopman_loss
from jka_model.models import FieldKoopmanAutoencoder
from jka_model.physics import PhysicsConstraint
from jka_model.problems import create_problem_adapter
from train.train_v0_5 import initialize_v0_5_model


@dataclass(slots=True)
class V05DiagnosticCase:
    model: FieldKoopmanAutoencoder
    batch: ProblemBatch
    normalizer: ChannelStandardizer
    spec: ProblemSpec
    loss_config: FieldLossConfig
    constraints: dict[str, PhysicsConstraint]
    precision: str


def prepare_v0_5_diagnostic_case(
    config: ProjectConfig | str | Path,
    *,
    device: str | torch.device,
    state_dict: dict[str, Any] | None = None,
) -> V05DiagnosticCase:
    """Build one deterministic two-sample training batch without creating a run."""
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    if resolved.field_loss is None or resolved.v0_5_training is None:
        raise ValueError("V0.5 diagnostics require loss and training sections")
    adapter = create_problem_adapter(resolved)
    records = adapter.build_dataset(seed=resolved.training.seed)
    spec = adapter.build_problem_spec()
    manifest = make_split_manifest(records, resolved.data.split)
    normalizer = ChannelStandardizer(eps=resolved.data.normalization.eps).fit(
        records, manifest, spec
    )
    windows = TrajectoryWindowDataset(
        select_split(records, manifest, "train"),
        history=resolved.data.history,
        horizon=resolved.data.horizon,
        normalizer=normalizer,
    )
    sample_count = min(2, len(windows))
    if sample_count < 1:
        raise ValueError("V0.5 diagnostics require at least one training window")
    batch = collate_problem_batches([windows[index] for index in range(sample_count)]).to(
        device=device, dtype=torch.float32
    )
    model = initialize_v0_5_model(resolved, device=device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return V05DiagnosticCase(
        model,
        batch,
        normalizer,
        spec,
        resolved.field_loss,
        dict(adapter.build_physics_constraints()),
        resolved.v0_5_training.precision,
    )


def _module_gradient_norm(module: torch.nn.Module) -> float:
    values = [
        parameter.grad.norm() for parameter in module.parameters() if parameter.grad is not None
    ]
    return 0.0 if not values else float(torch.stack(values).norm())


def run_v0_5_diagnostic_step(
    case: V05DiagnosticCase,
    *,
    backward_physics: bool,
) -> dict[str, Any]:
    """Capture component outputs and optionally backpropagate isolated raw physics."""
    model, batch = case.model, case.batch
    model.zero_grad(set_to_none=True)
    device_type = batch.future_dts.device.type
    amp_enabled = device_type == "cuda" and case.precision != "fp32"
    amp_dtype = torch.float16 if case.precision == "amp_fp16" else torch.bfloat16
    context = torch.autocast("cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
    with context:
        encoded = model.encode(batch.context_states_model[:, -1])
        stepped = model.core.step(encoded, batch.future_dts[:, 0])
        decoded = model.decode(stepped)
        losses = compute_field_koopman_loss(
            model,
            batch,
            case.normalizer,
            case.spec,
            case.loss_config,
            case.constraints,
            physics_scale=1.0,
        )
        physics = (
            case.loss_config.lambda_mass * losses.mass
            + case.loss_config.lambda_operator * losses.operator
        )
    if backward_physics:
        physics.backward()
    generator_norm = 0.0 if model.core.A.grad is None else float(model.core.A.grad.norm())
    return {
        "encoder_output": encoded.detach().float().cpu(),
        "koopman_step_output": stepped.detach().float().cpu(),
        "decoder_output": decoded.detach().float().cpu(),
        "mass_penalty": float(losses.mass.detach()),
        "operator_penalty": float(losses.operator.detach()),
        "physics_loss": float(physics.detach()),
        "gradient_norms": {
            "encoder": _module_gradient_norm(model.encoder),
            "decoder": _module_gradient_norm(model.decoder),
            "generator": generator_norm,
        },
    }
