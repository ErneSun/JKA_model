"""Canonical held-out V0.5 field rollout evaluation."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from jka_model.config import ProjectConfig, load_config
from jka_model.data import (
    ChannelStandardizer,
    SplitManifest,
    select_split,
)
from jka_model.physics import weighted_integral_2d
from jka_model.problems import create_problem_adapter
from jka_model.utils import load_checkpoint
from train.train_v0_5 import initialize_v0_5_model


def _field_metrics(prediction: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    error = prediction - truth
    return {
        "rmse": float(error.square().mean().sqrt()),
        "relative_l2": float(error.norm() / truth.norm().clamp_min(1e-12)),
    }


def evaluate_v0_5(
    config: ProjectConfig | str | Path,
    *,
    checkpoint: str | Path,
    device: str | torch.device | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate fixed short/medium/long held-out rollouts without fitting anything."""
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    if resolved.v0_5_evaluation is None:
        raise ValueError("evaluate_v0_5 requires complete V0.5 data/evaluation sections")
    selected = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
    )
    saved = load_checkpoint(checkpoint, map_location=selected)
    if saved.config_hash != resolved.stable_hash or saved.online_model_state is None:
        raise ValueError("evaluation checkpoint/config mismatch or missing model state")
    model = initialize_v0_5_model(resolved, device=selected)
    model.load_state_dict(saved.online_model_state)
    model.eval()
    normalizer = ChannelStandardizer(eps=resolved.data.normalization.eps)
    if saved.normalizer_state is None:
        raise ValueError("evaluation checkpoint lacks normalizer state")
    normalizer.load_state_dict(saved.normalizer_state)
    adapter = create_problem_adapter(resolved)
    records = adapter.build_dataset(seed=resolved.training.seed)
    spec = adapter.build_problem_spec()
    reference = adapter.compute_reference_metrics()
    if not isinstance(saved.split_manifest, dict):
        raise ValueError("evaluation checkpoint lacks split manifest")
    manifest = SplitManifest.from_dict(saved.split_manifest)
    test_records = select_split(records, manifest, "test")
    if not test_records:
        raise ValueError("V0.5 test split must be non-empty")
    horizons = {
        "short": resolved.v0_5_evaluation.short_horizon,
        "medium": resolved.v0_5_evaluation.medium_horizon,
        "long": resolved.v0_5_evaluation.long_horizon,
    }
    aggregates: dict[str, dict[str, list[float]]] = {
        name: {
            "rmse": [],
            "relative_l2": [],
            "persistence_rmse": [],
            "mass_drift": [],
            "operator": [],
        }
        for name in horizons
    }
    constraints = adapter.build_physics_constraints()
    try:
        operator = constraints["operator"]
    except KeyError as error:
        raise ValueError("problem adapter must provide an 'operator' constraint") from error
    reconstruction_errors: list[float] = []
    latent_samples: list[torch.Tensor] = []
    with torch.no_grad():
        for record in test_records:
            if record.cell_weights is None or record.mu_static is None:
                raise ValueError("V0.5 evaluation records require cell weights and parameters")
            initial_raw = record.states_raw[0:1].to(selected, torch.float32)
            initial_model = normalizer.transform(initial_raw)
            full_model = normalizer.transform(record.states_raw.to(selected, torch.float32))
            latent_samples.append(model.encode(full_model).detach().cpu())
            reconstructed_raw = normalizer.inverse_transform(
                model.decode(model.encode(initial_model))
            )
            reconstruction_errors.append(
                float((reconstructed_raw - initial_raw).square().mean().sqrt())
            )
            for name, horizon in horizons.items():
                dts = record.dts[:horizon].to(selected, torch.float32).unsqueeze(0)
                predicted_model, _ = model.rollout(initial_model, dts)
                predicted_raw = normalizer.inverse_transform(predicted_model)[0]
                truth = record.states_raw[1 : horizon + 1].to(selected, torch.float32)
                values = _field_metrics(predicted_raw, truth)
                aggregates[name]["rmse"].append(values["rmse"])
                aggregates[name]["relative_l2"].append(values["relative_l2"])
                persistence = initial_raw.expand_as(truth)
                aggregates[name]["persistence_rmse"].append(
                    float((persistence - truth).square().mean().sqrt())
                )
                weights_raw = record.cell_weights.to(selected, torch.float32)
                weights = weights_raw.unsqueeze(0)
                initial_mass = weighted_integral_2d(initial_raw, weights)
                masses = weighted_integral_2d(predicted_raw, weights_raw)
                aggregates[name]["mass_drift"].append(float((masses - initial_mass).abs().max()))
                previous = initial_raw
                terms: list[torch.Tensor] = []
                metadata = {
                    "mu_static": record.mu_static.to(selected, torch.float32).unsqueeze(0),
                    "cell_weights": weights,
                }
                for index in range(horizon):
                    current = predicted_raw[index : index + 1]
                    operator_result = operator.loss(
                        current,
                        prev_state_raw=previous,
                        dt=dts[:, index],
                        spec=spec,
                        metadata=metadata,
                    )
                    if len(operator_result) != 1:
                        raise ValueError("operator constraint must return one scalar penalty")
                    terms.append(next(iter(operator_result.values())))
                    previous = current
                aggregates[name]["operator"].append(float(torch.stack(terms).mean()))
    rollout = {
        name: {metric: sum(values) / len(values) for metric, values in metrics.items()}
        for name, metrics in aggregates.items()
    }
    eigenvalues = torch.linalg.eigvals(model.core.A.detach().cpu())
    true_frequency = abs(reference["angular_frequency"])
    candidates = eigenvalues.imag.abs()
    selected_index = int((candidates - true_frequency).abs().argmin())
    selected_eigenvalue = eigenvalues[selected_index]
    learned_frequency = float(candidates[selected_index])
    learned_decay = float(-selected_eigenvalue.real)
    true_decay = float(reference["decay_rate"])
    frequency_error = abs(learned_frequency - true_frequency) / max(true_frequency, 1e-12)
    decay_error = abs(learned_decay - true_decay) / max(true_decay, 1e-12)
    latent = torch.cat(latent_samples, dim=0)
    latent_std = latent.std(dim=0, unbiased=False)
    result: dict[str, Any] = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "device": str(selected),
        "split": "test",
        "trajectory_ids": [record.trajectory_id for record in test_records],
        "reconstruction_rmse": sum(reconstruction_errors) / len(reconstruction_errors),
        "rollout": rollout,
        "learned_angular_frequency": learned_frequency,
        "true_angular_frequency": true_frequency,
        "learned_frequency_hz": learned_frequency / (2.0 * math.pi),
        "true_frequency_hz": true_frequency / (2.0 * math.pi),
        "frequency_relative_error": frequency_error,
        "learned_decay_rate": learned_decay,
        "true_decay_rate": true_decay,
        "decay_relative_error": decay_error,
        "selected_eigenvalue": [float(selected_eigenvalue.real), float(selected_eigenvalue.imag)],
        "latent_min_std": float(latent_std.min()),
        "latent_max_std": float(latent_std.max()),
        "frequency_threshold": resolved.v0_5_evaluation.max_frequency_relative_error,
        "scientific_acceptance": "PENDING_GPU"
        if selected.type != "cuda"
        else "MEASURED_NOT_AUTOMATICALLY_ACCEPTED",
        "finite": all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
    }
    if run_dir is not None:
        run_path = Path(run_dir)
        destination = run_path / "evaluation"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        (destination / "final_metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        with (destination / "rollout_by_horizon.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "horizon",
                    "steps",
                    "rmse",
                    "relative_l2",
                    "persistence_rmse",
                    "mass_drift",
                    "operator",
                ],
            )
            writer.writeheader()
            for name, values in rollout.items():
                writer.writerow({"horizon": name, "steps": horizons[name], **values})
        (destination / "spectrum.json").write_text(
            json.dumps(
                {
                    "eigenvalues": [
                        [float(value.real), float(value.imag)] for value in eigenvalues
                    ],
                    "learned_angular_frequency": learned_frequency,
                    "true_angular_frequency": true_frequency,
                    "learned_frequency_hz": learned_frequency / (2.0 * math.pi),
                    "true_frequency_hz": true_frequency / (2.0 * math.pi),
                    "relative_error": frequency_error,
                    "learned_decay_rate": learned_decay,
                    "true_decay_rate": true_decay,
                    "decay_relative_error": decay_error,
                    "selected_eigenvalue": [
                        float(selected_eigenvalue.real),
                        float(selected_eigenvalue.imag),
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (destination / "physics_metrics.json").write_text(
            json.dumps(
                {
                    name: {"mass_drift": values["mass_drift"], "operator": values["operator"]}
                    for name, values in rollout.items()
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (destination / "baseline_metrics.json").write_text(
            json.dumps(
                {
                    name: {
                        "persistence_rmse": values["persistence_rmse"],
                        "model_rmse": values["rmse"],
                    }
                    for name, values in rollout.items()
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        reports = run_path / "reports"
        reports.mkdir(exist_ok=True)
        report = (
            "# V0.5 held-out test record\n\n"
            f"- device: `{selected}`\n- checkpoint: `{checkpoint}`\n"
            f"- learned/true angular frequency: {learned_frequency:.6g} / {true_frequency:.6g}\n"
            f"- relative frequency error: {frequency_error:.6g}\n"
            f"- learned/true decay rate: {learned_decay:.6g} / {true_decay:.6g}\n"
            f"- latent std range: {float(latent_std.min()):.6g} .. {float(latent_std.max()):.6g}\n"
            f"- long rollout RMSE: {rollout['long']['rmse']:.6g}\n"
            f"- long persistence RMSE: {rollout['long']['persistence_rmse']:.6g}\n"
            f"- scientific acceptance: {result['scientific_acceptance']}\n"
        )
        (reports / "test_record.md").write_text(report)
        (reports / "final_report.md").write_text(report)
        os.environ.setdefault("MPLCONFIGDIR", str((run_path / "profiler" / "matplotlib").resolve()))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = [horizons[name] for name in ("short", "medium", "long")]
        figures = {
            "rollout_error.png": (
                [rollout[name]["rmse"] for name in ("short", "medium", "long")],
                "RMSE",
            ),
            "physics_metrics.png": (
                [rollout[name]["operator"] for name in ("short", "medium", "long")],
                "operator penalty",
            ),
        }
        for filename, (plot_values, ylabel) in figures.items():
            figure, axis = plt.subplots(figsize=(4.5, 3.0))
            axis.plot(x, plot_values, marker="o")
            axis.set(xlabel="rollout steps", ylabel=ylabel)
            figure.tight_layout()
            figure.savefig(run_path / "plots" / filename, dpi=120)
            plt.close(figure)
        history_path = run_path / "logs" / "epoch_metrics.csv"
        if history_path.is_file():
            with history_path.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            figure, axis = plt.subplots(figsize=(4.5, 3.0))
            axis.plot(
                [int(row["epoch"]) for row in rows],
                [float(row["L_total"]) for row in rows],
                marker="o",
            )
            axis.set(xlabel="epoch", ylabel="total loss")
            figure.tight_layout()
            figure.savefig(run_path / "plots" / "training_losses.png", dpi=120)
            plt.close(figure)
    return result
