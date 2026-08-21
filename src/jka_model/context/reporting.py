"""Nested-seed V0.8 aggregation, compact reports, and required diagnostic figures."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

V0_9_REQUIRED_BACKBONE_FRACTION = 1.0


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "NO_RECORDS"}]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _compact_audit_artifacts(
    session: Path, evaluation: Path, selected_family: str
) -> dict[str, Any]:
    """Retain compact evidence needed to audit training without checkpoints or raw tensors."""

    def read_json(path: Path) -> dict[str, Any] | None:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    physical = {
        path.stem.removeprefix("physical_acceptance_seed_"): read_json(path)
        for path in sorted((session / "data").glob("physical_acceptance_seed_*.json"))
    }
    backbones = {
        path.parent.name.removeprefix("seed_"): read_json(path)
        for path in sorted(session.glob("seeds/seed_*/backbone_acceptance.json"))
    }
    grid = read_json(session / "data" / "grid_adequacy.json")
    route = read_json(session / "v0_7_assessment" / "evaluation" / "memory_classification.json")
    family_selection = read_json(session / "v0_8_family_selection.json")

    candidate_rows: list[dict[str, Any]] = []
    selected_curve_rows: list[dict[str, Any]] = []
    for path in sorted(
        session.glob("seeds/seed_*/candidates/*/init_*/evaluation/training_summary.json")
    ):
        parts = path.parts
        candidate_index = parts.index("candidates")
        family = parts[candidate_index + 1]
        seed = int(parts[candidate_index - 1].removeprefix("seed_"))
        initialization = int(parts[candidate_index + 2].removeprefix("init_"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload.get("validation", {})
        candidate_rows.append(
            {
                "backbone_seed": seed,
                "context_init_seed": initialization,
                "family": family,
                "completed_epochs": payload.get("completed_epochs"),
                "validation_residual_nrmse": validation.get("residual_nrmse"),
                "validation_residual_standardized_mse": validation.get("residual_standardized_mse"),
                "validation_adequacy_r2": validation.get("adequacy_r2"),
                "test_status": payload.get("test_locked_confirmation"),
            }
        )
        curve = path.parent.parent / "logs" / "epoch_metrics.csv"
        if family == selected_family.lower() and curve.is_file():
            with curve.open(newline="", encoding="utf-8") as stream:
                selected_curve_rows.extend(
                    {
                        "backbone_seed": seed,
                        "context_init_seed": initialization,
                        "family": family,
                        **row,
                    }
                    for row in csv.DictReader(stream)
                )
    _write_csv(evaluation / "candidate_training_summary.csv", candidate_rows)
    _write_csv(evaluation / "selected_training_curves.csv", selected_curve_rows)
    route_name = None if route is None else route.get("residual_route")
    expected_candidates = 9 if route_name == "R2" else 36
    family_selection_complete = family_selection is not None or route_name == "R2"
    complete = bool(
        len(physical) == 3
        and len(backbones) == 3
        and grid is not None
        and route is not None
        and family_selection_complete
        and len(candidate_rows) == expected_candidates
        and selected_curve_rows
    )
    audit = {
        "complete": complete,
        "physical_acceptance": physical,
        "grid_adequacy": grid,
        "backbone_acceptance": backbones,
        "v0_7_route_assessment": route,
        "v0_8_family_selection": family_selection,
        "candidate_training_summary_count": len(candidate_rows),
        "selected_training_curve_row_count": len(selected_curve_rows),
    }
    (evaluation / "compact_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def aggregate_v0_8_results(session_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    session = Path(session_dir)
    sources = sorted(
        session.glob("seeds/seed_*/contexts/init_*/evaluation/v0_8_scientific_decision.json")
    )
    if not sources:
        raise ValueError("no completed V0.8 context evaluations found")
    records: list[dict[str, Any]] = []
    for source in sources:
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["source_file"] = str(source.resolve())
        records.append(payload)
    routes = {record["v0_7_route_on_new_problem"] for record in records}
    families = {record["context_family"] for record in records}
    if len(routes) != 1 or len(families) != 1:
        raise ValueError("V0.8 formal records disagree on locked route/context family")
    by_backbone: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for record in records:
        seed = int(record["backbone_seed"])
        by_backbone[seed].append(record)
        test = record["test_locked_confirmation"]
        ablated = record["context_ablation"]
        rows.append(
            {
                "backbone_seed": seed,
                "context_init_seed": int(record["context_init_seed"]),
                "route": record["v0_7_route_on_new_problem"],
                "context_family": record["context_family"],
                "residual_nrmse": test["residual_nrmse"],
                "residual_r2": test["residual_r2"],
                "adequacy_r2": test["adequacy_r2"],
                "adequacy_correlation": test["adequacy_correlation"],
                "context_ablation_nrmse": ablated["residual_nrmse"],
                "context_ablation_gain": record["context_ablation_gain"],
                "history_over_shuffled_gain": record["history_over_shuffled_gain"],
                "context_effective_rank": record["context_diagnostics"]["effective_rank"],
                "context_collapsed": record["context_diagnostics"]["collapsed"],
                "context_rank_status": record["context_rank_status"],
                "koopman_adequacy": record["koopman_adequacy"],
                "closed_loop_utility": record["closed_loop_utility"],
                "longest_horizon_utility": record["longest_horizon_utility"],
                "physics_status": record["physics_status"],
                "dynamic_context": record["dynamic_context"],
                "source_file": record["source_file"],
            }
        )
    expected_context_seeds = {int(row["context_init_seed"]) for row in rows}
    if len(by_backbone) != 3 or len(expected_context_seeds) != 3:
        raise ValueError("V0.8 formal assessment requires 3 backbone x 3 context seeds")
    if any(
        {int(item["context_init_seed"]) for item in items} != expected_context_seeds
        for items in by_backbone.values()
    ):
        raise ValueError("V0.8 nested context-seed matrix is incomplete")
    consistency_values = {
        float(record["evaluation_thresholds"]["seed_consistency_fraction"]) for record in records
    }
    if len(consistency_values) != 1:
        raise ValueError("V0.8 formal records disagree on seed-consistency threshold")
    consistency = next(iter(consistency_values))
    material_gain_values = {
        float(record["evaluation_thresholds"]["material_relative_gain"]) for record in records
    }
    if len(material_gain_values) != 1:
        raise ValueError("V0.8 formal records disagree on material-gain threshold")
    material_gain = next(iter(material_gain_values))
    backbone_support: dict[str, dict[str, Any]] = {}
    for seed, items in sorted(by_backbone.items()):
        supported = sum(item["dynamic_context"] == "SUPPORTED" for item in items) / len(items)
        rank = sum(item["context_rank_status"] == "PASS" for item in items) / len(items)
        adequacy = sum(item["koopman_adequacy"] == "CALIBRATED" for item in items) / len(items)
        history = sum(item["history_value"] in {"SUPPORTED", "N/A"} for item in items) / len(items)
        physics = sum(item["physics_status"] == "PASS" for item in items) / len(items)
        rollout = sum(item["closed_loop_utility"] == "POSITIVE" for item in items) / len(items)
        longest = sum(item["longest_horizon_utility"] == "POSITIVE" for item in items) / len(items)
        context_supported = supported >= consistency and rank >= consistency
        v0_9_supported = bool(
            context_supported
            and adequacy >= consistency
            and history >= consistency
            and physics >= consistency
            and rollout >= consistency
            and longest >= consistency
        )
        backbone_support[str(seed)] = {
            "context_init_support_fraction": supported,
            "context_rank_pass_fraction": rank,
            "adequacy_pass_fraction": adequacy,
            "history_pass_fraction": history,
            "physics_pass_fraction": physics,
            "positive_rollout_fraction": rollout,
            "longest_horizon_pass_fraction": longest,
            "supported": context_supported and physics >= consistency,
            "v0_9_supported": v0_9_supported,
        }
    across = sum(item["supported"] for item in backbone_support.values()) / len(backbone_support)
    dynamic = "SUPPORTED" if across >= consistency else "NOT_SUPPORTED"
    positive = sum(
        item["positive_rollout_fraction"] >= consistency for item in backbone_support.values()
    ) / len(backbone_support)
    physics = sum(
        item["physics_pass_fraction"] >= consistency for item in backbone_support.values()
    ) / len(backbone_support)
    adequacy = sum(
        item["adequacy_pass_fraction"] >= consistency for item in backbone_support.values()
    ) / len(backbone_support)
    longest = sum(
        item["longest_horizon_pass_fraction"] >= consistency for item in backbone_support.values()
    ) / len(backbone_support)
    route = next(iter(routes))
    family = next(iter(families))
    selection_path = session / "v0_8_family_selection.json"
    family_selection = (
        json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else None
    )
    history_supported = sum(
        row["history_over_shuffled_gain"] is not None
        and float(row["history_over_shuffled_gain"]) >= material_gain
        for row in rows
    ) / len(rows)
    attention_controls_pass = False
    if family == "ATTENTION" and family_selection is not None:
        scores = family_selection["candidate_mean_validation_standardized_mse"]
        attention_controls_pass = bool(
            float(scores["attention"]) <= float(scores["history_mlp"])
            and float(scores["attention"]) <= 0.98 * float(scores["instantaneous_matched"])
        )
    attention = (
        "SUPPORTED"
        if family == "ATTENTION"
        and attention_controls_pass
        and history_supported >= consistency
        and dynamic == "SUPPORTED"
        else ("N/A" if family != "ATTENTION" else "NOT_SUPPORTED")
    )
    joint_readiness = sum(item["v0_9_supported"] for item in backbone_support.values()) / len(
        backbone_support
    )
    readiness = "READY" if joint_readiness >= V0_9_REQUIRED_BACKBONE_FRACTION else "NOT_READY"
    decision = {
        "schema_version": 1,
        "physical_problem": "cylinder_wake_2d",
        "backbone_status": "PASS",
        "v0_7_route_on_new_problem": route,
        "context_family": family,
        "residual_prediction": dynamic,
        "history_value": (
            "SUPPORTED"
            if history_supported >= consistency
            else ("N/A" if route == "R2" else "NOT_SUPPORTED")
        ),
        "koopman_adequacy": "CALIBRATED" if adequacy >= consistency else "UNCALIBRATED",
        "temporal_attention_context": attention,
        "dynamic_context": dynamic,
        "closed_loop_utility": "POSITIVE" if positive >= consistency else "NEUTRAL",
        "longest_horizon_utility": "POSITIVE" if longest >= consistency else "NEUTRAL",
        "physics_status": "PASS" if physics >= consistency else "FAIL",
        "v0_9_operator_adaptation_readiness": readiness,
        "v0_9_ready": readiness == "READY",
        "run_count": len(records),
        "seed_consistency_threshold": consistency,
        "joint_v0_9_support_fraction": joint_readiness,
        "v0_9_required_backbone_fraction": V0_9_REQUIRED_BACKBONE_FRACTION,
        "nested_seed_support": backbone_support,
        "validation_family_selection": family_selection,
        "attention_parameter_matched_control_pass": attention_controls_pass,
        "claims": {
            "A0_frozen": True,
            "eta_t_implemented": False,
            "adaptive_A_t_implemented": False,
            "persistent_z_R_present": False,
            "attention_weights_are_causal_explanations": False,
        },
    }
    output = Path(output_dir)
    evaluation = output / "evaluation"
    plots = output / "plots"
    reports = output / "reports"
    for directory in (evaluation, plots, reports):
        directory.mkdir(parents=True, exist_ok=True)
    _write_csv(evaluation / "context_model_comparison.csv", rows)
    _write_csv(
        evaluation / "context_ablation.csv",
        [
            {
                "backbone_seed": row["backbone_seed"],
                "context_init_seed": row["context_init_seed"],
                "full_nrmse": row["residual_nrmse"],
                "no_context_nrmse": row["context_ablation_nrmse"],
                "relative_gain": row["context_ablation_gain"],
            }
            for row in rows
        ],
    )
    closed_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    for source in sources:
        root = source.parent
        for name, target in (
            ("closed_loop_metrics.csv", closed_rows),
            ("physical_metrics.csv", physical_rows),
        ):
            with (root / name).open(newline="", encoding="utf-8") as stream:
                target.extend(
                    dict(row, source_file=str(root / name)) for row in csv.DictReader(stream)
                )
    metric_fields = (
        "residual_nrmse",
        "residual_r2",
        "adequacy_r2",
        "adequacy_correlation",
        "context_ablation_gain",
        "history_over_shuffled_gain",
        "context_effective_rank",
    )
    aggregate_metrics: dict[str, dict[str, float | int]] = {}
    for field in metric_fields:
        values = [float(row[field]) for row in rows if row.get(field) not in {None, "", "None"}]
        if values:
            mean, std = _mean_std(values)
            aggregate_metrics[field] = {
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
        else:
            aggregate_metrics[field] = {"mean": None, "std": None, "n": 0}
    rollout_horizon_summary: dict[str, dict[str, float | int]] = {}
    for horizon in sorted({int(row["horizon"]) for row in closed_rows}):
        horizon_rows = [row for row in closed_rows if int(row["horizon"]) == horizon]
        gains = [
            float(record["closed_loop_by_horizon"][str(horizon)]["relative_gain"])
            for record in records
        ]
        selected = [float(row["latent_rmse"]) for row in horizon_rows]
        baseline = [float(row["koopman_only_latent_rmse"]) for row in horizon_rows]
        gain_mean, gain_std = _mean_std(gains)
        rollout_horizon_summary[str(horizon)] = {
            "relative_gain_mean": gain_mean,
            "relative_gain_std": gain_std,
            "ratio_of_mean_rmse": statistics.mean(selected) / statistics.mean(baseline),
            "material_gain_fraction": sum(value >= material_gain for value in gains) / len(gains),
            "n": len(gains),
        }
    audit = _compact_audit_artifacts(session, evaluation, family)
    decision["aggregate_metrics"] = aggregate_metrics
    decision["rollout_horizon_summary"] = rollout_horizon_summary
    decision["compact_audit"] = {
        "complete": audit["complete"],
        "candidate_training_summary_count": audit["candidate_training_summary_count"],
        "selected_training_curve_row_count": audit["selected_training_curve_row_count"],
    }
    _write_csv(evaluation / "closed_loop_metrics.csv", closed_rows)
    _write_csv(evaluation / "physical_metrics.csv", physical_rows)
    (evaluation / "v0_8_scientific_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    diagnostic_paths = [source.parent / "diagnostic_series.pt" for source in sources]
    _write_figures(rows, closed_rows, physical_rows, diagnostic_paths, plots)
    report = _scientific_report(decision)
    (reports / "v0_8_scientific_report.md").write_text(report, encoding="utf-8")
    (output / "report.md").write_text(report, encoding="utf-8")
    return decision


def _errorbar(ax: Any, labels: list[str], groups: list[list[float]], ylabel: str) -> None:
    means, stds = zip(*(_mean_std(values) for values in groups), strict=True)
    ax.errorbar(labels, means, yerr=stds, marker="o", capsize=4)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for index, values in enumerate(groups):
        ax.text(index, means[index], f"n={len(values)}", fontsize=8)


def _write_figures(
    rows: list[dict[str, Any]],
    closed_rows: list[dict[str, Any]],
    physical_rows: list[dict[str, Any]],
    diagnostic_paths: list[Path],
    root: Path,
) -> None:
    del closed_rows, physical_rows
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    diagnostics = []
    for path in diagnostic_paths:
        try:
            diagnostics.append(torch.load(path, map_location="cpu", weights_only=False))
        except TypeError:  # pragma: no cover
            diagnostics.append(torch.load(path, map_location="cpu"))
    diagnostic = diagnostics[0]
    rollout = diagnostic.get("representative_rollout")
    if rollout is not None:
        truth = rollout["true_vorticity"]
        predicted = rollout["predicted_vorticity"]
        indices = sorted({0, truth.shape[0] // 2, truth.shape[0] - 1})
        figure, axes = plt.subplots(2, len(indices), figsize=(12, 5), sharex=True, sharey=True)
        limit = float(torch.quantile(truth.abs().flatten(), 0.99).clamp_min(1e-8))
        for column, index in enumerate(indices):
            for row_index, (label, values) in enumerate(
                (("True", truth), ("Context rollout", predicted))
            ):
                axes[row_index, column].imshow(
                    values[index].T,
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    aspect="auto",
                )
                axes[row_index, column].set_title(f"{label}, step {index + 1}")
        figure.suptitle("Representative transient cylinder-wake vorticity (one locked run)")
        figure.tight_layout()
        figure.savefig(root / "cylinder_wake_transient_vorticity.png", dpi=160)
        plt.close(figure)

        aligned_rollouts = [item["representative_rollout"] for item in diagnostics]
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        time = rollout["time"]
        for name, label in (("true_lift", "True"), ("predicted_lift", "Context rollout")):
            values = torch.stack([item[name] for item in aligned_rollouts])
            mean, std = values.mean(0), values.std(0)
            axis.plot(time, mean, label=f"{label} mean")
            axis.fill_between(time, mean - std, mean + std, alpha=0.18)
        axis.set(xlabel="Nondimensional time", ylabel="$C_L$", title="Lift transient and shedding")
        axis.legend()
        axis.grid(alpha=0.25)
        axis.text(0.02, 0.96, f"n={len(aligned_rollouts)}", transform=axis.transAxes, va="top")
        figure.tight_layout()
        figure.savefig(root / "lift_transient_to_shedding.png", dpi=160)
        plt.close(figure)

        for filename, title, selected_name, base_name, ylabel in (
            (
                "closed_loop_error.png",
                "Teacher-free closed-loop latent error",
                "closed_loop_error",
                "koopman_only_error",
                "Latent RMSE",
            ),
            (
                "closure_burden.png",
                "Additive closure burden",
                "closure_burden",
                None,
                "Burden ratio",
            ),
        ):
            figure, axis = plt.subplots(figsize=(6.4, 4.0))
            selected_values = torch.stack([item[selected_name] for item in aligned_rollouts])
            selected_mean, selected_std = selected_values.mean(0), selected_values.std(0)
            axis.plot(time, selected_mean, label="Context residual mean")
            axis.fill_between(
                time, selected_mean - selected_std, selected_mean + selected_std, alpha=0.18
            )
            if base_name is not None:
                base_values = torch.stack([item[base_name] for item in aligned_rollouts])
                base_mean, base_std = base_values.mean(0), base_values.std(0)
                axis.plot(time, base_mean, label="Koopman-only mean")
                axis.fill_between(time, base_mean - base_std, base_mean + base_std, alpha=0.18)
            axis.set(xlabel="Nondimensional time", ylabel=ylabel, title=title)
            axis.legend()
            axis.grid(alpha=0.25)
            axis.text(0.02, 0.96, f"n={len(aligned_rollouts)}", transform=axis.transAxes, va="top")
            figure.tight_layout()
            figure.savefig(root / filename, dpi=160)
            plt.close(figure)

    target_residual = torch.stack([item["residual_target"] for item in diagnostics])
    predicted_residual = torch.stack([item["residual_prediction"] for item in diagnostics])
    target_adequacy = torch.stack([item["adequacy_target"].flatten() for item in diagnostics])
    predicted_adequacy = torch.stack(
        [item["adequacy_prediction"].flatten() for item in diagnostics]
    )
    context = diagnostic["contexts"]
    series_specs = (
        (
            "nominal_koopman_residual_norm.png",
            target_residual.norm(dim=-1),
            predicted_residual.norm(dim=-1),
            "Nominal Koopman residual norm",
        ),
        (
            "predicted_vs_true_adequacy.png",
            target_adequacy,
            predicted_adequacy,
            "Predicted Koopman inadequacy",
        ),
    )
    for filename, truth_values, predicted_values, title in series_specs:
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        for values, label in ((truth_values, "True"), (predicted_values, "Predicted")):
            mean, std = values.mean(0), values.std(0)
            axis.plot(mean, label=f"{label} mean", alpha=0.9)
            axis.fill_between(torch.arange(mean.numel()), mean - std, mean + std, alpha=0.18)
        axis.set(xlabel="Test-window index", title=title)
        axis.legend()
        axis.grid(alpha=0.25)
        axis.text(0.02, 0.96, f"n={len(diagnostics)}", transform=axis.transAxes, va="top")
        figure.tight_layout()
        figure.savefig(root / filename, dpi=160)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.0, 5.0))
    axis.scatter(target_residual.flatten(), predicted_residual.flatten(), s=5, alpha=0.12)
    extent = float(max(target_residual.abs().max(), predicted_residual.abs().max()))
    axis.plot([-extent, extent], [-extent, extent], color="black", linewidth=1)
    axis.set(xlabel="True residual coordinate", ylabel="Predicted", title="Residual prediction")
    figure.tight_layout()
    figure.savefig(root / "predicted_vs_true_residual.png", dpi=160)
    plt.close(figure)

    centered = context - context.mean(dim=0)
    _, _, right = torch.linalg.svd(centered, full_matrices=False)
    principal = centered @ right[:2].T
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(principal[:, 0], principal[:, 1], linewidth=1)
    axis.set(
        xlabel="PC1",
        ylabel="PC2",
        title="Representative context coordinates (PCA; one locked run)",
    )
    figure.tight_layout()
    figure.savefig(root / "context_coordinates_pca.png", dpi=160)
    plt.close(figure)

    specifications = [
        (
            "context_model_residual_error.png",
            "Selected context vs zero-context ablation",
            ("residual_nrmse", "context_ablation_nrmse"),
        ),
        (
            "real_vs_shuffled_history.png",
            "Real-history relative gain over shuffled history",
            ("history_over_shuffled_gain",),
        ),
    ]
    for filename, title, fields in specifications:
        groups = [
            [float(row[field]) for row in rows if row.get(field) not in {None, "", "None"}]
            for field in fields
        ]
        groups = [values for values in groups if values]
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        if groups:
            _errorbar(axis, list(fields)[: len(groups)], groups, "Metric")
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(root / filename, dpi=160)
        plt.close(figure)


def _scientific_report(decision: dict[str, Any]) -> str:
    metrics = decision["aggregate_metrics"]
    horizons = decision["rollout_horizon_summary"]

    def metric_line(label: str, field: str, *, include_n: bool = False) -> str:
        item = metrics[field]
        if not item["n"]:
            return f"- {label}: N/A"
        suffix = f" (n={item['n']})" if include_n else ""
        return f"- {label}: {item['mean']:.6g} ± {item['std']:.6g}{suffix}"

    lines = [
        "# V0.8 scientific report",
        "",
        f"PHYSICAL PROBLEM: {decision['physical_problem']}  ",
        f"BACKBONE STATUS: {decision['backbone_status']}  ",
        f"V0.7 ROUTE ON NEW PROBLEM: {decision['v0_7_route_on_new_problem']}  ",
        f"CONTEXT FAMILY: {decision['context_family']}  ",
        f"RESIDUAL PREDICTION: {decision['residual_prediction']}  ",
        f"HISTORY VALUE: {decision['history_value']}  ",
        f"KOOPMAN ADEQUACY: {decision['koopman_adequacy']}  ",
        f"DYNAMIC CONTEXT: {decision['dynamic_context']}  ",
        f"CLOSED LOOP UTILITY: {decision['closed_loop_utility']}  ",
        f"LONGEST HORIZON UTILITY: {decision['longest_horizon_utility']}  ",
        f"PHYSICS STATUS: {decision['physics_status']}  ",
        f"V0.9 OPERATOR-ADAPTATION READINESS: {decision['v0_9_operator_adaptation_readiness']}",
        "",
        "## Aggregate evidence",
        "",
        metric_line("Residual NRMSE", "residual_nrmse", include_n=True),
        metric_line("Residual R2", "residual_r2"),
        metric_line("History-over-shuffled gain", "history_over_shuffled_gain"),
        metric_line("Context effective rank", "context_effective_rank"),
        metric_line("Adequacy R2", "adequacy_r2"),
        "",
        "## Teacher-free rollout by horizon",
        "",
        "| Horizon | Mean relative gain | Ratio of mean RMSE | Material-gain fraction | n |",
        "|---:|---:|---:|---:|---:|",
    ]
    for horizon, item in sorted(horizons.items(), key=lambda value: int(value[0])):
        lines.append(
            f"| {horizon} | {item['relative_gain_mean']:.6g} | "
            f"{item['ratio_of_mean_rmse']:.6g} | {item['material_gain_fraction']:.6g} | "
            f"{item['n']} |"
        )
    lines.extend(
        [
            "",
            "## Nested backbone support",
            "",
            "| Seed | Context | Rank | Adequacy | History | Rollout | Longest | Physics | "
            "V0.9 joint |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for seed, item in decision["nested_seed_support"].items():
        lines.append(
            f"| {seed} | {item['context_init_support_fraction']:.3f} | "
            f"{item['context_rank_pass_fraction']:.3f} | "
            f"{item['adequacy_pass_fraction']:.3f} | {item['history_pass_fraction']:.3f} | "
            f"{item['positive_rollout_fraction']:.3f} | "
            f"{item['longest_horizon_pass_fraction']:.3f} | "
            f"{item['physics_pass_fraction']:.3f} | {item['v0_9_supported']} |"
        )
    lines.extend(
        [
            "",
            f"Formal nested run count: {decision['run_count']}. Joint V0.9 support fraction: "
            f"{decision['joint_v0_9_support_fraction']:.6g}; V0.9 required: "
            f"{decision['v0_9_required_backbone_fraction']:.6g}. Context-init consistency "
            f"within each backbone: {decision['seed_consistency_threshold']:.6g}.",
            f"Compact audit complete: {decision['compact_audit']['complete']}.",
            "",
            "The additive residual is a utility probe. A0 remains frozen; eta_t, adaptive A_t, "
            "and persistent z_R are absent. Attention weights, when available, are diagnostics "
            "rather than causal explanations.",
            "",
        ]
    )
    return "\n".join(lines)
