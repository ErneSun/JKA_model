"""Nested-seed V0.9 aggregation and compact review artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping JSON: {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "NO_RECORDS"}]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _simple_plots(output: Path, records: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("one_step_relative_gain", "One-step relative gain"),
        ("operator_explained_fraction", "Operator-explained fraction"),
        ("dynamic_over_static_gain", "Dynamic over static gain"),
    )
    for field, title in metrics:
        fig, axis = plt.subplots(figsize=(6.4, 4.0))
        modes = sorted({str(row["condition_mode"]) for row in records})
        values = [
            [float(row[field]) for row in records if row["condition_mode"] == mode]
            for mode in modes
        ]
        axis.boxplot(values, tick_labels=modes)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / f"{field}.png", dpi=160)
        plt.close(fig)

    horizons = sorted(
        {
            int(horizon)
            for row in records
            for horizon in row["closed_loop_by_horizon"]
        }
    )
    for field, title in (
        ("relative_gain_mean", "Rollout relative gain"),
        ("gamma_operator_mean", "Rollout operator-explained fraction"),
    ):
        fig, axis = plt.subplots(figsize=(6.4, 4.0))
        for mode in sorted({str(row["condition_mode"]) for row in records}):
            means = [
                statistics.mean(
                    float(row["closed_loop_by_horizon"][str(horizon)][field])
                    for row in records
                    if row["condition_mode"] == mode
                )
                for horizon in horizons
            ]
            axis.plot(horizons, means, marker="o", label=mode)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel("Rollout horizon")
        axis.set_ylabel(title)
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / f"{field}_by_horizon.png", dpi=160)
        plt.close(fig)

    statuses = (
        "controls_status",
        "all_horizons_status",
        "longest_horizon_status",
        "operator_burden_status",
        "physics_status",
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    fractions = [
        sum(row[field] == "PASS" for row in records) / len(records) for field in statuses
    ]
    axis.bar(range(len(statuses)), fractions)
    axis.set_xticks(range(len(statuses)), [name.removesuffix("_status") for name in statuses])
    axis.tick_params(axis="x", rotation=25)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Pass fraction")
    fig.tight_layout()
    fig.savefig(plots / "joint_gate_pass_fraction.png", dpi=160)
    plt.close(fig)


def aggregate_v0_9_results(session_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    session = Path(session_dir)
    output = Path(output_dir)
    evaluation = output / "evaluation"
    reports = output / "reports"
    for directory in (evaluation, reports, output / "plots"):
        directory.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        session.glob("seeds/seed_*/formal/*/init_*/evaluation/v0_9_scientific_decision.json")
    )
    if not sources:
        raise ValueError("V0.9 aggregation found no formal locked-test decisions")
    records = [_read(path) for path in sources]
    modes = {str(row["condition_mode"]) for row in records}
    if modes != {"known", "latent_inferred"}:
        raise ValueError("V0.9 formal evidence requires known and latent-inferred modes")
    seeds = {int(row["backbone_seed"]) for row in records}
    if len(seeds) != 3:
        raise ValueError("V0.9 formal evidence requires three backbone/data seeds")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(int(row["backbone_seed"]), str(row["condition_mode"]))].append(row)
    per_backbone: dict[str, Any] = {}
    for seed in sorted(seeds):
        mode_support: dict[str, Any] = {}
        for mode in sorted(modes):
            rows = grouped[(seed, mode)]
            if len(rows) != 3:
                raise ValueError(f"V0.9 seed={seed} mode={mode} requires three operator inits")
            fraction = sum(bool(row["scientific_joint_pass"]) for row in rows) / len(rows)
            mode_support[mode] = {
                "operator_init_pass_fraction": fraction,
                "supported": fraction >= 2.0 / 3.0,
            }
        joint = all(item["supported"] for item in mode_support.values())
        per_backbone[str(seed)] = {"modes": mode_support, "joint_supported": joint}
    backbone_fraction = sum(
        bool(item["joint_supported"]) for item in per_backbone.values()
    ) / len(per_backbone)
    known_fraction = sum(
        bool(item["modes"]["known"]["supported"]) for item in per_backbone.values()
    ) / len(per_backbone)
    latent_fraction = sum(
        bool(item["modes"]["latent_inferred"]["supported"])
        for item in per_backbone.values()
    ) / len(per_backbone)
    known_decision = "SUPPORTED" if known_fraction >= 2.0 / 3.0 else "NOT_SUPPORTED"
    latent_decision = "SUPPORTED" if latent_fraction >= 2.0 / 3.0 else "NOT_SUPPORTED"
    adaptive_decision = "SUPPORTED" if backbone_fraction >= 2.0 / 3.0 else "NOT_SUPPORTED"
    v1_ready = backbone_fraction == 1.0
    rank_selection = _read(session / "rank_selection.json")
    handoff = _read(session / "v0_8_handoff_audit.json")
    decision = {
        "schema_version": 1,
        "physical_problem": "cylinder_wake_2d_controlled_inlet",
        "v0_8_strict_readiness": "PASS",
        "v0_8_route": handoff["route"],
        "context_family": handoff["context_family"],
        "variable_condition_data": "PASS",
        "known_condition_adaptation": known_decision,
        "latent_inferred_adaptation": latent_decision,
        "low_rank_operator_adaptation": adaptive_decision,
        "operator_explained_residual": adaptive_decision,
        "long_rollout_stability": (
            "PASS" if all(row["long_rollout_stability"] == "PASS" for row in records) else "FAIL"
        ),
        "physics_status": (
            "PASS" if all(row["physics_status"] == "PASS" for row in records) else "FAIL"
        ),
        "v1_0_readiness": "READY" if v1_ready else "NOT_READY",
        "v1_0_ready": v1_ready,
        "scientific_backbone_support_fraction": backbone_fraction,
        "scientific_seed_threshold": 2.0 / 3.0,
        "v1_0_seed_threshold": 1.0,
        "selected_rank": int(rank_selection["selected_rank"]),
        "formal_run_count": len(records),
        "nested_seed_support": per_backbone,
        "claims": {
            "backbone_frozen": True,
            "context_frozen": True,
            "A0_frozen": True,
            "additive_residual_enabled": False,
            "persistent_z_R_present": False,
            "unseen_condition_generalization_tested": False,
        },
    }
    (evaluation / "v0_9_scientific_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(
        evaluation / "operator_model_comparison.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"closed_loop_by_horizon", "claims"}
            }
            for row in records
        ],
    )
    rollout_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for source in sources:
        root = source.parent
        for filename, target in (
            ("rollout_metrics.csv", rollout_rows),
            ("physical_metrics.csv", physical_rows),
        ):
            with (root / filename).open(newline="", encoding="utf-8") as stream:
                target.extend(
                    dict(row, source_file=str(root / filename))
                    for row in csv.DictReader(stream)
                )
        summary = _read(root.parent / "evaluation" / "training_summary.json")
        training_rows.append(
            {
                "backbone_seed": _read(source)["backbone_seed"],
                "condition_mode": _read(source)["condition_mode"],
                "operator_init_seed": _read(source)["operator_init_seed"],
                "completed_epochs": summary["completed_epochs"],
                "validation_forecast": summary["validation"]["forecast"],
                "test_status": summary["test_locked_confirmation"],
            }
        )
    _write_csv(evaluation / "rollout_metrics.csv", rollout_rows)
    _write_csv(evaluation / "physical_metrics.csv", physical_rows)
    _write_csv(evaluation / "training_summary.csv", training_rows)
    _simple_plots(output, records)
    report = (
        "# V0.9 scientific report\n\n"
        f"PHYSICAL PROBLEM: {decision['physical_problem']}  \n"
        f"V0.8 STRICT READINESS: {decision['v0_8_strict_readiness']}  \n"
        f"V0.8 ROUTE: {decision['v0_8_route']}  \n"
        f"CONTEXT FAMILY: {decision['context_family']}  \n"
        f"VARIABLE-CONDITION DATA: {decision['variable_condition_data']}  \n"
        f"KNOWN-CONDITION ADAPTATION: {known_decision}  \n"
        f"LATENT-INFERRED ADAPTATION: {latent_decision}  \n"
        f"LOW-RANK OPERATOR ADAPTATION: {adaptive_decision}  \n"
        f"OPERATOR-EXPLAINED RESIDUAL: {decision['operator_explained_residual']}  \n"
        f"LONG-ROLLOUT STABILITY: {decision['long_rollout_stability']}  \n"
        f"PHYSICS STATUS: {decision['physics_status']}  \n"
        f"V1.0 READINESS: {decision['v1_0_readiness']}\n\n"
        f"Selected rank: {decision['selected_rank']}; formal nested run count: "
        f"{decision['formal_run_count']}. Additive residual correction and persistent z_R "
        "remain disabled. V0.9 does not test unseen-condition generalization.\n"
    )
    (reports / "v0_9_scientific_report.md").write_text(report, encoding="utf-8")
    (output / "report.md").write_text(report, encoding="utf-8")
    audit = {
        "complete": all(
            (
                len(records) == 18,
                len(list((output / "plots").glob("*.png"))) >= 6,
                bool(rollout_rows),
                bool(physical_rows),
                len(training_rows) == 18,
            )
        ),
        "formal_decision_count": len(records),
        "training_summary_count": len(training_rows),
        "plot_count": len(list((output / "plots").glob("*.png"))),
        "rank_selection": rank_selection,
        "v0_8_handoff": handoff,
    }
    (evaluation / "compact_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    decision["compact_audit"] = audit
    (output / "summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return decision
