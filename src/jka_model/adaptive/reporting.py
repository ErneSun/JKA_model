"""Nested-seed V0.9 aggregation and compact review artifacts."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable
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
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
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
        {int(horizon) for row in records for horizon in row["closed_loop_by_horizon"]}
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
        "observable_status",
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    fractions = [
        sum(str(row.get(field, row.get("physics_status", "FAIL"))) == "PASS" for row in records)
        / len(records)
        for field in statuses
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
    problem_names = {
        str(row.get("problem_name", "cylinder_wake_2d_controlled_inlet")) for row in records
    }
    observable_objectives = {
        str(row.get("observable_objective", "legacy_cylinder_observables")) for row in records
    }
    if len(problem_names) != 1 or len(observable_objectives) != 1:
        raise ValueError("V0.9 formal evidence cannot mix problems or observable objectives")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(int(row["backbone_seed"]), str(row["condition_mode"]))].append(row)

    def nested_gate_support(
        status_field: str,
        *,
        fallback: Callable[[dict[str, Any]], str],
        evaluated_modes: set[str] | None = None,
    ) -> dict[str, Any]:
        selected_modes = modes if evaluated_modes is None else modes & evaluated_modes
        if not selected_modes:
            raise ValueError("V0.9 nested support selected no condition modes")
        per_seed: dict[str, Any] = {}
        for seed in sorted(seeds):
            modes_for_seed: dict[str, Any] = {}
            for mode in sorted(selected_modes):
                rows = grouped[(seed, mode)]
                statuses = [
                    str(row[status_field]) if status_field in row else fallback(row) for row in rows
                ]
                passed = sum(status == "PASS" for status in statuses)
                inconclusive = sum(status == "INCONCLUSIVE" for status in statuses)
                required = math.ceil(2.0 * len(statuses) / 3.0)
                if passed >= required:
                    status = "PASS"
                elif passed + inconclusive < required:
                    status = "FAIL"
                else:
                    status = "INCONCLUSIVE"
                modes_for_seed[mode] = {
                    "operator_init_pass_fraction": passed / len(statuses),
                    "inconclusive_count": inconclusive,
                    "status": status,
                    "supported": status == "PASS",
                }
            mode_statuses = [item["status"] for item in modes_for_seed.values()]
            joint_status = (
                "PASS"
                if all(status == "PASS" for status in mode_statuses)
                else "FAIL"
                if any(status == "FAIL" for status in mode_statuses)
                else "INCONCLUSIVE"
            )
            per_seed[str(seed)] = {
                "modes": modes_for_seed,
                "joint_status": joint_status,
                "joint_supported": joint_status == "PASS",
            }
        joint_statuses = [item["joint_status"] for item in per_seed.values()]
        support_count = sum(status == "PASS" for status in joint_statuses)
        inconclusive_count = sum(status == "INCONCLUSIVE" for status in joint_statuses)
        required = math.ceil(2.0 * len(joint_statuses) / 3.0)
        if support_count >= required:
            aggregate_status = "SUPPORTED"
        elif support_count + inconclusive_count < required:
            aggregate_status = "NOT_SUPPORTED"
        else:
            aggregate_status = "INCONCLUSIVE"
        return {
            "status": aggregate_status,
            "backbone_support_fraction": support_count / len(per_seed),
            "inconclusive_backbone_count": inconclusive_count,
            "nested_seed_support": per_seed,
        }

    operator_support = nested_gate_support(
        "operator_explained_status",
        fallback=lambda row: (
            "PASS" if float(row["operator_explained_fraction"]) >= 0.02 else "FAIL"
        ),
    )
    dynamic_support = nested_gate_support(
        "dynamic_over_condition_only_status",
        fallback=lambda row: (
            "PASS"
            if float(row.get("dynamic_over_condition_only_gain", row["dynamic_over_static_gain"]))
            >= 0.02
            else "FAIL"
        ),
    )
    condition_only_support = nested_gate_support(
        "condition_only_status", fallback=lambda row: "INCONCLUSIVE"
    )
    condition_observer_support = nested_gate_support(
        "condition_observer_status",
        fallback=lambda row: "INCONCLUSIVE",
        evaluated_modes={"latent_inferred"},
    )
    paired_history_support = nested_gate_support(
        "paired_identifiability_status", fallback=lambda row: "INCONCLUSIVE"
    )
    observable_support = nested_gate_support(
        "observable_status",
        fallback=lambda row: str(row.get("physics_status", "FAIL")),
    )
    representation_support = nested_gate_support(
        "representation_physical_floor_status",
        fallback=lambda row: "INCONCLUSIVE",
    )
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
    backbone_fraction = sum(bool(item["joint_supported"]) for item in per_backbone.values()) / len(
        per_backbone
    )
    known_fraction = sum(
        bool(item["modes"]["known"]["supported"]) for item in per_backbone.values()
    ) / len(per_backbone)
    latent_fraction = sum(
        bool(item["modes"]["latent_inferred"]["supported"]) for item in per_backbone.values()
    ) / len(per_backbone)
    known_decision = "SUPPORTED" if known_fraction >= 2.0 / 3.0 else "NOT_SUPPORTED"
    latent_decision = "SUPPORTED" if latent_fraction >= 2.0 / 3.0 else "NOT_SUPPORTED"
    adaptive_mechanism = "SUPPORTED" if backbone_fraction >= 2.0 / 3.0 else "NOT_SUPPORTED"
    rank_selection = _read(session / "rank_selection.json")
    handoff = _read(session / "v0_8_handoff_audit.json")
    strict_handoff = bool(handoff.get("strict_readiness", True))
    handoff_policy = str(handoff.get("handoff_policy", "strict"))
    adaptive_decision = (
        adaptive_mechanism
        if strict_handoff or adaptive_mechanism == "NOT_SUPPORTED"
        else "CONDITIONALLY_SUPPORTED"
    )
    phase1_attribution_complete = all(
        bool(row.get("claims", {}).get("phase1_error_attribution_complete")) for row in records
    )
    phase1_artifacts_complete = all(
        (
            (source.parent / "error_attribution.csv").is_file()
            and (source.parent.parent / "logs" / "gradient_geometry.jsonl").is_file()
            and bool(
                _read(source.parent / "training_summary.json")
                .get("phase1", {})
                .get("observable_scale_state", {})
                .get("split_fingerprint")
            )
            and int(
                _read(source.parent / "training_summary.json")
                .get("phase1", {})
                .get("gradient_audit_records", 0)
            )
            > 0
        )
        for source in sources
    )
    phase2_enabled = all(
        bool(row.get("claims", {}).get("phase2_factorized_operator")) for row in records
    )
    phase2_artifacts_complete = bool(
        phase2_enabled
        and all(
            (source.parent / "condition_observer_metrics.json").is_file()
            and (source.parent / "matched_history_pairs.json").is_file()
            for source in sources
        )
    )
    phase2_classification = (
        "DYNAMIC_ADAPTIVE_KOOPMAN_SUPPORTED"
        if dynamic_support["status"] == "SUPPORTED"
        and paired_history_support["status"] == "SUPPORTED"
        else "PARAMETERIZED_KOOPMAN_SUPPORTED; HISTORY_ADAPTATION_NOT_REQUIRED"
        if condition_only_support["status"] == "SUPPORTED"
        and condition_observer_support["status"] == "SUPPORTED"
        else "LATENT_CONDITION_NOT_IDENTIFIABLE"
        if condition_observer_support["status"] == "NOT_SUPPORTED"
        else "PHASE2_INCONCLUSIVE"
    )
    v1_ready = bool(
        strict_handoff
        and backbone_fraction == 1.0
        and representation_support["status"] == "SUPPORTED"
        and phase1_attribution_complete
        and phase1_artifacts_complete
    )
    decision = {
        "schema_version": 5,
        "physical_problem": next(iter(problem_names)),
        "observable_objective": next(iter(observable_objectives)),
        "v0_8_strict_readiness": "PASS" if strict_handoff else "NOT_READY",
        "v0_8_handoff_policy": handoff_policy,
        "evidence_tier": "CONFIRMATORY" if strict_handoff else "EXPLORATORY_CONDITIONAL",
        "v0_8_route": handoff["route"],
        "context_family": handoff["context_family"],
        "variable_condition_data": "PASS",
        "known_condition_adaptation": known_decision,
        "latent_inferred_adaptation": latent_decision,
        "low_rank_operator_adaptation": adaptive_decision,
        "adaptive_mechanism_result": adaptive_mechanism,
        "operator_explained_residual": operator_support["status"],
        "dynamic_operator_adaptation": dynamic_support["status"],
        "condition_parameterized_operator": condition_only_support["status"],
        "condition_observer": condition_observer_support["status"],
        "paired_history_identifiability": paired_history_support["status"],
        "phase2_classification": phase2_classification,
        "phase2_diagnostic_artifacts": ("COMPLETE" if phase2_artifacts_complete else "INCOMPLETE"),
        "observable_support": observable_support["status"],
        "representation_physical_floor": representation_support["status"],
        "phase1_diagnosis": (
            "REPRESENTATION_BLOCKED"
            if representation_support["status"] == "NOT_SUPPORTED"
            else "OPERATOR_OPTIMIZATION_IDENTIFIABLE"
            if representation_support["status"] == "SUPPORTED"
            else "INCONCLUSIVE"
        ),
        "phase1_error_attribution": ("COMPLETE" if phase1_attribution_complete else "INCOMPLETE"),
        "phase1_diagnostic_artifacts": ("COMPLETE" if phase1_artifacts_complete else "INCOMPLETE"),
        "numerical_stability": (
            "PASS"
            if all(
                row.get("numerical_stability", row["long_rollout_stability"]) == "PASS"
                for row in records
            )
            else "FAIL"
        ),
        "long_rollout_skill": (
            "PASS"
            if all(
                row.get("long_rollout_skill", row.get("all_horizons_status")) == "PASS"
                for row in records
            )
            else "FAIL"
        ),
        "long_rollout_stability": (
            "PASS"
            if all(
                row.get("numerical_stability", row["long_rollout_stability"]) == "PASS"
                for row in records
            )
            else "FAIL"
        ),
        "physics_status": (
            "PASS" if all(row["physics_status"] == "PASS" for row in records) else "FAIL"
        ),
        "strict_all_run_status": {
            "numerical_stability_pass_count": sum(
                row.get("numerical_stability", row["long_rollout_stability"]) == "PASS"
                for row in records
            ),
            "long_rollout_skill_pass_count": sum(
                row.get("long_rollout_skill", row.get("all_horizons_status")) == "PASS"
                for row in records
            ),
            "long_rollout_pass_count": sum(
                row.get("long_rollout_skill", row.get("all_horizons_status")) == "PASS"
                for row in records
            ),
            "physics_pass_count": sum(row["physics_status"] == "PASS" for row in records),
            "formal_run_count": len(records),
            "requires_all_runs": True,
        },
        "v1_0_readiness": "READY" if v1_ready else "NOT_READY",
        "v1_0_ready": v1_ready,
        "scientific_backbone_support_fraction": backbone_fraction,
        "scientific_seed_threshold": 2.0 / 3.0,
        "v1_0_seed_threshold": 1.0,
        "selected_rank": int(rank_selection["selected_rank"]),
        "rank_physics_feasibility": (
            "PASS"
            if int(rank_selection["selected_rank"])
            in {int(value) for value in rank_selection.get("physics_eligible_ranks", [])}
            else "FAIL"
        ),
        "rank_selection_contract": (
            "PASS" if bool(rank_selection.get("constraints_satisfied")) else "FAIL"
        ),
        "formal_run_count": len(records),
        "nested_seed_support": per_backbone,
        "independent_gate_support": {
            "operator_explained_residual": operator_support,
            "dynamic_operator_adaptation": dynamic_support,
            "condition_parameterized_operator": condition_only_support,
            "condition_observer": condition_observer_support,
            "paired_history_identifiability": paired_history_support,
            "observables": observable_support,
            "representation_physical_floor": representation_support,
        },
        "claims": {
            "backbone_frozen": True,
            "context_frozen": True,
            "A0_frozen": True,
            "additive_residual_enabled": False,
            "persistent_z_R_present": False,
            "phase2_factorized_operator": phase2_enabled,
            "oracle_condition_curriculum_train_only": phase2_enabled,
            "locked_latent_evaluation_is_teacher_free": phase2_enabled,
            "known_oracle_excludes_observer_gate": phase2_enabled,
            "causal_observer_uses_state_mean_trend": phase2_enabled,
            "dynamic_context_is_condition_residualized": phase2_enabled,
            "physics_feasible_rank_selection": bool(
                rank_selection.get("physics_eligible_ranks") is not None
            ),
            "innovation_variance_floor": False,
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
    epoch_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    observable_gate_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    gradient_audit_rows: list[dict[str, Any]] = []
    matched_pair_rows: list[dict[str, Any]] = []
    observer_rows: list[dict[str, Any]] = []
    for source in sources:
        root = source.parent
        source_decision = _read(source)
        for filename, target in (
            ("rollout_metrics.csv", rollout_rows),
            ("physical_metrics.csv", physical_rows),
            ("matched_history_pairs.csv", matched_pair_rows),
        ):
            source_csv = root / filename
            if not source_csv.is_file():
                continue
            with source_csv.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
                if filename == "matched_history_pairs.csv":
                    rows = [row for row in rows if "pair_index" in row]
                target.extend(dict(row, source_file=str(source_csv)) for row in rows)
        observer_path = root / "condition_observer_metrics.json"
        if observer_path.is_file():
            observer_rows.append(
                {
                    "backbone_seed": source_decision["backbone_seed"],
                    "condition_mode": source_decision["condition_mode"],
                    "operator_init_seed": source_decision["operator_init_seed"],
                    **_read(observer_path),
                }
            )
        observable_gate_path = root / "observable_gate_results.csv"
        if observable_gate_path.is_file():
            with observable_gate_path.open(newline="", encoding="utf-8") as stream:
                observable_gate_rows.extend(
                    dict(row, source_file=str(observable_gate_path))
                    for row in csv.DictReader(stream)
                )
        attribution_path = root / "error_attribution.csv"
        if attribution_path.is_file():
            with attribution_path.open(newline="", encoding="utf-8") as stream:
                attribution_rows.extend(
                    dict(row, source_file=str(attribution_path)) for row in csv.DictReader(stream)
                )
        summary = _read(root.parent / "evaluation" / "training_summary.json")
        training_row = {
            "backbone_seed": source_decision["backbone_seed"],
            "condition_mode": source_decision["condition_mode"],
            "operator_init_seed": source_decision["operator_init_seed"],
            "completed_epochs": summary["completed_epochs"],
            "test_status": summary["test_locked_confirmation"],
        }
        training_row.update(
            {
                f"validation_{key}": value
                for key, value in summary["validation"].items()
                if isinstance(value, int | float | str | bool)
            }
        )
        training_row["curriculum"] = json.dumps(summary.get("curriculum", {}), sort_keys=True)
        training_row["claims"] = json.dumps(summary.get("claims", {}), sort_keys=True)
        phase1_summary = summary.get("phase1", {})
        training_row["phase1_gradient_audit_records"] = phase1_summary.get(
            "gradient_audit_records", 0
        )
        training_row["phase1_minimum_gradient_cosine"] = phase1_summary.get(
            "minimum_gradient_cosine"
        )
        scale_state = phase1_summary.get("observable_scale_state") or {}
        training_row["phase1_scale_split_fingerprint"] = scale_state.get("split_fingerprint")
        training_rows.append(training_row)
        gradient_path = root.parent / "logs" / "gradient_geometry.jsonl"
        if gradient_path.is_file():
            for line in gradient_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    gradient_audit_rows.append(
                        {
                            "backbone_seed": source_decision["backbone_seed"],
                            "condition_mode": source_decision["condition_mode"],
                            "operator_init_seed": source_decision["operator_init_seed"],
                            **json.loads(line),
                        }
                    )
        epoch_path = root.parent / "logs" / "epoch_metrics.csv"
        if epoch_path.is_file():
            with epoch_path.open(newline="", encoding="utf-8") as stream:
                epoch_rows.extend(
                    {
                        "backbone_seed": source_decision["backbone_seed"],
                        "condition_mode": source_decision["condition_mode"],
                        "operator_init_seed": source_decision["operator_init_seed"],
                        **row,
                    }
                    for row in csv.DictReader(stream)
                )
        for name, gate in source_decision.get("scientific_gates", {}).items():
            gate_rows.append(
                {
                    "backbone_seed": source_decision["backbone_seed"],
                    "condition_mode": source_decision["condition_mode"],
                    "operator_init_seed": source_decision["operator_init_seed"],
                    "gate": name,
                    **{
                        key: json.dumps(value, sort_keys=True)
                        if isinstance(value, dict | list)
                        else value
                        for key, value in gate.items()
                    },
                }
            )
    _write_csv(evaluation / "rollout_metrics.csv", rollout_rows)
    _write_csv(evaluation / "physical_metrics.csv", physical_rows)
    _write_csv(evaluation / "training_summary.csv", training_rows)
    _write_csv(evaluation / "training_epoch_metrics.csv", epoch_rows)
    _write_csv(evaluation / "scientific_gate_results.csv", gate_rows)
    _write_csv(evaluation / "observable_gate_results.csv", observable_gate_rows)
    _write_csv(evaluation / "error_attribution.csv", attribution_rows)
    _write_csv(evaluation / "gradient_geometry.csv", gradient_audit_rows)
    _write_csv(evaluation / "matched_history_pairs.csv", matched_pair_rows)
    _write_csv(evaluation / "condition_observer_metrics.csv", observer_rows)
    _simple_plots(output, records)
    report = (
        "# V0.9 scientific report\n\n"
        f"PHYSICAL PROBLEM: {decision['physical_problem']}  \n"
        f"V0.8 STRICT READINESS: {decision['v0_8_strict_readiness']}  \n"
        f"V0.8 HANDOFF POLICY: {decision['v0_8_handoff_policy']}  \n"
        f"EVIDENCE TIER: {decision['evidence_tier']}  \n"
        f"V0.8 ROUTE: {decision['v0_8_route']}  \n"
        f"CONTEXT FAMILY: {decision['context_family']}  \n"
        f"VARIABLE-CONDITION DATA: {decision['variable_condition_data']}  \n"
        f"KNOWN-CONDITION ADAPTATION: {known_decision}  \n"
        f"LATENT-INFERRED ADAPTATION: {latent_decision}  \n"
        f"LOW-RANK OPERATOR ADAPTATION: {adaptive_decision}  \n"
        f"ADAPTIVE MECHANISM RESULT: {adaptive_mechanism}  \n"
        f"OPERATOR-EXPLAINED RESIDUAL: {decision['operator_explained_residual']}  \n"
        f"DYNAMIC OPERATOR ADAPTATION: {decision['dynamic_operator_adaptation']}  \n"
        f"CONDITION-PARAMETERIZED OPERATOR: "
        f"{decision['condition_parameterized_operator']}  \n"
        f"LATENT CONDITION OBSERVER: {decision['condition_observer']}  \n"
        f"PAIRED HISTORY IDENTIFIABILITY: "
        f"{decision['paired_history_identifiability']}  \n"
        f"PHASE-2 CLASSIFICATION: {decision['phase2_classification']}  \n"
        f"OBSERVABLE SUPPORT: {decision['observable_support']}  \n"
        f"REPRESENTATION PHYSICAL FLOOR: {decision['representation_physical_floor']}  \n"
        f"PHASE-1 DIAGNOSIS: {decision['phase1_diagnosis']}  \n"
        f"PHASE-1 DIAGNOSTIC ARTIFACTS: {decision['phase1_diagnostic_artifacts']}  \n"
        f"NUMERICAL STABILITY: {decision['numerical_stability']}  \n"
        f"LONG-ROLLOUT SKILL: {decision['long_rollout_skill']}  \n"
        f"PHYSICS STATUS: {decision['physics_status']}  \n"
        f"SELECTED-RANK PHYSICS FEASIBILITY: {decision['rank_physics_feasibility']}  \n"
        f"RANK-SELECTION FULL CONTRACT: {decision['rank_selection_contract']}  \n"
        f"STRICT ALL-RUN STATUS: "
        f"numerical={decision['strict_all_run_status']['numerical_stability_pass_count']}/"
        f"{decision['strict_all_run_status']['formal_run_count']}, "
        f"rollout_skill={decision['strict_all_run_status']['long_rollout_skill_pass_count']}/"
        f"{decision['strict_all_run_status']['formal_run_count']}, "
        f"physics={decision['strict_all_run_status']['physics_pass_count']}/"
        f"{decision['strict_all_run_status']['formal_run_count']}  \n"
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
                bool(epoch_rows),
                len(gate_rows) >= 6 * len(records),
                bool(observable_gate_rows),
                bool(observer_rows) if phase2_enabled else True,
                phase2_artifacts_complete if phase2_enabled else True,
                len({row["source_file"] for row in attribution_rows}) == len(records),
                len(
                    {
                        (
                            row["backbone_seed"],
                            row["condition_mode"],
                            row["operator_init_seed"],
                        )
                        for row in gradient_audit_rows
                    }
                )
                == len(records),
                sum(bool(row.get("phase1_scale_split_fingerprint")) for row in training_rows)
                == len(records),
            )
        ),
        "formal_decision_count": len(records),
        "training_summary_count": len(training_rows),
        "training_epoch_metric_count": len(epoch_rows),
        "scientific_gate_result_count": len(gate_rows),
        "observable_gate_result_count": len(observable_gate_rows),
        "error_attribution_row_count": len(attribution_rows),
        "gradient_audit_record_count": len(gradient_audit_rows),
        "condition_observer_record_count": len(observer_rows),
        "matched_history_pair_count": len(matched_pair_rows),
        "phase2_artifacts_complete": phase2_artifacts_complete,
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
