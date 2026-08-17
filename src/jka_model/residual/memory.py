"""Problem-agnostic, trained-result-only V0.7 memory characterization."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PROVENANCE_KEYS = (
    "backbone_checkpoint_sha256",
    "backbone_config_hash",
    "cache_fingerprint",
    "data_fingerprint",
    "split_fingerprint",
    "normalizer_fingerprint",
    "evaluation_trajectory_ids",
)


def _longest(record: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    horizon = max((int(value) for value in record["closed_loop"]), default=0)
    if horizon == 0:
        raise ValueError("evaluation record has no closed-loop horizon")
    return horizon, record["closed_loop"][str(horizon)]


def _closure_seed(record: dict[str, Any]) -> int:
    if "closure_initialization_seed" not in record:
        raise ValueError("evaluation record lacks closure_initialization_seed")
    return int(record["closure_initialization_seed"])


def load_evaluation_records(session_dir: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(Path(session_dir).glob("seeds/seed_*/evaluation/**/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("phase") == "v0.7" and "teacher_forced" in payload:
            payload["source_file"] = str(path.resolve())
            records.append(payload)
    if not records:
        raise ValueError(f"no trained V0.7 evaluation records found below {session_dir}")
    return records


def _same_metric(left: dict[str, Any], right: dict[str, Any], key: str) -> bool:
    return abs(float(left[key]) - float(right[key])) <= 1e-10 * max(
        abs(float(left[key])), abs(float(right[key])), 1.0
    )


def validate_sweep_provenance(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_seed[int(record["seed"])].append(record)
        if record.get("target_encoder_used") is not False:
            raise ValueError("V0.7 residual evaluation must not use the EMA target encoder")
        if record.get("rollout_uses_predicted_history") is not True:
            raise ValueError("V0.7 closed-loop evaluation must feed back predicted history")
        if record.get("physics_used_for_training") is not False:
            raise ValueError("V0.7 closure training must not use a physics loss")
        if "teacher_forced_validation" not in record:
            raise ValueError("V0.7 comparison requires validation metrics for model selection")
    if len(by_seed) < 3:
        raise ValueError("memory characterization requires at least three backbone/data seeds")
    expected_initializations = sorted(
        int(value) for value in records[0]["memory_sweep_config"]["initialization_seeds"]
    )
    parameter_matches: dict[str, float] = {}
    h1_pairing: dict[str, bool] = {}
    for seed, seed_items in by_seed.items():
        reference = seed_items[0]["provenance"]
        for item in seed_items[1:]:
            for key in PROVENANCE_KEYS:
                if item["provenance"][key] != reference[key]:
                    raise ValueError(f"seed {seed} comparison provenance mismatch: {key}")
        actual_initializations = sorted({_closure_seed(item) for item in seed_items})
        if actual_initializations != expected_initializations:
            raise ValueError(
                f"seed {seed} closure initialization sweep incomplete: "
                f"{actual_initializations} != {expected_initializations}"
            )
        for initialization in expected_initializations:
            items = [item for item in seed_items if _closure_seed(item) == initialization]
            histories = sorted(
                int(item["history_length_steps"]) for item in items if item["variant"] == "history"
            )
            expected_histories = list(items[0]["memory_sweep_config"]["history_lengths"])
            if histories != expected_histories:
                raise ValueError(
                    f"seed {seed} initialization {initialization} history sweep incomplete"
                )
            lookup = {
                (str(item["variant"]), int(item["history_length_steps"])): item for item in items
            }
            for required in (("zero", 1), ("linear", 1)):
                if required not in lookup:
                    raise ValueError(f"missing required control {required}")
            tolerance = float(items[0]["memory_sweep_config"]["parameter_match_tolerance"])
            for history in histories:
                for variant in ("instantaneous", "history"):
                    if (variant, history) not in lookup:
                        raise ValueError(f"missing {variant} H={history}")
                if history > 1 and ("shuffled_history", history) not in lookup:
                    raise ValueError(f"missing shuffled-history H={history}")
                ordered = lookup[("history", history)]
                instant = lookup[("instantaneous", history)]
                relative = abs(int(ordered["parameter_count"]) - int(instant["parameter_count"]))
                relative /= max(int(ordered["parameter_count"]), 1)
                key = f"seed_{seed}_init_{initialization}_h{history}"
                parameter_matches[key] = relative
                if relative > tolerance:
                    raise ValueError(f"{key} parameter mismatch exceeds tolerance")
            ordered_h1 = lookup[("history", 1)]
            instant_h1 = lookup[("instantaneous", 1)]
            paired = all(
                _same_metric(ordered_h1[split], instant_h1[split], metric)
                for split in ("teacher_forced", "teacher_forced_validation")
                for metric in ("mse", "normalized_rmse", "r2")
            )
            h1_key = f"seed_{seed}_init_{initialization}"
            h1_pairing[h1_key] = paired
            if not paired:
                raise ValueError(f"H=1 Markovian pair is not training-identical: {h1_key}")
    return {
        "seeds": sorted(by_seed),
        "closure_initialization_seeds": expected_initializations,
        "same_backbone_data_split_normalizer_and_trajectories": True,
        "frozen_online_residual_and_predicted_history_contract": True,
        "h1_markovian_pairing": h1_pairing,
        "parameter_match_relative_differences": parameter_matches,
    }


def _rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        horizon, closed = _longest(record)
        teacher = record["teacher_forced"]
        validation = record["teacher_forced_validation"]
        rows.append(
            {
                "seed": int(record["seed"]),
                "closure_seed": _closure_seed(record),
                "closure_family": record["closure_family"],
                "variant": record["variant"],
                "history_steps": int(record["history_length_steps"]),
                "history_physical_time": float(record["history_length_physical_time"]["mean"]),
                "parameter_count": int(record["parameter_count"]),
                "parameter_matched": bool(record["parameter_matched_control"]),
                "history_shuffled": bool(record["history_shuffled"]),
                "validation_residual_nrmse": float(validation["normalized_rmse"]),
                "validation_residual_r2": float(validation["r2"]),
                "residual_mse": float(teacher["mse"]),
                "residual_nrmse": float(teacher["normalized_rmse"]),
                "residual_r2": float(teacher["r2"]),
                "residual_target_rms": float(teacher["target_rms"]),
                "rollout_horizon": horizon,
                "closed_loop_latent_rmse": float(closed["latent_rmse"]),
                "closed_loop_field_rmse": float(closed["field_rmse"]),
                "closed_loop_relative_l2": float(closed["relative_l2"]),
                "mass_drift": float(closed["mass_drift"]),
                "operator_mse": float(closed["operator_mse"]),
                "closure_burden": float(closed["closure_burden"]),
                "source_file": record["source_file"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["seed"],
            row["closure_seed"],
            row["history_steps"],
            row["variant"],
        ),
    )


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.mean(float(row[field]) for row in rows)


def _stdev(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.stdev(float(row[field]) for row in rows) if len(rows) > 1 else 0.0


def _fraction(values: list[bool]) -> float:
    return sum(values) / len(values)


def _physics_ok(selected: dict[str, Any], zero: dict[str, Any], record: dict[str, Any]) -> bool:
    degradation = float(record["v0_7_evaluation_config"]["max_physics_degradation"])
    limits = record["physics_limits"]
    mass_limit = float(limits["max_relative_mass_drift"])
    operator_limit = float(limits["max_operator_mse"])

    def acceptable(value: float, baseline: float, absolute_limit: float) -> bool:
        tolerance = degradation * absolute_limit
        return value <= absolute_limit and value <= baseline * (1 + degradation) + tolerance

    return (
        acceptable(selected["mass_drift"], zero["mass_drift"], mass_limit)
        and acceptable(selected["operator_mse"], zero["operator_mse"], operator_limit)
        and selected["closure_burden"]
        <= float(record["v0_7_evaluation_config"]["max_closure_burden"])
    )


def classify_memory_sweep(
    records: list[dict[str, Any]], provenance: dict[str, Any]
) -> dict[str, Any]:
    rows = _rows(records)
    config = records[0]["memory_sweep_config"]
    evaluation = records[0]["v0_7_evaluation_config"]
    seeds = [int(value) for value in provenance["seeds"]]
    initializations = [int(value) for value in provenance["closure_initialization_seeds"]]
    histories = [int(value) for value in config["history_lengths"]]
    consistency = float(config["seed_consistency_fraction"])
    material = float(config["material_relative_gain"])
    grouped = {
        (row["seed"], row["closure_seed"], row["variant"], row["history_steps"]): row
        for row in rows
    }
    record_lookup = {
        (
            int(record["seed"]),
            _closure_seed(record),
            str(record["variant"]),
            int(record["history_length_steps"]),
        ): record
        for record in records
    }

    selected: dict[tuple[int, int], dict[str, Any]] = {}
    selected_models: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        for initialization in initializations:
            zero = grouped[(seed, initialization, "zero", 1)]
            candidates = [grouped[(seed, initialization, "linear", 1)]] + [
                grouped[(seed, initialization, variant, history)]
                for history in histories
                for variant in ("instantaneous", "history")
            ]
            best = min(candidates, key=lambda row: row["validation_residual_nrmse"])
            if best["validation_residual_nrmse"] > 1.0 - material:
                best = zero
            selected[(seed, initialization)] = best
            selected_models[f"seed_{seed}_init_{initialization}"] = {
                "variant": best["variant"],
                "history_steps": best["history_steps"],
                "validation_residual_nrmse": best["validation_residual_nrmse"],
                "test_residual_r2": best["residual_r2"],
            }

    signal_rms = statistics.mean(row["residual_target_rms"] for row in rows)
    seed_selected_r2 = {
        str(seed): statistics.median(
            selected[(seed, initialization)]["residual_r2"] for initialization in initializations
        )
        for seed in seeds
    }
    if signal_rms < float(evaluation["min_residual_rms"]):
        learnability = "NONE"
    else:
        learnability = "NONE"
        for label, threshold in (
            ("STRONG", float(config["strong_r2"])),
            ("MODERATE", float(config["moderate_r2"])),
            ("WEAK", float(config["weak_r2"])),
        ):
            if (
                _fraction([seed_selected_r2[str(seed)] >= threshold for seed in seeds])
                >= consistency
            ):
                learnability = label
                break

    seed_utility: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        outcomes: list[str] = []
        gains: list[float] = []
        physics: list[bool] = []
        for initialization in initializations:
            best = selected[(seed, initialization)]
            zero = grouped[(seed, initialization, "zero", 1)]
            gain = (zero["closed_loop_field_rmse"] - best["closed_loop_field_rmse"]) / max(
                zero["closed_loop_field_rmse"], 1e-12
            )
            record = record_lookup[(seed, initialization, best["variant"], best["history_steps"])]
            passes_physics = _physics_ok(best, zero, record)
            gains.append(gain)
            physics.append(passes_physics)
            if best["variant"] == "zero" or abs(gain) < material:
                outcomes.append("NEUTRAL")
            elif gain >= material and passes_physics:
                outcomes.append("POSITIVE")
            else:
                outcomes.append("NEGATIVE")
        positive = _fraction([value == "POSITIVE" for value in outcomes])
        negative = _fraction([value == "NEGATIVE" for value in outcomes])
        label = (
            "POSITIVE"
            if positive >= consistency
            else ("NEGATIVE" if negative >= consistency else "NEUTRAL")
        )
        seed_utility[str(seed)] = {
            "label": label,
            "median_field_gain": statistics.median(gains),
            "physics_pass_fraction": _fraction(physics),
            "repeat_outcomes": outcomes,
        }
    if _fraction([item["label"] == "POSITIVE" for item in seed_utility.values()]) >= consistency:
        utility = "POSITIVE"
    elif _fraction([item["label"] == "NEGATIVE" for item in seed_utility.values()]) >= consistency:
        utility = "NEGATIVE"
    else:
        utility = "NEUTRAL"

    aggregate: dict[int, dict[str, Any]] = {}
    memory_seed_fraction: dict[int, float] = {1: 0.0}
    scores: dict[int, float] = {}
    for history in histories:
        ordered_rows = [
            grouped[(seed, initialization, "history", history)]
            for seed in seeds
            for initialization in initializations
        ]
        instant_rows = [
            grouped[(seed, initialization, "instantaneous", history)]
            for seed in seeds
            for initialization in initializations
        ]
        shuffled_rows = (
            [
                grouped[(seed, initialization, "shuffled_history", history)]
                for seed in seeds
                for initialization in initializations
            ]
            if history > 1
            else []
        )
        aggregate[history] = {
            "physical_time": _mean(ordered_rows, "history_physical_time"),
            "validation_residual_nrmse": _mean(ordered_rows, "validation_residual_nrmse"),
            "validation_residual_nrmse_std": _stdev(ordered_rows, "validation_residual_nrmse"),
            "residual_nrmse": _mean(ordered_rows, "residual_nrmse"),
            "residual_nrmse_std": _stdev(ordered_rows, "residual_nrmse"),
            "residual_r2": _mean(ordered_rows, "residual_r2"),
            "field_rmse": _mean(ordered_rows, "closed_loop_field_rmse"),
            "field_rmse_std": _stdev(ordered_rows, "closed_loop_field_rmse"),
            "instant_residual_nrmse": _mean(instant_rows, "residual_nrmse"),
            "instant_validation_residual_nrmse": _mean(instant_rows, "validation_residual_nrmse"),
            "instant_field_rmse": _mean(instant_rows, "closed_loop_field_rmse"),
            "shuffled_residual_nrmse": (
                _mean(shuffled_rows, "residual_nrmse") if shuffled_rows else None
            ),
            "shuffled_validation_residual_nrmse": (
                _mean(shuffled_rows, "validation_residual_nrmse") if shuffled_rows else None
            ),
            "shuffled_field_rmse": (
                _mean(shuffled_rows, "closed_loop_field_rmse") if shuffled_rows else None
            ),
        }
    # Select a memory candidate using validation residuals only. The held-out test
    # split can confirm that candidate, but never chooses H or the control family.
    validation_choices: dict[tuple[int, int], tuple[int | None, float]] = {}
    for seed in seeds:
        for initialization in initializations:
            gains: dict[int, float] = {}
            for history in histories[1:]:
                ordered = grouped[(seed, initialization, "history", history)]
                instant = grouped[(seed, initialization, "instantaneous", history)]
                shuffled = grouped[(seed, initialization, "shuffled_history", history)]
                control = min(
                    instant["validation_residual_nrmse"],
                    shuffled["validation_residual_nrmse"],
                )
                gains[history] = (control - ordered["validation_residual_nrmse"]) / max(
                    control, 1e-12
                )
            chosen = max(gains, key=gains.get)
            validation_choices[(seed, initialization)] = (
                (chosen if gains[chosen] >= material else None),
                gains[chosen],
            )
    for history in histories[1:]:
        seed_passes: list[bool] = []
        for seed in seeds:
            repeat_passes: list[bool] = []
            for initialization in initializations:
                chosen, _ = validation_choices[(seed, initialization)]
                if chosen != history:
                    repeat_passes.append(False)
                    continue
                ordered = grouped[(seed, initialization, "history", history)]
                instant = grouped[(seed, initialization, "instantaneous", history)]
                shuffled = grouped[(seed, initialization, "shuffled_history", history)]
                h1 = grouped[(seed, initialization, "history", 1)]
                residual_control = min(instant["residual_nrmse"], shuffled["residual_nrmse"])
                field_control = min(
                    h1["closed_loop_field_rmse"],
                    instant["closed_loop_field_rmse"],
                    shuffled["closed_loop_field_rmse"],
                )
                residual_gain = (residual_control - ordered["residual_nrmse"]) / max(
                    residual_control, 1e-12
                )
                field_gain = (field_control - ordered["closed_loop_field_rmse"]) / max(
                    field_control, 1e-12
                )
                repeat_passes.append(residual_gain >= material and field_gain >= material)
            seed_passes.append(_fraction(repeat_passes) >= consistency)
        memory_seed_fraction[history] = _fraction(seed_passes)
    scores[1] = 0.0
    for history in histories[1:]:
        item = aggregate[history]
        control = min(
            item["instant_validation_residual_nrmse"],
            item["shuffled_validation_residual_nrmse"],
        )
        scores[history] = (control - item["validation_residual_nrmse"]) / max(control, 1e-12)
    material_histories = [
        history
        for history in histories[1:]
        if memory_seed_fraction[history] >= consistency and scores[history] >= material
    ]
    if not material_histories:
        if max(scores.values()) < material:
            memory_class = "MARKOVIAN"
            effective_h: int | None = 1
        else:
            memory_class = "INCONCLUSIVE"
            effective_h = None
    else:
        maximum = max(scores[history] for history in material_histories)
        effective_h = next(
            history
            for history in material_histories
            if scores[history] >= float(config["effective_gain_fraction"]) * maximum
        )
        marginal = scores[histories[-1]] - scores[histories[-2]]
        if marginal <= float(config["plateau_relative_gain"]):
            memory_class = "SHORT_MEMORY"
        elif memory_seed_fraction[histories[-1]] >= consistency:
            memory_class = "LONG_MEMORY_CANDIDATE"
        else:
            memory_class = "INCONCLUSIVE"
    effective_time = None if effective_h is None else aggregate[effective_h]["physical_time"]
    confidence = (
        "HIGH"
        if len(seeds) >= 3 and len(initializations) >= 2 and memory_class != "INCONCLUSIVE"
        else "LIMITED"
    )
    return {
        "schema_version": 2,
        "selection_protocol": (
            "validation_residual_nrmse_selects_model_and_memory_candidate_then_test_confirms"
        ),
        "residual_learnability": learnability,
        "closed_loop_utility": utility,
        "memory_class": memory_class,
        "effective_history_steps": effective_h,
        "effective_history_physical_time": effective_time,
        "confidence": confidence,
        "thresholds": config,
        "provenance_validation": provenance,
        "evidence": {
            "residual_signal_rms": signal_rms,
            "selected_test_r2_by_seed": seed_selected_r2,
            "selected_models": selected_models,
            "utility_by_seed": seed_utility,
            "joint_gain_score_by_history": {str(key): value for key, value in scores.items()},
            "consistent_memory_seed_fraction_by_history": {
                str(key): value for key, value in memory_seed_fraction.items()
            },
            "validation_selected_memory_candidate_by_repeat": {
                f"seed_{seed}_init_{initialization}": {
                    "history_steps": choice[0],
                    "validation_relative_gain": choice[1],
                }
                for (seed, initialization), choice in validation_choices.items()
            },
            "aggregate_by_history": {str(key): value for key, value in aggregate.items()},
        },
    }


def _write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _decision_report(classification: dict[str, Any]) -> str:
    evidence = classification["evidence"]
    lines = [
        "# V0.7 residual decision report",
        "",
        "## Residual Learnability",
        "",
        f"**{classification['residual_learnability']}**",
        "",
        "## Closed-Loop Utility",
        "",
        f"**{classification['closed_loop_utility']}**",
        "",
        "## Memory Class",
        "",
        f"**{classification['memory_class']}**",
        "",
        f"Effective history: `{classification['effective_history_steps']}` steps; physical time "
        f"`{classification['effective_history_physical_time']}`. Confidence: "
        f"`{classification['confidence']}`.",
        "",
        "Model/H selection used validation residual NRMSE only. Test metrics were read once after "
        "selection; no test-set oracle selection was used.",
        "",
        "## Selected-model evidence",
        "",
        "| backbone/data seed | selected test R2 | utility | median field gain | physics pass |",
        "|---:|---:|:---:|---:|---:|",
    ]
    for seed, r2 in evidence["selected_test_r2_by_seed"].items():
        utility = evidence["utility_by_seed"][seed]
        lines.append(
            f"| {seed} | {r2:.6g} | {utility['label']} | "
            f"{utility['median_field_gain']:.6g} | {utility['physics_pass_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## History-length sweep",
            "",
            "| H | physical time | ordered NRMSE mean +/- std | ordered R2 | field RMSE "
            "mean +/- std | instantaneous NRMSE | shuffled NRMSE | validation gain |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    aggregate = evidence["aggregate_by_history"]
    scores = evidence["joint_gain_score_by_history"]
    for history in sorted(aggregate, key=int):
        values = aggregate[history]
        shuffled = values["shuffled_residual_nrmse"]
        shuffled_text = "n/a" if shuffled is None else f"{shuffled:.6g}"
        lines.append(
            f"| {history} | {values['physical_time']:.6g} | "
            f"{values['residual_nrmse']:.6g} +/- {values['residual_nrmse_std']:.3g} | "
            f"{values['residual_r2']:.6g} | {values['field_rmse']:.6g} +/- "
            f"{values['field_rmse_std']:.3g} | {values['instant_residual_nrmse']:.6g} | "
            f"{shuffled_text} | {scores[history]:.6g} |"
        )
    lines.extend(
        [
            "",
            "Physics acceptance requires both the absolute physical limit and a baseline-relative "
            "tolerance. Closure burden must also remain below its configured bound.",
            "",
            "H=1 is a paired Markovian control. ACF remains auxiliary; finite-history evidence is "
            "not identification of an exact Mori-Zwanzig kernel.",
            "",
        ]
    )
    return "\n".join(lines)


def _route_report(classification: dict[str, Any]) -> str:
    memory = classification["memory_class"]
    utility = classification["closed_loop_utility"]
    if utility == "NEGATIVE":
        route = "Do not increase closure complexity; diagnose rollout stability and target scale."
    elif memory == "MARKOVIAN":
        route = "Prefer the minimal instantaneous closure; recurrent memory is not supported."
    elif memory == "SHORT_MEMORY":
        route = "Use the reported finite effective history as the candidate V0.8 context length."
    elif memory == "LONG_MEMORY_CANDIDATE":
        route = "Only then consider a compact recurrent or state-space closure in V0.8."
    else:
        route = "Do not authorize V0.8 memory changes; gather clearer evidence first."
    return (
        "# V0.8 route recommendation\n\n"
        f"V0.7 result: `{memory}` memory and `{utility}` closed-loop utility.\n\n"
        f"{route}\n\nThis report recommends a route only; it does not implement V0.8.\n"
    )


def _write_plots(
    records: list[dict[str, Any]],
    classification: dict[str, Any],
    root: Path,
    session_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = classification["evidence"]["aggregate_by_history"]
    histories = sorted(int(value) for value in aggregate)
    values = [aggregate[str(history)] for history in histories]

    def errorbar_plot(name: str, value_key: str, std_key: str, ylabel: str) -> None:
        plt.figure(figsize=(7, 4.5))
        plt.errorbar(
            histories,
            [item[value_key] for item in values],
            yerr=[item[std_key] for item in values],
            marker="o",
            capsize=3,
            label="ordered history mean +/- std",
        )
        plt.xlabel("history length H (steps)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(root / name, dpi=160)
        plt.close()

    errorbar_plot(
        "history_length_residual_error.png",
        "residual_nrmse",
        "residual_nrmse_std",
        "residual NRMSE",
    )
    errorbar_plot(
        "history_length_rollout_error.png",
        "field_rmse",
        "field_rmse_std",
        "closed-loop field RMSE",
    )
    valid = histories[1:]
    plt.figure(figsize=(7, 4.5))
    plt.plot(valid, [aggregate[str(h)]["residual_nrmse"] for h in valid], "o-", label="ordered")
    plt.plot(
        valid,
        [aggregate[str(h)]["shuffled_residual_nrmse"] for h in valid],
        "s--",
        label="shuffled",
    )
    plt.xlabel("history length H (steps)")
    plt.ylabel("residual NRMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "real_vs_shuffled_history.png", dpi=160)
    plt.close()

    history_records = [record for record in records if record["variant"] == "history"]
    plt.figure(figsize=(7, 4.5))
    for history in histories:
        curves = []
        for record in history_records:
            if int(record["history_length_steps"]) == history:
                _, closed = _longest(record)
                curves.append([float(value) for value in closed["closure_burden_by_step"]])
        mean_curve = [
            statistics.mean(values_at_step) for values_at_step in zip(*curves, strict=True)
        ]
        plt.plot(range(1, len(mean_curve) + 1), mean_curve, label=f"H={history}")
    plt.xlabel("closed-loop step")
    plt.ylabel("mean closure burden")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "closure_burden_along_rollout.png", dpi=160)
    plt.close()

    diagnostics = sorted(session_dir.glob("seeds/seed_*/cache/residual_diagnostics.json"))
    if diagnostics:
        plt.figure(figsize=(7, 4.5))
        for path in diagnostics:
            payload = json.loads(path.read_text(encoding="utf-8"))
            acf = payload["splits"]["test"]["acf"]
            acf_values = [
                statistics.mean(float(series[lag]) for series in acf.values())
                for lag in range(len(next(iter(acf.values()))))
            ]
            plt.plot(
                range(1, len(acf_values) + 1),
                acf_values,
                label=path.parent.parent.name,
            )
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("residual lag (steps)")
        plt.ylabel("mean latent-dimension ACF")
        plt.legend()
        plt.tight_layout()
        plt.savefig(root / "residual_autocorrelation_vs_lag.png", dpi=160)
        plt.close()


def compare_residual_memory_v0_7(session_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    records = load_evaluation_records(session_dir)
    provenance = validate_sweep_provenance(records)
    rows = _rows(records)
    classification = classify_memory_sweep(records, provenance)
    history_rows = {
        (row["seed"], row["closure_seed"], row["history_steps"]): row
        for row in rows
        if row["variant"] == "history"
    }
    histories = sorted({row["history_steps"] for row in rows if row["variant"] == "history"})
    for row in rows:
        row["H"] = row["history_steps"]
        row["physical_history_time"] = row["history_physical_time"]
        row["parameters"] = row["parameter_count"]
        row["history_shuffle"] = row["history_shuffled"]
        row["residual_MSE"] = row["residual_mse"]
        row["residual_NRMSE"] = row["residual_nrmse"]
        row["residual_R2"] = row["residual_r2"]
        row["latent_rollout_error"] = row["closed_loop_latent_rmse"]
        row["field_rollout_error"] = row["closed_loop_field_rmse"]
        row["operator_residual"] = row["operator_mse"]
        row["history_gain"] = None
        row["marginal_history_gain"] = None
        row["marginal_gain"] = None
        if row["variant"] == "history":
            key = (row["seed"], row["closure_seed"])
            baseline = history_rows[(*key, 1)]
            row["history_gain"] = (
                baseline["closed_loop_field_rmse"] - row["closed_loop_field_rmse"]
            ) / max(baseline["closed_loop_field_rmse"], 1e-12)
            index = histories.index(row["history_steps"])
            if index > 0:
                previous = history_rows[(*key, histories[index - 1])]
                row["marginal_history_gain"] = (
                    previous["closed_loop_field_rmse"] - row["closed_loop_field_rmse"]
                ) / max(previous["closed_loop_field_rmse"], 1e-12)
                row["marginal_gain"] = row["marginal_history_gain"]
        row["memory_classification"] = classification["memory_class"]
    output = Path(output_dir)
    evaluation_dir = output / "evaluation"
    plot_dir = output / "plots"
    report_dir = output / "reports"
    for directory in (evaluation_dir, plot_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, evaluation_dir / "history_sweep.csv")
    (evaluation_dir / "memory_classification.json").write_text(
        json.dumps(classification, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "residual_decision_report.md").write_text(
        _decision_report(classification), encoding="utf-8"
    )
    (report_dir / "v0_8_route_recommendation.md").write_text(
        _route_report(classification), encoding="utf-8"
    )
    _write_plots(records, classification, plot_dir, Path(session_dir))
    return classification
