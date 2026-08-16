"""Trained-result-only V0.7 history sweep comparison and decision reports."""

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
    if len(by_seed) < 3:
        raise ValueError("memory characterization requires at least three seeds")
    parameter_matches: dict[str, float] = {}
    for seed, items in by_seed.items():
        reference = items[0]["provenance"]
        for item in items[1:]:
            for key in PROVENANCE_KEYS:
                if item["provenance"][key] != reference[key]:
                    raise ValueError(f"seed {seed} comparison provenance mismatch: {key}")
        histories = sorted(
            int(item["history_length_steps"]) for item in items if item["variant"] == "history"
        )
        expected = list(items[0]["memory_sweep_config"]["history_lengths"])
        if histories != expected:
            raise ValueError(f"seed {seed} history sweep incomplete: {histories} != {expected}")
        lookup = {(item["variant"], int(item["history_length_steps"])): item for item in items}
        for required in (("zero", 1), ("linear", 1)):
            if required not in lookup:
                raise ValueError(f"seed {seed} is missing required control {required}")
        tolerance = float(items[0]["memory_sweep_config"]["parameter_match_tolerance"])
        for history in histories:
            for variant in ("instantaneous", "history"):
                if (variant, history) not in lookup:
                    raise ValueError(f"seed {seed} is missing {variant} H={history}")
            if history > 1 and ("shuffled_history", history) not in lookup:
                raise ValueError(f"seed {seed} is missing shuffled-history H={history}")
            history_count = int(lookup[("history", history)]["parameter_count"])
            instant_count = int(lookup[("instantaneous", history)]["parameter_count"])
            relative = abs(history_count - instant_count) / max(history_count, 1)
            parameter_matches[f"seed_{seed}_h{history}"] = relative
            if relative > tolerance:
                raise ValueError(
                    f"seed {seed} H={history} parameter mismatch {relative:.3%} > {tolerance:.3%}"
                )
    return {
        "seeds": sorted(by_seed),
        "same_backbone_data_split_normalizer_and_trajectories": True,
        "frozen_online_residual_and_predicted_history_contract": True,
        "parameter_match_relative_differences": parameter_matches,
    }


def _rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        horizon, closed = _longest(record)
        teacher = record["teacher_forced"]
        rows.append(
            {
                "seed": int(record["seed"]),
                "closure_family": record["closure_family"],
                "variant": record["variant"],
                "history_steps": int(record["history_length_steps"]),
                "history_physical_time": float(record["history_length_physical_time"]["mean"]),
                "parameter_count": int(record["parameter_count"]),
                "parameter_matched": bool(record["parameter_matched_control"]),
                "history_shuffled": bool(record["history_shuffled"]),
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
    return sorted(rows, key=lambda row: (row["seed"], row["history_steps"], row["variant"]))


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.mean(float(row[field]) for row in rows)


def classify_memory_sweep(
    records: list[dict[str, Any]], provenance: dict[str, Any]
) -> dict[str, Any]:
    rows = _rows(records)
    config = records[0]["memory_sweep_config"]
    evaluation = records[0]["v0_7_evaluation_config"]
    seeds = provenance["seeds"]
    histories = [int(value) for value in config["history_lengths"]]
    grouped = {(row["seed"], row["variant"], row["history_steps"]): row for row in rows}

    best_r2 = max(
        row["residual_r2"] for row in rows if row["variant"] not in {"zero", "shuffled_history"}
    )
    signal_rms = statistics.mean(row["residual_target_rms"] for row in rows)
    if signal_rms < float(evaluation["min_residual_rms"]):
        learnability = "NONE"
    elif best_r2 >= float(config["strong_r2"]):
        learnability = "STRONG"
    elif best_r2 >= float(config["moderate_r2"]):
        learnability = "MODERATE"
    elif best_r2 >= float(config["weak_r2"]):
        learnability = "WEAK"
    else:
        learnability = "NONE"

    material = float(config["material_relative_gain"])
    seed_fraction = float(config["seed_consistency_fraction"])
    utility_gains: list[float] = []
    physics_ok: list[bool] = []
    per_seed_best_h: dict[str, int] = {}
    per_seed_best_model: dict[str, str] = {}
    for seed in seeds:
        zero = grouped[(seed, "zero", 1)]
        candidates = [grouped[(seed, "linear", 1)]] + [
            grouped[(seed, variant, history)]
            for history in histories
            for variant in ("instantaneous", "history")
        ]
        best = min(candidates, key=lambda row: row["closed_loop_field_rmse"])
        per_seed_best_h[str(seed)] = int(best["history_steps"])
        per_seed_best_model[str(seed)] = str(best["variant"])
        utility_gains.append(
            (zero["closed_loop_field_rmse"] - best["closed_loop_field_rmse"])
            / max(zero["closed_loop_field_rmse"], 1e-12)
        )
        allowed = 1.0 + float(evaluation["max_physics_degradation"])
        physics_ok.append(
            best["mass_drift"] <= max(zero["mass_drift"] * allowed, 0.01)
            and best["operator_mse"] <= max(zero["operator_mse"] * allowed, 1e-4)
        )
    positive_fraction = sum(gain >= material for gain in utility_gains) / len(seeds)
    negative_fraction = sum(gain <= -material for gain in utility_gains) / len(seeds)
    if positive_fraction >= seed_fraction and all(physics_ok):
        utility = "POSITIVE"
    elif negative_fraction >= seed_fraction or not all(physics_ok):
        utility = "NEGATIVE"
    else:
        utility = "NEUTRAL"

    aggregate: dict[int, dict[str, Any]] = {}
    for history in histories:
        history_rows = [grouped[(seed, "history", history)] for seed in seeds]
        instant_rows = [grouped[(seed, "instantaneous", history)] for seed in seeds]
        shuffled_rows = (
            [grouped[(seed, "shuffled_history", history)] for seed in seeds] if history > 1 else []
        )
        aggregate[history] = {
            "physical_time": _mean(history_rows, "history_physical_time"),
            "residual_nrmse": _mean(history_rows, "residual_nrmse"),
            "residual_r2": _mean(history_rows, "residual_r2"),
            "field_rmse": _mean(history_rows, "closed_loop_field_rmse"),
            "instant_residual_nrmse": _mean(instant_rows, "residual_nrmse"),
            "instant_field_rmse": _mean(instant_rows, "closed_loop_field_rmse"),
            "shuffled_residual_nrmse": (
                _mean(shuffled_rows, "residual_nrmse") if shuffled_rows else None
            ),
            "shuffled_field_rmse": (
                _mean(shuffled_rows, "closed_loop_field_rmse") if shuffled_rows else None
            ),
        }
    baseline = aggregate[1]
    scores: dict[int, float] = {}
    consistent_gains: dict[int, float] = {}
    for history in histories:
        item = aggregate[history]
        residual_gain = (baseline["residual_nrmse"] - item["residual_nrmse"]) / max(
            baseline["residual_nrmse"], 1e-12
        )
        rollout_gain = (baseline["field_rmse"] - item["field_rmse"]) / max(
            baseline["field_rmse"], 1e-12
        )
        scores[history] = 0.5 * (residual_gain + rollout_gain)
        if history == 1:
            consistent_gains[history] = 0.0
            continue
        passes = 0
        for seed in seeds:
            current = grouped[(seed, "history", history)]
            instant = grouped[(seed, "instantaneous", history)]
            shuffled = grouped[(seed, "shuffled_history", history)]
            residual_control = min(instant["residual_nrmse"], shuffled["residual_nrmse"])
            field_control = min(
                grouped[(seed, "history", 1)]["closed_loop_field_rmse"],
                instant["closed_loop_field_rmse"],
                shuffled["closed_loop_field_rmse"],
            )
            residual_gain = (residual_control - current["residual_nrmse"]) / max(
                residual_control, 1e-12
            )
            rollout_gain = (field_control - current["closed_loop_field_rmse"]) / max(
                field_control, 1e-12
            )
            passes += residual_gain >= material and rollout_gain >= material
        consistent_gains[history] = passes / len(seeds)
    material_histories = [
        history
        for history in histories[1:]
        if consistent_gains[history] >= seed_fraction and scores[history] >= material
    ]
    if not material_histories:
        memory_class = "MARKOVIAN" if max(scores.values()) < material else "INCONCLUSIVE"
        effective_h: int | None = 1 if memory_class == "MARKOVIAN" else None
    else:
        maximum = max(scores[history] for history in material_histories)
        threshold = float(config["effective_gain_fraction"]) * maximum
        effective_h = next(
            history for history in material_histories if scores[history] >= threshold
        )
        last, previous = histories[-1], histories[-2]
        marginal = scores[last] - scores[previous]
        if marginal <= float(config["plateau_relative_gain"]):
            memory_class = "SHORT_MEMORY"
        elif consistent_gains[last] >= seed_fraction:
            memory_class = "LONG_MEMORY_CANDIDATE"
        else:
            memory_class = "INCONCLUSIVE"
    effective_time = None if effective_h is None else aggregate[effective_h]["physical_time"]
    confidence = "HIGH" if len(seeds) >= 3 and memory_class != "INCONCLUSIVE" else "LIMITED"
    control_summary = {
        variant: {
            "residual_nrmse": _mean(
                [grouped[(seed, variant, 1)] for seed in seeds], "residual_nrmse"
            ),
            "residual_r2": _mean([grouped[(seed, variant, 1)] for seed in seeds], "residual_r2"),
            "field_rmse": _mean(
                [grouped[(seed, variant, 1)] for seed in seeds], "closed_loop_field_rmse"
            ),
        }
        for variant in ("zero", "linear")
    }
    return {
        "schema_version": 1,
        "residual_learnability": learnability,
        "closed_loop_utility": utility,
        "memory_class": memory_class,
        "effective_history_steps": effective_h,
        "effective_history_physical_time": effective_time,
        "confidence": confidence,
        "thresholds": config,
        "provenance_validation": provenance,
        "evidence": {
            "best_residual_r2": best_r2,
            "residual_signal_rms": signal_rms,
            "closed_loop_relative_gains_by_seed": dict(
                zip(map(str, seeds), utility_gains, strict=True)
            ),
            "physics_noninferiority_by_seed": dict(zip(map(str, seeds), physics_ok, strict=True)),
            "best_history_by_seed": per_seed_best_h,
            "best_closure_variant_by_seed": per_seed_best_model,
            "joint_gain_score_by_history": {str(key): value for key, value in scores.items()},
            "consistent_material_gain_fraction_by_history": {
                str(key): value for key, value in consistent_gains.items()
            },
            "aggregate_by_history": {str(key): value for key, value in aggregate.items()},
            "control_summary": control_summary,
        },
    }


def _write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    fields = list(rows[0])
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
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
        "## Residual statistics and minimal baselines",
        "",
        f"Residual signal RMS: `{evidence['residual_signal_rms']:.6g}`; best held-out R2: "
        f"`{evidence['best_residual_r2']:.6g}`.",
        "",
        "| model | residual NRMSE | residual R2 | longest field RMSE |",
        "|---|---:|---:|---:|",
    ]
    for variant, values in evidence["control_summary"].items():
        lines.append(
            f"| {variant} | {values['residual_nrmse']:.6g} | {values['residual_r2']:.6g} | "
            f"{values['field_rmse']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## History-length sweep and controls",
            "",
            "| H | physical time | ordered NRMSE | ordered R2 | ordered field RMSE | "
            "instantaneous NRMSE | shuffled NRMSE | joint gain |",
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
            f"{values['residual_nrmse']:.6g} | {values['residual_r2']:.6g} | "
            f"{values['field_rmse']:.6g} | {values['instant_residual_nrmse']:.6g} | "
            f"{shuffled_text} | {scores[history]:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "H=1 is the operational Markovian baseline: current latent, next dt, and parameters. "
            "H>1 is accepted as memory evidence only when it beats parameter-matched "
            "instantaneous and shuffled-history controls in teacher-forced and closed-loop tests.",
            "",
            "Physics metrics are evaluation-only non-inferiority gates, not trained losses. "
            "Autocorrelation is auxiliary and does not determine the memory class. This is a "
            "finite-history closure diagnosis, not identification of an exact Mori-Zwanzig kernel.",
            "",
        ]
    )
    return "\n".join(lines)


def _route_report(classification: dict[str, Any]) -> str:
    memory = classification["memory_class"]
    utility = classification["closed_loop_utility"]
    if utility == "NEGATIVE":
        route = "Do not increase closure complexity; diagnose rollout instability and target scale."
    elif memory == "MARKOVIAN":
        route = "Prefer the minimal instantaneous closure; recurrent memory is not supported."
    elif memory == "SHORT_MEMORY":
        route = "Use the reported finite effective history as the candidate V0.8 context length."
    elif memory == "LONG_MEMORY_CANDIDATE":
        route = "Only then consider a compact recurrent or state-space closure in V0.8."
    else:
        route = "Do not authorize V0.8 memory architecture changes; gather clearer evidence first."
    return "\n".join(
        [
            "# V0.8 route recommendation",
            "",
            f"V0.7 result: `{memory}` memory and `{utility}` closed-loop utility.",
            "",
            route,
            "",
            "This report recommends a route only; it does not implement V0.8.",
            "",
        ]
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

    def line(name: str, y_history: str, y_control: str | None, ylabel: str) -> None:
        plt.figure(figsize=(7, 4.5))
        plt.plot(histories, [item[y_history] for item in values], "o-", label="ordered history")
        if y_control:
            plt.plot(histories, [item[y_control] for item in values], "s--", label="instantaneous")
        plt.xlabel("history length H (steps)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(root / name, dpi=160)
        plt.close()

    line(
        "history_length_residual_error.png",
        "residual_nrmse",
        "instant_residual_nrmse",
        "residual NRMSE",
    )
    line(
        "history_length_rollout_error.png",
        "field_rmse",
        "instant_field_rmse",
        "closed-loop field RMSE",
    )
    plt.figure(figsize=(7, 4.5))
    valid = histories[1:]
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

    history_records = [item for item in records if item["variant"] == "history"]
    plt.figure(figsize=(7, 4.5))
    for record in history_records:
        horizon, closed = _longest(record)
        plt.plot(
            range(1, horizon + 1),
            closed["closure_burden_by_step"],
            alpha=0.35,
            label=f"seed {record['seed']} H={record['history_length_steps']}",
        )
    plt.xlabel("closed-loop step")
    plt.ylabel("closure burden")
    plt.tight_layout()
    plt.savefig(root / "closure_burden_along_rollout.png", dpi=160)
    plt.close()

    diagnostics = sorted(session_dir.glob("seeds/seed_*/cache/residual_diagnostics.json"))
    if diagnostics:
        plt.figure(figsize=(7, 4.5))
        for path in diagnostics:
            payload = json.loads(path.read_text(encoding="utf-8"))
            acf = payload["splits"]["test"]["acf"]
            values = [
                statistics.mean(float(series[lag]) for series in acf.values())
                for lag in range(len(next(iter(acf.values()))))
            ]
            plt.plot(range(1, len(values) + 1), values, alpha=0.6, label=path.parts[-4])
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
        (row["seed"], row["history_steps"]): row for row in rows if row["variant"] == "history"
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
            baseline = history_rows[(row["seed"], 1)]
            row["history_gain"] = (
                baseline["closed_loop_field_rmse"] - row["closed_loop_field_rmse"]
            ) / max(baseline["closed_loop_field_rmse"], 1e-12)
            index = histories.index(row["history_steps"])
            if index > 0:
                previous = history_rows[(row["seed"], histories[index - 1])]
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
