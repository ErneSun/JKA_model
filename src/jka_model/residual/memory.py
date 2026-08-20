"""Problem-agnostic, trained-result-only V0.7 memory characterization."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from jka_model.residual.assessment import assess_residual_structure

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
    reference_sweep = records[0]["memory_sweep_config"]
    reference_evaluation = records[0]["v0_7_evaluation_config"]
    reference_physics = records[0]["physics_limits"]
    identities = [
        (
            int(record["seed"]),
            _closure_seed(record),
            str(record["variant"]),
            int(record["history_length_steps"]),
        )
        for record in records
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("V0.7 sweep contains duplicated run identity")
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        seed = int(record["seed"])
        closure_seed = _closure_seed(record)
        by_seed[seed].append(record)
        if int(record.get("backbone_data_seed", -1)) != seed:
            raise ValueError("V0.7 record backbone_data_seed disagrees with seed")
        if int(record.get("closure_init_seed", -1)) != closure_seed:
            raise ValueError("V0.7 record closure_init_seed provenance mismatch")
        if not record.get("run_id") or not record.get("git_commit"):
            raise ValueError("V0.7 record lacks run_id or git_commit")
        if not record.get("residual_scale_fingerprint"):
            raise ValueError("V0.7 record lacks residual scale fingerprint")
        if record["memory_sweep_config"] != reference_sweep:
            raise ValueError("V0.7 memory sweep config changed within formal matrix")
        if record["v0_7_evaluation_config"] != reference_evaluation:
            raise ValueError("V0.7 evaluation config changed within formal matrix")
        if record["physics_limits"] != reference_physics:
            raise ValueError("V0.7 physics limits changed within formal matrix")
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
    expected_histories = [
        int(value) for value in records[0]["memory_sweep_config"]["history_lengths"]
    ]
    expected_identities = {
        (seed, initialization, variant, history)
        for seed in by_seed
        for initialization in expected_initializations
        for history in expected_histories
        for variant in (
            ("zero", "linear", "history", "instantaneous")
            if history == 1
            else ("history", "instantaneous", "shuffled_history")
        )
    }
    actual_identities = set(identities)
    if actual_identities != expected_identities:
        missing = sorted(expected_identities - actual_identities)
        unexpected = sorted(actual_identities - expected_identities)
        raise ValueError(
            "V0.7 formal experiment matrix mismatch: "
            f"missing={missing[:5]!r}, unexpected={unexpected[:5]!r}"
        )
    expected_record_count = (
        len(by_seed) * len(expected_initializations) * (4 + 3 * (len(expected_histories) - 1))
    )
    configured_record_count = int(
        records[0]["v0_7_evaluation_config"].get("formal_record_count", expected_record_count)
    )
    if configured_record_count != expected_record_count:
        raise ValueError("V0.7 configured formal record count disagrees with the experiment matrix")
    if len(records) != expected_record_count:
        raise ValueError(f"V0.7 formal record count {len(records)} != {expected_record_count}")
    parameter_matches: dict[str, float] = {}
    h1_pairing: dict[str, bool] = {}
    for seed, seed_items in by_seed.items():
        reference = seed_items[0]["provenance"]
        reference_scale = seed_items[0]["residual_scale_fingerprint"]
        reference_structure = seed_items[0]["residual_structure"]
        for item in seed_items[1:]:
            for key in PROVENANCE_KEYS:
                if item["provenance"][key] != reference[key]:
                    raise ValueError(f"seed {seed} comparison provenance mismatch: {key}")
            if item["residual_scale_fingerprint"] != reference_scale:
                raise ValueError(f"seed {seed} residual scale fingerprint mismatch")
            if item["residual_structure"] != reference_structure:
                raise ValueError(f"seed {seed} closure-independent residual structure mismatch")
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
    result = {
        "seeds": sorted(by_seed),
        "closure_initialization_seeds": expected_initializations,
        "actual_record_count": len(records),
        "expected_record_count": expected_record_count,
        "no_duplicate_run_identity": True,
        "exact_expected_identity_matrix": True,
        "same_backbone_data_split_normalizer_and_trajectories": True,
        "frozen_online_residual_and_predicted_history_contract": True,
        "h1_markovian_pairing": h1_pairing,
        "parameter_match_relative_differences": parameter_matches,
    }
    return result


def _rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        horizon, closed = _longest(record)
        teacher = record["teacher_forced"]
        validation = record["teacher_forced_validation"]
        rows.append(
            {
                "seed": int(record["seed"]),
                "run_id": record["run_id"],
                "git_commit": record["git_commit"],
                "backbone_data_seed": int(record["backbone_data_seed"]),
                "closure_seed": _closure_seed(record),
                "closure_init_seed": int(record["closure_init_seed"]),
                "closure_family": record["closure_family"],
                "variant": record["variant"],
                "history_steps": int(record["history_length_steps"]),
                "history_physical_time": float(record["history_length_physical_time"]["mean"]),
                "parameter_count": int(record["parameter_count"]),
                "parameter_matched": bool(record["parameter_matched_control"]),
                "history_shuffled": bool(record["history_shuffled"]),
                "validation_residual_nrmse": float(validation["normalized_rmse"]),
                "validation_residual_r2": float(validation["r2"]),
                "validation_residual_mse": float(validation["mse"]),
                "validation_residual_normalized_mse": float(validation["standardized_mse"]),
                "residual_mse": float(teacher["mse"]),
                "residual_normalized_mse": float(teacher["standardized_mse"]),
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
                "residual_significance": float(
                    record["residual_structure"]["validation"]["residual_significance"]
                ),
                "residual_scale_fingerprint": record["residual_scale_fingerprint"],
                "data_fingerprint": record["provenance"]["data_fingerprint"],
                "split_fingerprint": record["provenance"]["split_fingerprint"],
                "normalizer_fingerprint": record["provenance"]["normalizer_fingerprint"],
                "backbone_checkpoint_fingerprint": record["provenance"][
                    "backbone_checkpoint_sha256"
                ],
                "max_relative_mass_drift": float(
                    record["physics_limits"]["max_relative_mass_drift"]
                ),
                "max_operator_mse": float(record["physics_limits"]["max_operator_mse"]),
                "max_physics_degradation": float(
                    record["v0_7_evaluation_config"]["max_physics_degradation"]
                ),
                "max_closure_burden": float(record["v0_7_evaluation_config"]["max_closure_burden"]),
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
            "instant_validation_residual_nrmse_std": _stdev(
                instant_rows, "validation_residual_nrmse"
            ),
            "instant_field_rmse": _mean(instant_rows, "closed_loop_field_rmse"),
            "shuffled_residual_nrmse": (
                _mean(shuffled_rows, "residual_nrmse") if shuffled_rows else None
            ),
            "shuffled_validation_residual_nrmse": (
                _mean(shuffled_rows, "validation_residual_nrmse") if shuffled_rows else None
            ),
            "shuffled_validation_residual_nrmse_std": (
                _stdev(shuffled_rows, "validation_residual_nrmse") if shuffled_rows else None
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
    result = {
        "schema_version": 3,
        "selection_protocol": (
            "validation_residual_nrmse_selects_model_and_memory_candidate_then_test_confirms"
        ),
        "residual_learnability": learnability,
        "closed_loop_utility": utility,
        "secondary_memory_closed_loop_utility": utility,
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
    assessment = assess_residual_structure(records, provenance, result)
    assessment_evidence = assessment.pop("evidence")
    result.update(assessment)
    result["evidence"]["residual_structure"] = assessment_evidence
    result["v0_8_recommendation"] = {
        "R1": "DIAGNOSTIC_BRANCH",
        "R2": "INSTANTANEOUS_DYNAMIC_CONTEXT",
        "R3": "TEMPORAL_DYNAMIC_CONTEXT",
        "INCONCLUSIVE": "NO_ARCHITECTURE_DECISION",
    }[result["residual_route"]]
    return result


def _write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _decision_report(classification: dict[str, Any]) -> str:
    evidence = classification["evidence"]
    route_evidence = evidence["residual_structure"]
    lines = [
        "# V0.7 final scientific decision",
        "",
        f"**RESIDUAL MAGNITUDE (DIAGNOSTIC):** `{classification['residual_magnitude']}`  ",
        f"**RESIDUAL LEARNABILITY:** `{classification['residual_learnability']}`  ",
        f"**CONDITIONAL HISTORY GAIN:** `{classification['conditional_history_gain']}`  ",
        f"**CLOSED-LOOP CLOSURE UTILITY:** `{classification['closed_loop_utility']}`  ",
        f"**MEMORY CLASS:** `{classification['memory_class']}`  ",
        f"**PHYSICS ACCEPTANCE:** `{classification['physics_acceptance']}`  ",
        f"**RESIDUAL ROUTE:** `{classification['residual_route']}`  ",
        f"**V0.8 RECOMMENDATION:** `{classification['v0_8_recommendation']}`",
        "",
        "## Validation-first evidence chain",
        "",
        f"Validation route: `{classification['validation_residual_route']}`; locked test "
        f"confirmation: `{classification['test_confirmation_pass']}`. Confidence: "
        f"`{classification['confidence']}`.",
        "",
        f"Residual magnitude reference threshold: "
        f"`{classification['thresholds']['min_residual_significance']}`. Validation "
        f"magnitude ratio by backbone seed: "
        f"`{evidence['residual_structure']['residual_magnitude_validation_by_seed']}`.",
        "",
        "Magnitude is diagnostic only: low instantaneous energy does not discard the residual "
        "or create an early-exit route, because small errors may accumulate in time.",
        "",
        "Predictability is `1 - best Markovian validation standardized MSE / zero "
        "standardized MSE`, using train-split per-dimension RMS. "
        "Closure family, H, history decision, and preliminary route use validation only; "
        "test is a locked confirmation and never selects a configuration.",
        "",
        "## Hierarchical history evidence",
        "",
        f"Locked H: `{classification['locked_history_steps']}`. Effective secondary-memory H: "
        f"`{classification['effective_history_steps']}` steps; physical time "
        f"`{classification['effective_history_physical_time']}`.",
        "",
        f"Closure-init consistency: `{classification['history_gain_consistency_closure_seed']}`.",
        "",
        f"Backbone/data consistency: `{classification['history_gain_consistency_backbone_seed']}`.",
        "",
        "## Route-locked closed-loop evidence",
        "",
        "| backbone/data seed | selected variants | utility | median field gain |",
        "|---:|:---|:---:|---:|",
    ]
    for seed, utility in route_evidence["route_utility_by_backbone_seed"].items():
        prefix = f"seed_{seed}_init_"
        variants = sorted(
            {
                f"{item['variant']}(H={item['history_steps']})"
                for key, item in route_evidence["route_selected_models"].items()
                if key.startswith(prefix)
            }
        )
        lines.append(
            f"| {seed} | {', '.join(variants)} | {utility['label']} | "
            f"{utility['median_field_gain']:.6g} |"
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
            "Physics acceptance is the logical AND of the inherited V0.5 absolute limit, "
            "zero-closure non-inferiority, and closure burden <= 0.25.",
            "",
            f"Absolute pass: `{classification['physics_absolute_pass']}`; non-inferiority pass: "
            f"`{classification['physics_noninferiority_pass']}`; burden pass: "
            f"`{classification['closure_burden_pass']}`.",
            "A failed locked test confirmation or failed physics acceptance forces the final "
            "residual route to INCONCLUSIVE.",
            "",
            "H=1 is a paired Markovian control. ACF remains auxiliary; finite-history evidence is "
            "not identification of an exact Mori-Zwanzig kernel.",
            "",
        ]
    )
    return "\n".join(lines)


def _route_report(classification: dict[str, Any]) -> str:
    route_name = classification["residual_route"]
    route = {
        "R1": (
            "Enter the diagnostic branch for data quality, missing observables/forcing, latent "
            "adequacy, stochasticity, or Koopman capacity. Do not add Attention by default."
        ),
        "R2": (
            "V0.8 may test an instantaneous dynamic-context small MLP. Attention is not "
            "scientifically justified by V0.7."
        ),
        "R3": (
            "V0.8 may test a small causal-Attention temporal context encoder using the locked "
            "history horizon."
        ),
        "INCONCLUSIVE": "Do not select a V0.8 context family; gather clearer evidence first.",
    }[route_name]
    return (
        "# V0.8 route recommendation\n\n"
        f"V0.7 residual route: `{route_name}`. Closed-loop utility: "
        f"`{classification['closed_loop_utility']}`.\n\n"
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
        "validation_residual_nrmse",
        "validation_residual_nrmse_std",
        "validation residual NRMSE",
    )
    errorbar_plot(
        "history_length_rollout_error.png",
        "field_rmse",
        "field_rmse_std",
        "closed-loop field RMSE",
    )
    valid = histories[1:]
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(
        valid,
        [aggregate[str(h)]["validation_residual_nrmse"] for h in valid],
        yerr=[aggregate[str(h)]["validation_residual_nrmse_std"] for h in valid],
        fmt="o-",
        capsize=3,
        label="ordered history",
    )
    plt.errorbar(
        valid,
        [aggregate[str(h)]["instant_validation_residual_nrmse"] for h in valid],
        yerr=[aggregate[str(h)]["instant_validation_residual_nrmse_std"] for h in valid],
        fmt="^-.",
        capsize=3,
        label="parameter-matched instantaneous",
    )
    plt.errorbar(
        valid,
        [aggregate[str(h)]["shuffled_validation_residual_nrmse"] for h in valid],
        yerr=[aggregate[str(h)]["shuffled_validation_residual_nrmse_std"] for h in valid],
        fmt="s--",
        capsize=3,
        label="shuffled history",
    )
    plt.xlabel("history length H (steps)")
    plt.ylabel("validation residual NRMSE")
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
        by_step = list(zip(*curves, strict=True))
        mean_curve = [statistics.mean(values_at_step) for values_at_step in by_step]
        std_curve = [
            statistics.stdev(values_at_step) if len(values_at_step) > 1 else 0.0
            for values_at_step in by_step
        ]
        steps = list(range(1, len(mean_curve) + 1))
        line = plt.plot(steps, mean_curve, label=f"H={history}, n={len(curves)}")[0]
        plt.fill_between(
            steps,
            [mean - std for mean, std in zip(mean_curve, std_curve, strict=True)],
            [mean + std for mean, std in zip(mean_curve, std_curve, strict=True)],
            alpha=0.12,
            color=line.get_color(),
        )
    plt.axhline(0.25, color="black", linestyle="--", linewidth=1, label="burden limit 0.25")
    plt.xlabel("closed-loop step")
    plt.ylabel("mean closure burden")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "closure_burden_along_rollout.png", dpi=160)
    plt.close()

    diagnostics = sorted(session_dir.glob("seeds/seed_*/cache/residual_diagnostics.json"))
    expected_diagnostics = len(classification["provenance_validation"]["seeds"])
    if len(diagnostics) != expected_diagnostics:
        raise ValueError(
            "V0.7 formal plots require one residual diagnostics file per backbone/data seed"
        )
    labels: list[str] = []
    rms: list[float] = []
    normalized: list[float] = []
    for path in diagnostics:
        payload = json.loads(path.read_text(encoding="utf-8"))
        labels.append(path.parent.parent.name)
        validation = payload["splits"]["validation"]
        rms.append(float(validation["rms"]))
        normalized.append(float(validation["normalized_rms_by_true_increment"]))
    x = list(range(len(labels)))
    figure, left = plt.subplots(figsize=(7, 4.5))
    right = left.twinx()
    left.bar([value - 0.18 for value in x], rms, width=0.36, label="raw residual RMS")
    right.bar(
        [value + 0.18 for value in x],
        normalized,
        width=0.36,
        color="tab:orange",
        label="RMS / true-increment RMS",
    )
    left.set_xticks(x, labels)
    left.set_ylabel("raw residual RMS")
    right.set_ylabel("normalized residual RMS")
    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    left.legend(handles_left + handles_right, labels_left + labels_right)
    figure.tight_layout()
    figure.savefig(root / "residual_magnitude.png", dpi=160)
    plt.close(figure)

    plt.figure(figsize=(7, 4.5))
    for path in diagnostics:
        payload = json.loads(path.read_text(encoding="utf-8"))
        acf = payload["splits"]["validation"]["acf"]
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

    gains = classification["evidence"]["residual_structure"]["history_gain_validation_by_repeat"]
    plt.figure(figsize=(8, 4.8))
    for seed in classification["provenance_validation"]["seeds"]:
        means: list[float] = []
        stds: list[float] = []
        for history in valid:
            values_at_h = [
                float(value)
                for key, value in gains.items()
                if key.startswith(f"seed_{seed}_") and key.endswith(f"_h{history}")
            ]
            means.append(statistics.mean(values_at_h))
            stds.append(statistics.stdev(values_at_h) if len(values_at_h) > 1 else 0.0)
        plt.errorbar(valid, means, yerr=stds, marker="o", capsize=3, label=f"seed {seed}, n=3")
    plt.axhline(
        float(classification["thresholds"]["material_history_gain"]),
        color="black",
        linestyle="--",
        label="material gain threshold",
    )
    plt.xlabel("history length H (steps)")
    plt.ylabel("validation conditional history gain")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "history_gain_by_backbone_seed.png", dpi=160)
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
    row_lookup = {
        (row["seed"], row["closure_seed"], row["variant"], row["history_steps"]): row
        for row in rows
    }
    for row in rows:
        row["H"] = row["history_steps"]
        row["physical_history_time"] = row["history_physical_time"]
        row["parameters"] = row["parameter_count"]
        row["history_shuffle"] = row["history_shuffled"]
        row["residual_MSE"] = row["residual_mse"]
        row["residual_normalized_MSE"] = row["residual_normalized_mse"]
        row["residual_NRMSE"] = row["residual_nrmse"]
        row["residual_R2"] = row["residual_r2"]
        row["latent_rollout_error"] = row["closed_loop_latent_rmse"]
        row["field_rollout_error"] = row["closed_loop_field_rmse"]
        row["operator_residual"] = row["operator_mse"]
        zero = row_lookup[(row["seed"], row["closure_seed"], "zero", 1)]
        row["physics_absolute_pass"] = (
            row["mass_drift"] <= row["max_relative_mass_drift"]
            and row["operator_mse"] <= row["max_operator_mse"]
        )
        row["physics_noninferiority_pass"] = (
            row["mass_drift"]
            <= zero["mass_drift"] * (1 + row["max_physics_degradation"])
            + row["max_physics_degradation"] * row["max_relative_mass_drift"]
            and row["operator_mse"]
            <= zero["operator_mse"] * (1 + row["max_physics_degradation"])
            + row["max_physics_degradation"] * row["max_operator_mse"]
        )
        row["closure_burden_pass"] = row["closure_burden"] <= row["max_closure_burden"]
        row["physics_pass"] = (
            row["physics_absolute_pass"]
            and row["physics_noninferiority_pass"]
            and row["closure_burden_pass"]
        )
        row["validation_history_gain"] = None
        row["test_history_gain"] = None
        row["history_gain"] = None
        row["marginal_history_gain"] = None
        row["marginal_gain"] = None
        if row["variant"] == "history":
            key = (row["seed"], row["closure_seed"])
            if row["history_steps"] == 1:
                row["validation_history_gain"] = 0.0
                row["test_history_gain"] = 0.0
                row["history_gain"] = 0.0
            else:
                instant = row_lookup[(*key, "instantaneous", row["history_steps"])]
                shuffled = row_lookup[(*key, "shuffled_history", row["history_steps"])]
                validation_control = min(
                    instant["validation_residual_normalized_mse"],
                    shuffled["validation_residual_normalized_mse"],
                )
                test_control = min(
                    instant["residual_normalized_mse"],
                    shuffled["residual_normalized_mse"],
                )
                row["validation_history_gain"] = (
                    validation_control - row["validation_residual_normalized_mse"]
                ) / max(validation_control, 1e-12)
                row["test_history_gain"] = (test_control - row["residual_normalized_mse"]) / max(
                    test_control, 1e-12
                )
                row["history_gain"] = row["validation_history_gain"]
            index = histories.index(row["history_steps"])
            if index > 0:
                previous = history_rows[(*key, histories[index - 1])]
                row["marginal_history_gain"] = (
                    previous["validation_residual_normalized_mse"]
                    - row["validation_residual_normalized_mse"]
                ) / max(previous["validation_residual_normalized_mse"], 1e-12)
                row["marginal_gain"] = row["marginal_history_gain"]
        row["memory_classification"] = classification["memory_class"]
        row["residual_route"] = classification["residual_route"]
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
    (evaluation_dir / "residual_structure_assessment.json").write_text(
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
