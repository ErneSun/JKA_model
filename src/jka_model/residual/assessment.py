"""Problem-agnostic V0.7 residual structure assessment and R1/R2/R3 routing."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


def _closure_seed(record: dict[str, Any]) -> int:
    return int(record["closure_initialization_seed"])


def _identity(record: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(record["seed"]),
        _closure_seed(record),
        str(record["variant"]),
        int(record["history_length_steps"]),
    )


def _fraction(values: list[bool]) -> float:
    return sum(values) / len(values)


def _longest(record: dict[str, Any]) -> dict[str, Any]:
    horizon = max(int(value) for value in record["closed_loop"])
    return record["closed_loop"][str(horizon)]


def _label_predictability(
    by_seed: dict[int, float], thresholds: dict[str, Any], consistency: float
) -> str:
    for label, key in (
        ("STRONG", "strong_r2"),
        ("MODERATE", "moderate_r2"),
        ("WEAK", "weak_r2"),
    ):
        threshold = float(thresholds[key])
        if _fraction([value >= threshold for value in by_seed.values()]) >= consistency:
            return label
    return "NONE"


def _label_magnitude(by_seed: dict[int, float], threshold: float, consistency: float) -> str:
    material = _fraction([value >= threshold for value in by_seed.values()])
    low = _fraction([value < threshold for value in by_seed.values()])
    if material >= consistency:
        return "MATERIAL_MAGNITUDE"
    if low >= consistency:
        return "LOW_MAGNITUDE"
    return "INCONCLUSIVE"


def _route(learnability: str, history_gain: str) -> str:
    if learnability in {"WEAK", "NONE"}:
        return "R1"
    if learnability in {"STRONG", "MODERATE"}:
        if history_gain == "ABSENT":
            return "R2"
        if history_gain == "PRESENT":
            return "R3"
    return "INCONCLUSIVE"


def _physics_checks(
    selected: dict[str, Any], zero: dict[str, Any], record: dict[str, Any]
) -> dict[str, bool]:
    degradation = float(record["v0_7_evaluation_config"]["max_physics_degradation"])
    burden_limit = float(record["v0_7_evaluation_config"]["max_closure_burden"])
    mass_limit = float(record["physics_limits"]["max_relative_mass_drift"])
    operator_limit = float(record["physics_limits"]["max_operator_mse"])
    absolute = (
        float(selected["mass_drift"]) <= mass_limit
        and float(selected["operator_mse"]) <= operator_limit
    )

    def noninferior(value: float, baseline: float, limit: float) -> bool:
        return value <= baseline * (1 + degradation) + degradation * limit

    noninferiority = noninferior(
        float(selected["mass_drift"]), float(zero["mass_drift"]), mass_limit
    ) and noninferior(float(selected["operator_mse"]), float(zero["operator_mse"]), operator_limit)
    burden = float(selected["closure_burden"]) <= burden_limit
    return {
        "absolute": absolute,
        "noninferiority": noninferiority,
        "burden": burden,
        "all": absolute and noninferiority and burden,
    }


def assess_residual_structure(
    records: list[dict[str, Any]],
    provenance: dict[str, Any],
    memory_classification: dict[str, Any],
) -> dict[str, Any]:
    """Apply validation-first nested-seed gates, then confirm the locked route on test."""
    grouped = {_identity(record): record for record in records}
    seeds = [int(value) for value in provenance["seeds"]]
    initializations = [int(value) for value in provenance["closure_initialization_seeds"]]
    sweep = records[0]["memory_sweep_config"]
    evaluation = records[0]["v0_7_evaluation_config"]
    histories = [int(value) for value in sweep["history_lengths"]]
    consistency = float(sweep["seed_consistency_fraction"])
    material = float(sweep["material_relative_gain"])

    magnitude_validation_by_seed: dict[int, float] = {}
    magnitude_test_by_seed: dict[int, float] = {}
    for seed in seeds:
        reference = grouped[(seed, initializations[0], "zero", 1)]["residual_structure"]
        magnitude_validation_by_seed[seed] = float(reference["validation"]["residual_significance"])
        magnitude_test_by_seed[seed] = float(reference["test"]["residual_significance"])
    magnitude_threshold = float(evaluation["min_residual_significance"])
    magnitude = _label_magnitude(magnitude_validation_by_seed, magnitude_threshold, consistency)
    magnitude_test = _label_magnitude(magnitude_test_by_seed, magnitude_threshold, consistency)

    markov_selection: dict[tuple[int, int], dict[str, Any]] = {}
    predictability_validation: dict[tuple[int, int], float] = {}
    predictability_test: dict[tuple[int, int], float] = {}
    for seed in seeds:
        for initialization in initializations:
            zero = grouped[(seed, initialization, "zero", 1)]
            candidates = [
                grouped[(seed, initialization, "linear", 1)],
                grouped[(seed, initialization, "instantaneous", 1)],
            ]
            best = min(
                candidates,
                key=lambda item: item["teacher_forced_validation"]["standardized_mse"],
            )
            markov_selection[(seed, initialization)] = best
            zero_validation = float(zero["teacher_forced_validation"]["standardized_mse"])
            best_validation = float(best["teacher_forced_validation"]["standardized_mse"])
            predictability_validation[(seed, initialization)] = 1.0 - best_validation / max(
                zero_validation, 1e-12
            )
            zero_test = float(zero["teacher_forced"]["standardized_mse"])
            best_test = float(best["teacher_forced"]["standardized_mse"])
            predictability_test[(seed, initialization)] = 1.0 - best_test / max(zero_test, 1e-12)
    validation_predictability_by_seed = {
        seed: statistics.median(
            predictability_validation[(seed, initialization)] for initialization in initializations
        )
        for seed in seeds
    }
    test_predictability_by_seed = {
        seed: statistics.median(
            predictability_test[(seed, initialization)] for initialization in initializations
        )
        for seed in seeds
    }
    learnability = _label_predictability(validation_predictability_by_seed, sweep, consistency)
    test_learnability = _label_predictability(test_predictability_by_seed, sweep, consistency)

    history_gain_by_repeat: dict[tuple[int, int, int], float] = {}
    closure_consistency: dict[int, dict[int, float]] = defaultdict(dict)
    backbone_labels: dict[int, str] = {}
    for seed in seeds:
        for history in histories[1:]:
            gains: list[float] = []
            for initialization in initializations:
                ordered = grouped[(seed, initialization, "history", history)]
                instant = grouped[(seed, initialization, "instantaneous", history)]
                shuffled = grouped[(seed, initialization, "shuffled_history", history)]
                control = min(
                    float(instant["teacher_forced_validation"]["standardized_mse"]),
                    float(shuffled["teacher_forced_validation"]["standardized_mse"]),
                )
                gain = (
                    control - float(ordered["teacher_forced_validation"]["standardized_mse"])
                ) / max(control, 1e-12)
                history_gain_by_repeat[(seed, initialization, history)] = gain
                gains.append(gain)
            closure_consistency[seed][history] = _fraction([gain >= material for gain in gains])
        if any(value >= consistency for value in closure_consistency[seed].values()):
            backbone_labels[seed] = "PRESENT"
        elif (
            _fraction(
                [
                    max(
                        history_gain_by_repeat[(seed, initialization, history)]
                        for history in histories[1:]
                    )
                    < material
                    for initialization in initializations
                ]
            )
            >= consistency
        ):
            backbone_labels[seed] = "ABSENT"
        else:
            backbone_labels[seed] = "INCONCLUSIVE"
    backbone_support = {
        history: _fraction([closure_consistency[seed][history] >= consistency for seed in seeds])
        for history in histories[1:]
    }
    locked_histories = [
        history for history, support in backbone_support.items() if support >= consistency
    ]
    locked_history = (
        max(
            locked_histories,
            key=lambda history: statistics.median(
                history_gain_by_repeat[(seed, initialization, history)]
                for seed in seeds
                for initialization in initializations
            ),
        )
        if locked_histories
        else None
    )
    if locked_history is not None:
        history_gain = "PRESENT"
    elif _fraction([label == "ABSENT" for label in backbone_labels.values()]) >= consistency:
        history_gain = "ABSENT"
    else:
        history_gain = "INCONCLUSIVE"

    preliminary_route = _route(learnability, history_gain)
    magnitude_test_consistent = magnitude_test == magnitude
    route_confirmed = preliminary_route != "INCONCLUSIVE"
    if preliminary_route in {"R2", "R3"}:
        route_confirmed = route_confirmed and test_learnability in {"STRONG", "MODERATE"}
    elif preliminary_route == "R1":
        route_confirmed = route_confirmed and test_learnability in {"WEAK", "NONE"}

    history_test_confirmation: dict[str, Any] | None = None
    if preliminary_route == "R3" and locked_history is not None:
        residual_by_seed: dict[int, float] = {}
        closed_loop_by_seed: dict[int, float] = {}
        for seed in seeds:
            repeat_residual: list[bool] = []
            repeat_closed_loop: list[bool] = []
            for initialization in initializations:
                ordered = grouped[(seed, initialization, "history", locked_history)]
                instant = grouped[(seed, initialization, "instantaneous", locked_history)]
                shuffled = grouped[(seed, initialization, "shuffled_history", locked_history)]
                residual_control = min(
                    float(instant["teacher_forced"]["standardized_mse"]),
                    float(shuffled["teacher_forced"]["standardized_mse"]),
                )
                residual_gain = (
                    residual_control - float(ordered["teacher_forced"]["standardized_mse"])
                ) / max(residual_control, 1e-12)
                field_control = min(
                    float(_longest(instant)["field_rmse"]),
                    float(_longest(shuffled)["field_rmse"]),
                )
                field_gain = (field_control - float(_longest(ordered)["field_rmse"])) / max(
                    field_control, 1e-12
                )
                repeat_residual.append(residual_gain >= material)
                repeat_closed_loop.append(field_gain >= material)
            residual_by_seed[seed] = _fraction(repeat_residual)
            closed_loop_by_seed[seed] = _fraction(repeat_closed_loop)
        residual_fraction = _fraction([value >= consistency for value in residual_by_seed.values()])
        closed_loop_fraction = _fraction(
            [value >= consistency for value in closed_loop_by_seed.values()]
        )
        history_test_confirmation = {
            "locked_history_steps": locked_history,
            "residual_gain_pass_fraction": residual_fraction,
            "closed_loop_gain_pass_fraction": closed_loop_fraction,
            "residual_gain_closure_fraction_by_backbone_seed": residual_by_seed,
            "closed_loop_gain_closure_fraction_by_backbone_seed": closed_loop_by_seed,
        }
        route_confirmed = route_confirmed and residual_fraction >= consistency

    if preliminary_route == "R2":
        absent_by_seed: dict[int, bool] = {}
        for seed in seeds:
            absent_repeats: list[bool] = []
            for initialization in initializations:
                gains: list[float] = []
                for history in histories[1:]:
                    ordered = grouped[(seed, initialization, "history", history)]
                    instant = grouped[(seed, initialization, "instantaneous", history)]
                    shuffled = grouped[(seed, initialization, "shuffled_history", history)]
                    control = min(
                        float(instant["teacher_forced"]["standardized_mse"]),
                        float(shuffled["teacher_forced"]["standardized_mse"]),
                    )
                    gains.append(
                        (control - float(ordered["teacher_forced"]["standardized_mse"]))
                        / max(control, 1e-12)
                    )
                absent_repeats.append(max(gains) < material)
            absent_by_seed[seed] = _fraction(absent_repeats) >= consistency
        absent_fraction = _fraction(list(absent_by_seed.values()))
        history_test_confirmation = {
            "locked_history_decision": "ABSENT",
            "absence_confirmation_fraction": absent_fraction,
            "absence_by_backbone_seed": absent_by_seed,
        }
        route_confirmed = route_confirmed and absent_fraction >= consistency

    physics_by_repeat: dict[str, dict[str, bool]] = {}
    route_selected_models: dict[str, dict[str, Any]] = {}
    route_field_gain_by_repeat: dict[str, float] = {}
    for seed in seeds:
        for initialization in initializations:
            key = f"seed_{seed}_init_{initialization}"
            zero_record = grouped[(seed, initialization, "zero", 1)]
            if preliminary_route == "R2":
                selected_record = markov_selection[(seed, initialization)]
            elif preliminary_route == "R3" and locked_history is not None:
                selected_record = grouped[(seed, initialization, "history", locked_history)]
            else:
                selected_record = zero_record
            route_selected_models[key] = {
                "variant": selected_record["variant"],
                "history_steps": selected_record["history_length_steps"],
                "selected_on": "validation_standardized_mse_and_locked_route",
            }
            physics_by_repeat[key] = _physics_checks(
                _longest(selected_record), _longest(zero_record), selected_record
            )
            selected_field = float(_longest(selected_record)["field_rmse"])
            zero_field = float(_longest(zero_record)["field_rmse"])
            route_field_gain_by_repeat[key] = (zero_field - selected_field) / max(zero_field, 1e-12)
    physics_absolute = all(item["absolute"] for item in physics_by_repeat.values())
    physics_noninferiority = all(item["noninferiority"] for item in physics_by_repeat.values())
    burden_pass = all(item["burden"] for item in physics_by_repeat.values())
    physics_acceptance = physics_absolute and physics_noninferiority and burden_pass
    route_utility_by_seed: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        outcomes: list[str] = []
        gains: list[float] = []
        for initialization in initializations:
            key = f"seed_{seed}_init_{initialization}"
            gain = route_field_gain_by_repeat[key]
            gains.append(gain)
            if route_selected_models[key]["variant"] == "zero" or abs(gain) < material:
                outcomes.append("NEUTRAL")
            elif gain >= material and physics_by_repeat[key]["all"]:
                outcomes.append("POSITIVE")
            else:
                outcomes.append("NEGATIVE")
        positive = _fraction([outcome == "POSITIVE" for outcome in outcomes])
        negative = _fraction([outcome == "NEGATIVE" for outcome in outcomes])
        label = (
            "POSITIVE"
            if positive >= consistency
            else ("NEGATIVE" if negative >= consistency else "NEUTRAL")
        )
        route_utility_by_seed[str(seed)] = {
            "label": label,
            "median_field_gain": statistics.median(gains),
            "repeat_outcomes": outcomes,
        }
    if (
        _fraction([item["label"] == "POSITIVE" for item in route_utility_by_seed.values()])
        >= consistency
    ):
        closed_loop_utility = "POSITIVE"
    elif (
        _fraction([item["label"] == "NEGATIVE" for item in route_utility_by_seed.values()])
        >= consistency
    ):
        closed_loop_utility = "NEGATIVE"
    else:
        closed_loop_utility = "NEUTRAL"
    final_route = preliminary_route if route_confirmed and physics_acceptance else "INCONCLUSIVE"

    confidence = (
        "HIGH"
        if final_route != "INCONCLUSIVE" and route_confirmed and len(seeds) >= 3
        else "LIMITED"
    )
    return {
        "assessment_schema_version": 2,
        "assessment_protocol": (
            "all_residuals_retained_validation_structure_route_then_locked_test_confirmation"
        ),
        "routing_policy": "all_nominal_residuals_enter_R1_R2_R3_structure_assessment",
        "residual_magnitude": magnitude,
        "residual_magnitude_test_consistent": magnitude_test_consistent,
        "residual_predictability": {
            "validation_by_backbone_seed": {
                str(key): value for key, value in validation_predictability_by_seed.items()
            },
            "test_confirmation_by_backbone_seed": {
                str(key): value for key, value in test_predictability_by_seed.items()
            },
            "definition": "1-best_markovian_validation_standardized_mse/zero_standardized_mse",
        },
        "residual_learnability": learnability,
        "conditional_history_gain": history_gain,
        "history_gain": history_gain,
        "history_gain_consistency_closure_seed": {
            str(seed): {str(history): value for history, value in values.items()}
            for seed, values in closure_consistency.items()
        },
        "history_gain_consistency_backbone_seed": {
            "labels": {str(key): value for key, value in backbone_labels.items()},
            "support_fraction_by_history": {
                str(key): value for key, value in backbone_support.items()
            },
        },
        "locked_history_steps": locked_history,
        "memory_class": memory_classification["memory_class"],
        "effective_history_steps": memory_classification["effective_history_steps"],
        "effective_history_physical_time": memory_classification["effective_history_physical_time"],
        "closed_loop_utility": closed_loop_utility,
        "physics_absolute_pass": physics_absolute,
        "physics_noninferiority_pass": physics_noninferiority,
        "closure_burden_pass": burden_pass,
        "physics_acceptance": "PASS" if physics_acceptance else "FAIL",
        "validation_residual_route": preliminary_route,
        "test_confirmation_pass": route_confirmed,
        "residual_route": final_route,
        "confidence": confidence,
        "thresholds": {
            "min_residual_significance": magnitude_threshold,
            "residual_significance_role": "diagnostic_only_not_a_routing_gate",
            "material_history_gain": material,
            "seed_consistency_fraction": consistency,
            "max_closure_burden": float(evaluation["max_closure_burden"]),
        },
        "evidence": {
            "residual_magnitude_validation_by_seed": {
                str(key): value for key, value in magnitude_validation_by_seed.items()
            },
            "residual_magnitude_test_by_seed": {
                str(key): value for key, value in magnitude_test_by_seed.items()
            },
            "markovian_validation_selection": {
                f"seed_{seed}_init_{initialization}": {
                    "variant": record["variant"],
                    "history_steps": record["history_length_steps"],
                    "validation_predictability": predictability_validation[(seed, initialization)],
                    "test_predictability": predictability_test[(seed, initialization)],
                }
                for (seed, initialization), record in markov_selection.items()
            },
            "history_gain_validation_by_repeat": {
                f"seed_{seed}_init_{initialization}_h{history}": value
                for (seed, initialization, history), value in history_gain_by_repeat.items()
            },
            "history_test_confirmation": history_test_confirmation,
            "route_selected_models": route_selected_models,
            "route_field_gain_by_repeat": route_field_gain_by_repeat,
            "route_utility_by_backbone_seed": route_utility_by_seed,
            "physics_by_selected_repeat": physics_by_repeat,
        },
    }
