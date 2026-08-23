from __future__ import annotations

import csv
import json
from pathlib import Path

from gpu_validation.v0_9.scripts.gpu_validate_all import (
    dirty_source_paths,
    select_validation_rank,
    strict_v0_8_handoff_fields_present,
    validate_completion_payload,
)
from jka_model.adaptive import aggregate_v0_9_results, audit_v0_8_handoff
from jka_model.config import load_config
from jka_model.evaluation import (
    GateStatus,
    MetricDirection,
    MetricGateSpec,
    aggregate_gate_results,
    evaluate_metric_gate,
)
from jka_model.utils import create_versioned_session


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_v09_config_and_revision_id_contract(tmp_path: Path) -> None:
    config = load_config("gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml")
    assert config.project_version == "0.9.0"
    assert config.training.stage.value == "adaptive"
    assert config.cylinder_wake_2d is not None and config.cylinder_wake_2d.time_varying_boundary
    assert config.v0_9_adaptive is not None
    assert config.v0_9_adaptive.rank == 8
    assert config.v0_9_adaptive.rank_candidates == (2, 4, 8, 12)
    assert config.v0_9_adaptive.bounded_coordinates
    assert config.v0_9_adaptive.trust_gate
    assert config.v0_9_training is not None
    assert config.v0_9_training.rollout_horizons == (4, 8, 16, 32, 80)
    assert config.v0_9_training.observable_horizons == (4, 8, 16, 32, 80)
    assert config.v0_9_training.phase1_enabled
    assert config.v0_9_training.gradient_conflict_method == "pcgrad"
    assert sum(config.v0_9_training.observable_horizon_probabilities) == 1.0
    assert config.v0_9_training.observable_names[-2:] == ("lift", "drag")
    assert config.v0_9_phase2 is not None and config.v0_9_phase2.enabled
    assert config.v0_9_phase2.static_stage_end_fraction == 0.30
    assert config.v0_9_phase2.dynamic_stage_end_fraction == 0.70
    assert config.v0_9_phase2.symmetric_delta_budget == 0.15
    assert (
        config.v0_9_training.rollout_start_fractions[-1]
        < config.v0_9_phase2.dynamic_stage_end_fraction
    )
    assert config.v0_9_training.observable_horizon_probabilities[-1] == 0.35
    first = create_versioned_session(tmp_path, "v09-test")
    second = create_versioned_session(tmp_path, "v09-test")
    assert first.resolved_id == "v09-test"
    assert second.resolved_id == "v09-test-r1"


def test_rank_selection_enforces_long_horizon_and_burden_before_parsimony() -> None:
    selected = select_validation_rank(
        {
            2: [
                {
                    "total": 10.0,
                    "selection_score": 1.0,
                    "rollout_gain_h32": 0.03,
                    "burden_max": 0.2,
                },
                {
                    "total": 10.1,
                    "selection_score": 1.01,
                    "rollout_gain_h32": 0.02,
                    "burden_max": 0.25,
                },
            ],
            4: [
                {"total": 0.8, "rollout_gain_h32": -0.1, "burden_max": 0.2},
                {"total": 0.81, "rollout_gain_h32": -0.08, "burden_max": 0.2},
            ],
            8: [
                {"total": 0.7, "rollout_gain_h32": 0.1, "burden_max": 0.6},
                {"total": 0.71, "rollout_gain_h32": 0.1, "burden_max": 0.55},
            ],
        },
        longest_horizon=32,
        burden_limit=0.35,
    )
    assert selected["constraint_eligible_ranks"] == [2]
    assert selected["selected_rank"] == 2
    assert selected["constraints_satisfied"]

    fallback = select_validation_rank(
        {
            2: [{"total": 1.0, "rollout_gain_h32": -0.1, "burden_max": 0.2}],
            4: [{"total": 0.9, "rollout_gain_h32": -0.1, "burden_max": 0.2}],
        },
        longest_horizon=32,
        burden_limit=0.35,
    )
    assert fallback["selected_rank"] == 4
    assert not fallback["constraints_satisfied"]

    gain_driven = select_validation_rank(
        {
            2: [
                {
                    "selection_score": 22.99,
                    "rollout_gain_h80": 0.0012,
                    "burden_max": 0.011,
                }
            ],
            4: [
                {
                    "selection_score": 22.94,
                    "rollout_gain_h80": 0.0027,
                    "burden_max": 0.011,
                }
            ],
            8: [
                {
                    "selection_score": 22.90,
                    "rollout_gain_h80": -0.0024,
                    "burden_max": 0.011,
                }
            ],
        },
        longest_horizon=80,
        burden_limit=0.35,
        material_gain=0.02,
    )
    assert gain_driven["selected_rank"] == 4
    assert not gain_driven["constraints_satisfied"]
    assert gain_driven["gain_equivalent_ranks"] == [4]


def test_generic_metric_gates_handle_resolution_zero_baselines_and_inconclusive() -> None:
    frequency = evaluate_metric_gate(
        0.05,
        MetricGateSpec(
            "frequency_error",
            MetricDirection.LOWER_IS_BETTER,
            relative_margin=0.1,
            resolution_floor=0.0625,
        ),
        baseline=0.0,
    )
    assert frequency.status is GateStatus.PASS
    assert frequency.limit == 0.0625

    divergence = evaluate_metric_gate(
        0.15,
        MetricGateSpec(
            "divergence_rms",
            MetricDirection.LOWER_IS_BETTER,
            threshold=0.2,
            relative_margin=0.1,
        ),
        baseline=0.1,
    )
    assert divergence.status is GateStatus.FAIL
    assert divergence.limit is not None and abs(divergence.limit - 0.11) < 1e-12

    nonfinite = evaluate_metric_gate(
        float("nan"),
        MetricGateSpec("missing", MetricDirection.HIGHER_IS_BETTER, threshold=0.0),
    )
    combined = aggregate_gate_results("combined", [frequency, nonfinite], minimum_count=2)
    assert combined.status is GateStatus.INCONCLUSIVE


def test_v08_handoff_requires_three_jointly_passing_backbones(tmp_path: Path) -> None:
    raw = tmp_path / "runs" / "v08"
    compact = tmp_path / "results" / "v08"
    (compact / "evaluation").mkdir(parents=True)
    (compact / "completion.json").write_text(
        json.dumps({"status": "PASS", "all_required_stages_completed": True}), encoding="utf-8"
    )
    decision = {
        "v0_9_ready": True,
        "joint_v0_9_support_fraction": 1.0,
        "v0_7_route_on_new_problem": "R3",
        "context_family": "HISTORY_MLP",
        "nested_seed_support": {str(seed): {"v0_9_supported": True} for seed in (47, 53, 59)},
    }
    (compact / "evaluation" / "v0_8_scientific_decision.json").write_text(
        json.dumps(decision), encoding="utf-8"
    )
    for seed in (47, 53, 59):
        seed_root = raw / "seeds" / f"seed_{seed}"
        backbone = seed_root / "backbone.pt"
        backbone.parent.mkdir(parents=True, exist_ok=True)
        backbone.touch()
        (seed_root / "backbone_acceptance.json").write_text(
            json.dumps({"checkpoint": str(backbone)}), encoding="utf-8"
        )
        candidate = seed_root / "candidates" / "history_mlp" / "init_401"
        (candidate / "evaluation").mkdir(parents=True)
        (candidate / "checkpoints").mkdir()
        (candidate / "checkpoints" / "best.pt").touch()
        (candidate / "evaluation" / "training_summary.json").write_text(
            json.dumps({"validation": {"residual_standardized_mse": 0.1}}), encoding="utf-8"
        )
    handoff = audit_v0_8_handoff(
        "v08", runs_root=tmp_path / "runs", results_root=tmp_path / "results"
    )
    assert handoff.route == "R3" and len(handoff.seeds) == 3
    decision["nested_seed_support"]["53"]["v0_9_supported"] = False
    (compact / "evaluation" / "v0_8_scientific_decision.json").write_text(
        json.dumps(decision), encoding="utf-8"
    )
    try:
        audit_v0_8_handoff("v08", runs_root=tmp_path / "runs", results_root=tmp_path / "results")
    except ValueError as error:
        assert "requires 3/3" in str(error) and "53" in str(error)
    else:
        raise AssertionError("partial readiness was accepted")
    decision["v0_9_ready"] = False
    decision["joint_v0_9_support_fraction"] = 2.0 / 3.0
    decision["dynamic_context"] = "SUPPORTED"
    for support in decision["nested_seed_support"].values():
        support["supported"] = True
    decision["nested_seed_support"]["47"]["supported"] = False
    (compact / "evaluation" / "v0_8_scientific_decision.json").write_text(
        json.dumps(decision), encoding="utf-8"
    )
    supported = audit_v0_8_handoff(
        "v08",
        runs_root=tmp_path / "runs",
        results_root=tmp_path / "results",
        handoff_policy="supported",
    )
    assert not supported.strict_readiness
    assert supported.handoff_policy == "supported"


def test_legacy_v08_ready_report_is_detected_for_strict_reassessment() -> None:
    legacy = {
        "v0_9_ready": True,
        "nested_seed_support": {str(seed): {"supported": True} for seed in (47, 53, 59)},
    }
    assert not strict_v0_8_handoff_fields_present(legacy)
    strict = {
        **legacy,
        "joint_v0_9_support_fraction": 1.0,
        "v0_9_required_backbone_fraction": 1.0,
        "nested_seed_support": {
            str(seed): {"supported": True, "v0_9_supported": True} for seed in (47, 53, 59)
        },
    }
    assert strict_v0_8_handoff_fields_present(strict)


def test_clean_gate_ignores_generated_results_but_not_source_changes() -> None:
    assert dirty_source_paths(" M gpu_validation/v0_8/results/v08/completion.json\n") == []
    porcelain = "\n".join(
        (
            " M gpu_validation/v0_8/results/v08/report.md",
            "?? runs/v0_9/v09/failure.json",
            " M src/jka_model/adaptive/handoff.py",
        )
    )
    assert dirty_source_paths(porcelain) == ["src/jka_model/adaptive/handoff.py"]


def test_nested_v09_aggregation_and_completion_contract(tmp_path: Path) -> None:
    session = tmp_path / "session"
    output = tmp_path / "compact"
    (session / "rank_selection.json").parent.mkdir(parents=True)
    (session / "rank_selection.json").write_text(
        json.dumps({"selected_rank": 2, "selection_split": "validation"}), encoding="utf-8"
    )
    (session / "v0_8_handoff_audit.json").write_text(
        json.dumps({"route": "R3", "context_family": "history_mlp"}), encoding="utf-8"
    )
    for seed in (47, 53, 59):
        for mode in ("known", "latent_inferred"):
            for initialization in (701, 809, 907):
                evaluation = (
                    session
                    / "seeds"
                    / f"seed_{seed}"
                    / "formal"
                    / mode
                    / f"init_{initialization}"
                    / "evaluation"
                )
                evaluation.mkdir(parents=True)
                decision = {
                    "backbone_seed": seed,
                    "context_init_seed": 401,
                    "operator_init_seed": initialization,
                    "problem_name": "synthetic_problem",
                    "observable_objective": "synthetic_observables",
                    "condition_mode": mode,
                    "rank": 2,
                    "one_step_relative_gain": 0.2,
                    "operator_explained_fraction": 0.3,
                    "dynamic_over_static_gain": 0.1,
                    "history_over_shuffled_gain": 0.1,
                    "operator_explained_status": "PASS",
                    "dynamic_over_static_status": "PASS",
                    "controls_status": "PASS",
                    "closed_loop_by_horizon": {
                        "8": {"relative_gain_mean": 0.2, "gamma_operator_mean": 0.3, "pass": True}
                    },
                    "all_horizons_status": "PASS",
                    "longest_horizon_status": "PASS",
                    "operator_burden_status": "PASS",
                    "long_rollout_stability": "PASS",
                    "physics_status": "PASS",
                    "observable_status": "PASS",
                    "representation_physical_floor_status": "PASS",
                    "scientific_gates": {
                        "one_step_prediction": {"status": "PASS", "value": 0.2},
                        "operator_explained_residual": {
                            "status": "PASS",
                            "value": 0.3,
                        },
                        "dynamic_over_static": {"status": "PASS", "value": 0.1},
                        "history_over_shuffled": {"status": "PASS", "value": 0.1},
                        "dynamic_controls": {"status": "PASS", "value": 1.0},
                        "observable_noninferiority": {
                            "status": "PASS",
                            "value": 1.0,
                        },
                        "representation_physical_floor": {
                            "status": "PASS",
                            "value": 1.0,
                        },
                    },
                    "adaptive_koopman": "SUPPORTED",
                    "scientific_joint_pass": True,
                    "claims": {"phase1_error_attribution_complete": True},
                }
                (evaluation / "v0_9_scientific_decision.json").write_text(
                    json.dumps(decision), encoding="utf-8"
                )
                (evaluation / "training_summary.json").write_text(
                    json.dumps(
                        {
                            "completed_epochs": 4,
                            "validation": {"forecast": 0.1},
                            "test_locked_confirmation": "NOT_OPENED_DURING_TRAINING",
                            "phase1": {
                                "gradient_audit_records": 1,
                                "minimum_gradient_cosine": -0.2,
                                "observable_scale_state": {"split_fingerprint": f"train-{seed}"},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                _write_csv(
                    evaluation / "rollout_metrics.csv",
                    [{"horizon": 8, "relative_gain": 0.2, "gamma_operator": 0.3}],
                )
                _write_csv(
                    evaluation / "physical_metrics.csv",
                    [{"model": "adaptive", "velocity_relative_l2": 0.1}],
                )
                _write_csv(
                    evaluation / "observable_gate_results.csv",
                    [{"name": "velocity_relative_l2", "status": "PASS"}],
                )
                _write_csv(
                    evaluation / "error_attribution.csv",
                    [
                        {
                            "metric": "divergence_rms",
                            "data_floor": 0.01,
                            "reconstruction": 0.02,
                            "nominal": 0.03,
                            "adaptive": 0.025,
                        }
                    ],
                )
                _write_csv(
                    evaluation.parent / "logs" / "epoch_metrics.csv",
                    [{"epoch": 0, "validation_total": 0.1}],
                )
                (evaluation.parent / "logs" / "gradient_geometry.jsonl").write_text(
                    json.dumps(
                        {
                            "epoch": 1,
                            "names": ["forecast", "divergence"],
                            "cosine": [[1.0, -0.2], [-0.2, 1.0]],
                            "gradient_norms": {"forecast": 1.0, "divergence": 0.5},
                            "minimum_off_diagonal_cosine": -0.2,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
    result = aggregate_v0_9_results(session, output)
    assert result["low_rank_operator_adaptation"] == "SUPPORTED"
    assert result["physical_problem"] == "synthetic_problem"
    assert result["observable_objective"] == "synthetic_observables"
    assert result["operator_explained_residual"] == "SUPPORTED"
    assert result["dynamic_operator_adaptation"] == "SUPPORTED"
    assert result["observable_support"] == "SUPPORTED"
    assert result["representation_physical_floor"] == "SUPPORTED"
    assert result["numerical_stability"] == "PASS"
    assert result["long_rollout_skill"] == "PASS"
    assert not result["claims"]["oracle_condition_curriculum_train_only"]
    assert result["v1_0_ready"]
    assert result["compact_audit"]["complete"]
    assert (output / "report.md").is_file()
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "NUMERICAL STABILITY: PASS" in report
    assert "LONG-ROLLOUT SKILL: PASS" in report
    (session / "v0_8_handoff_audit.json").write_text(
        json.dumps(
            {
                "route": "R3",
                "context_family": "history_mlp",
                "handoff_policy": "supported",
                "strict_readiness": False,
            }
        ),
        encoding="utf-8",
    )
    for seed in (47, 53):
        for decision_path in session.glob(
            f"seeds/seed_{seed}/formal/*/init_*/evaluation/v0_9_scientific_decision.json"
        ):
            record = json.loads(decision_path.read_text(encoding="utf-8"))
            record["operator_explained_status"] = "INCONCLUSIVE"
            decision_path.write_text(json.dumps(record), encoding="utf-8")
    conditional = aggregate_v0_9_results(session, tmp_path / "conditional")
    assert conditional["adaptive_mechanism_result"] == "SUPPORTED"
    assert conditional["low_rank_operator_adaptation"] == "CONDITIONALLY_SUPPORTED"
    assert conditional["evidence_tier"] == "EXPLORATORY_CONDITIONAL"
    assert conditional["operator_explained_residual"] == "INCONCLUSIVE"
    assert not conditional["v1_0_ready"]
    validate_completion_payload(
        {
            "requested_validation_id": "v09",
            "resolved_validation_id": "v09-r1",
            "v0_8_validation_id": "v08",
            "status": "PASS",
            "git_commit": "abc",
            "all_required_stages_completed": True,
            "selected_rank": 2,
            "formal_training_run_count": 18,
            "formal_evaluation_run_count": 18,
            "scientific_status": "SUPPORTED",
            "v1_0_readiness": "READY",
            "output_paths": {},
        }
    )


def test_v09_docs_and_single_command_exist() -> None:
    required = {
        "README.md",
        "mathematical_contract.md",
        "physical_problem_extension.md",
        "operator_adaptation.md",
        "context_handoff.md",
        "stability_and_identifiability.md",
        "evaluation_and_observable_revision.md",
        "evaluation.md",
        "testing.md",
        "status.md",
        "technology_review.md",
        "technology_adoption_declaration.md",
        "code_walkthrough.md",
        "v0_9_scientific_report.md",
        "v1_0_handoff.md",
    }
    assert required <= {path.name for path in Path("docs/v0_9").glob("*.md")}
    readme = Path("gpu_validation/v0_9/README.md").read_text(encoding="utf-8")
    assert "gpu_validate_all.py" in readme and "--seeds 47 53 59" in readme
