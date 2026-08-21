from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from eval.evaluate_v0_8 import assess_context_acceptance, summarize_closed_loop_horizons
from gpu_validation.v0_8.scripts.gpu_validate_all import validate_completion_payload
from jka_model.config import load_config
from jka_model.context import aggregate_v0_8_results
from jka_model.utils import create_versioned_session


def test_v08_config_and_revision_id_contract(tmp_path: Path) -> None:
    config = load_config("gpu_validation/v0_8/configs/gpu_cylinder_context.yaml")
    assert config.project_version == "0.8.0"
    assert config.training.stage.value == "context"
    assert config.cylinder_wake_2d is not None
    assert config.cylinder_wake_2d.nx == 256
    assert config.koopman is not None and config.koopman.state_dim == 32
    assert config.v0_8_evaluation is not None
    assert config.v0_8_evaluation.context_initialization_seeds == (401, 503, 607)
    first = create_versioned_session(tmp_path, "v08-test")
    second = create_versioned_session(tmp_path, "v08-test")
    assert first.resolved_id == "v08-test"
    assert second.resolved_id == "v08-test-r1"


def test_completion_schema() -> None:
    validate_completion_payload(
        {
            "requested_validation_id": "v08",
            "resolved_validation_id": "v08-r1",
            "status": "PASS",
            "git_commit": "abc",
            "all_required_stages_completed": True,
            "backbone_run_count": 3,
            "v0_7_evaluation_run_count": 144,
            "v0_8_candidate_run_count": 27,
            "v0_8_test_run_count": 9,
            "final_route": "R3",
            "final_context_family": "attention",
            "scientific_status": "SUPPORTED",
            "output_paths": {"raw_run": "raw", "compact_result": "compact"},
        }
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_nested_aggregation_produces_complete_review_artifacts(tmp_path: Path) -> None:
    session = tmp_path / "session"
    data = session / "data"
    data.mkdir(parents=True)
    (data / "grid_adequacy.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    for seed in (47, 53, 59):
        (data / f"physical_acceptance_seed_{seed}.json").write_text(
            json.dumps({"status": "PASS"}), encoding="utf-8"
        )
    route_evaluation = session / "v0_7_assessment" / "evaluation"
    route_evaluation.mkdir(parents=True)
    (route_evaluation / "memory_classification.json").write_text(
        json.dumps({"residual_route": "R3"}), encoding="utf-8"
    )
    (session / "v0_8_family_selection.json").write_text(
        json.dumps(
            {
                "selected_family": "attention",
                "candidate_mean_validation_standardized_mse": {
                    "attention": 0.1,
                    "history_mlp": 0.2,
                    "instantaneous": 0.3,
                    "instantaneous_matched": 0.4,
                },
            }
        ),
        encoding="utf-8",
    )
    for seed in (47, 53, 59):
        backbone = session / "seeds" / f"seed_{seed}" / "backbone_acceptance.json"
        backbone.parent.mkdir(parents=True)
        backbone.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        for initialization in (401, 503, 607):
            for family in (
                "instantaneous",
                "instantaneous_matched",
                "history_mlp",
                "attention",
            ):
                candidate = (
                    session
                    / "seeds"
                    / f"seed_{seed}"
                    / "candidates"
                    / family
                    / f"init_{initialization}"
                )
                (candidate / "evaluation").mkdir(parents=True)
                (candidate / "evaluation" / "training_summary.json").write_text(
                    json.dumps(
                        {
                            "completed_epochs": 4,
                            "validation": {
                                "residual_nrmse": 0.4,
                                "residual_standardized_mse": 0.1,
                                "adequacy_r2": 0.7,
                            },
                            "test_locked_confirmation": "NOT_OPENED_DURING_TRAINING",
                        }
                    ),
                    encoding="utf-8",
                )
                if family == "attention":
                    (candidate / "logs").mkdir()
                    _write_csv(
                        candidate / "logs" / "epoch_metrics.csv",
                        [{"epoch": 1, "train_loss": 0.2, "validation_loss": 0.1}],
                    )
            evaluation = (
                session
                / "seeds"
                / f"seed_{seed}"
                / "contexts"
                / f"init_{initialization}"
                / "evaluation"
            )
            evaluation.mkdir(parents=True)
            decision = {
                "backbone_seed": seed,
                "context_init_seed": initialization,
                "v0_7_route_on_new_problem": "R3",
                "context_family": "ATTENTION",
                "test_locked_confirmation": {
                    "residual_nrmse": 0.4,
                    "residual_r2": 0.8,
                    "adequacy_r2": 0.7,
                    "adequacy_correlation": 0.9,
                },
                "context_ablation": {"residual_nrmse": 0.6},
                "context_ablation_gain": 0.25,
                "history_over_shuffled_gain": 0.1,
                "context_diagnostics": {"effective_rank": 3.0, "collapsed": False},
                "context_rank_status": "PASS",
                "koopman_adequacy": "CALIBRATED",
                "history_value": "SUPPORTED",
                "closed_loop_utility": "POSITIVE",
                "longest_horizon_utility": "POSITIVE",
                "closed_loop_by_horizon": {
                    "8": {
                        "relative_gain": 0.5,
                        "pass": True,
                    }
                },
                "physics_status": "PASS",
                "dynamic_context": "SUPPORTED",
                "evaluation_thresholds": {
                    "material_relative_gain": 0.02,
                    "min_context_effective_rank": 2.0,
                    "min_adequacy_r2": 0.0,
                    "min_adequacy_correlation": 0.5,
                    "seed_consistency_fraction": 2.0 / 3.0,
                },
            }
            (evaluation / "v0_8_scientific_decision.json").write_text(
                json.dumps(decision), encoding="utf-8"
            )
            _write_csv(
                evaluation / "closed_loop_metrics.csv",
                [
                    {
                        "horizon": 8,
                        "latent_rmse": 0.2,
                        "koopman_only_latent_rmse": 0.4,
                        "latent_relative_gain": 0.5,
                        "closure_burden_mean": 0.1,
                    }
                ],
            )
            _write_csv(
                evaluation / "physical_metrics.csv",
                [{"vorticity_relative_l2": 0.2, "lift_rmse": 0.1}],
            )
            torch.save(
                {
                    "residual_target": torch.randn(12, 8),
                    "residual_prediction": torch.randn(12, 8),
                    "adequacy_target": torch.rand(12, 1),
                    "adequacy_prediction": torch.rand(12, 1),
                    "contexts": torch.randn(12, 4),
                    "representative_rollout": {
                        "time": torch.arange(8),
                        "true_vorticity": torch.randn(8, 12, 6),
                        "predicted_vorticity": torch.randn(8, 12, 6),
                        "true_lift": torch.randn(8),
                        "predicted_lift": torch.randn(8),
                        "closed_loop_error": torch.rand(8),
                        "koopman_only_error": torch.rand(8),
                        "closure_burden": torch.rand(8),
                    },
                },
                evaluation / "diagnostic_series.pt",
            )
    output = tmp_path / "compact"
    result = aggregate_v0_8_results(session, output)
    assert result["dynamic_context"] == "SUPPORTED"
    assert result["v0_9_ready"]
    assert result["compact_audit"]["complete"]
    assert (output / "report.md").is_file()
    assert (output / "evaluation" / "compact_audit.json").is_file()
    assert (output / "evaluation" / "candidate_training_summary.csv").is_file()
    assert result["compact_audit"]["candidate_training_summary_count"] == 36
    assert result["compact_audit"]["selected_training_curve_row_count"] == 9
    assert len(list((output / "plots").glob("*.png"))) == 10

    # V0.9 readiness requires the same backbone seeds to satisfy all gates. It must
    # not combine context rank from one seed with long-horizon utility from another.
    for path in session.glob(
        "seeds/seed_53/contexts/init_*/evaluation/v0_8_scientific_decision.json"
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["context_rank_status"] = "FAIL"
        path.write_text(json.dumps(payload), encoding="utf-8")
    for path in session.glob(
        "seeds/seed_59/contexts/init_*/evaluation/v0_8_scientific_decision.json"
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["longest_horizon_utility"] = "NEUTRAL"
        path.write_text(json.dumps(payload), encoding="utf-8")
    stricter = aggregate_v0_8_results(session, tmp_path / "compact_strict")
    assert stricter["dynamic_context"] == "SUPPORTED"
    assert not stricter["v0_9_ready"]
    assert stricter["joint_v0_9_support_fraction"] == 1.0 / 3.0

    # Even two jointly passing backbones are not sufficient to activate the higher-risk
    # V0.9 adaptive-operator stage.
    for path in session.glob(
        "seeds/seed_59/contexts/init_*/evaluation/v0_8_scientific_decision.json"
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["longest_horizon_utility"] = "POSITIVE"
        path.write_text(json.dumps(payload), encoding="utf-8")
    two_of_three = aggregate_v0_8_results(session, tmp_path / "compact_two_of_three")
    assert two_of_three["joint_v0_9_support_fraction"] == 2.0 / 3.0
    assert not two_of_three["v0_9_ready"]


def test_rollout_horizon_summary_exposes_long_horizon_degradation() -> None:
    rows = [
        {"horizon": 8, "latent_rmse": 0.2, "koopman_only_latent_rmse": 0.4},
        {"horizon": 80, "latent_rmse": 0.5, "koopman_only_latent_rmse": 0.4},
    ]
    summary = summarize_closed_loop_horizons(rows, (8, 80), material_relative_gain=0.02)
    assert summary["8"]["pass"]
    assert not summary["80"]["pass"]


def test_context_acceptance_uses_configured_rank_and_r3_history() -> None:
    common = {
        "residual_route": "R3",
        "context_gain": 0.5,
        "history_gain": 0.3,
        "context_effective_rank": 1.9,
        "context_collapsed": False,
        "adequacy_r2": 0.4,
        "adequacy_correlation": 0.9,
        "material_relative_gain": 0.02,
        "min_context_effective_rank": 2.0,
        "min_adequacy_r2": 0.0,
        "min_adequacy_correlation": 0.5,
        "burden_pass": True,
    }
    low_rank = assess_context_acceptance(**common)
    assert not low_rank["rank_pass"] and not low_rank["context_supported"]
    no_history = assess_context_acceptance(
        **{**common, "context_effective_rank": 2.1, "history_gain": 0.0}
    )
    assert not no_history["history_pass"] and not no_history["context_supported"]
    uncalibrated = assess_context_acceptance(
        **{**common, "context_effective_rank": 2.1, "adequacy_r2": -0.1}
    )
    assert uncalibrated["context_supported"] and not uncalibrated["adequacy_pass"]


def test_mandatory_v08_docs_and_single_command_exist() -> None:
    docs = {
        "README.md",
        "mathematical_contract.md",
        "cylinder_wake_problem.md",
        "context_semantics.md",
        "evaluation.md",
        "testing.md",
        "status.md",
        "v0_8_scientific_report.md",
        "v0_9_problem_extension.md",
        "technology_review.md",
        "technology_adoption_declaration.md",
        "code_walkthrough.md",
    }
    assert docs <= {path.name for path in Path("docs/v0_8").glob("*.md")}
    assert Path("scripts/explain_v0_8.py").is_file()
    assert Path("gpu_validation/v0_8/scripts/gpu_validate_all.py").is_file()
    assert Path("gpu_validation/v0_8/scripts/gpu_reassess_existing.py").is_file()
    readme = Path("gpu_validation/v0_8/README.md").read_text(encoding="utf-8")
    assert "gpu_validate_all.py" in readme and "--seeds 47 53 59" in readme
    source = Path("src/jka_model/context/routing.py").read_text(encoding="utf-8")
    assert "cylinder" not in source.lower()
