from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

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
    for seed in (47, 53, 59):
        for initialization in (401, 503, 607):
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
                "closed_loop_utility": "POSITIVE",
                "physics_status": "PASS",
                "dynamic_context": "SUPPORTED",
            }
            (evaluation / "v0_8_scientific_decision.json").write_text(
                json.dumps(decision), encoding="utf-8"
            )
            _write_csv(
                evaluation / "closed_loop_metrics.csv",
                [{"latent_rmse": 0.2, "closure_burden_mean": 0.1}],
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
    assert (output / "report.md").is_file()
    assert len(list((output / "plots").glob("*.png"))) == 10


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
    readme = Path("gpu_validation/v0_8/README.md").read_text(encoding="utf-8")
    assert "gpu_validate_all.py" in readme and "--seeds 47 53 59" in readme
    source = Path("src/jka_model/context/routing.py").read_text(encoding="utf-8")
    assert "cylinder" not in source.lower()
