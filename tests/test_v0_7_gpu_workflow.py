from __future__ import annotations

from pathlib import Path

from gpu_validation.v0_7.scripts.gpu_validate_all import (
    _resolved,
    _with_history,
    validate_completion_payload,
    validate_failure_payload,
)
from jka_model.config import load_config
from jka_model.utils import create_versioned_session


def test_v0_7_gpu_template_is_strict_and_complete() -> None:
    config = load_config("gpu_validation/v0_7/configs/gpu_residual_multiseed.yaml")
    assert config.project_version == "0.7.0"
    assert config.training.stage.value == "residual"
    assert config.residual_closure is not None
    assert config.residual_training is not None
    assert config.memory_sweep is not None
    assert config.memory_sweep.history_lengths == (1, 2, 4, 8, 16)
    assert config.memory_sweep.seed_consistency_fraction == 2 / 3
    assert config.memory_sweep.initialization_seeds == (101, 211, 307)
    assert config.residual_training.initialization_seed == 101
    assert config.v0_7_evaluation is not None
    assert config.v0_7_evaluation.max_closure_burden == 0.25
    assert config.v0_7_evaluation.min_residual_significance == 0.01
    assert config.v0_7_evaluation.formal_record_count == 144
    assert set(config.residual_closure.variants) == {
        "zero",
        "linear",
        "instantaneous",
        "history",
        "shuffled_history",
    }


def test_validation_id_automatically_increments_revision(tmp_path: Path) -> None:
    first = create_versioned_session(tmp_path, "v07-test")
    second = create_versioned_session(tmp_path, "v07-test")
    third = create_versioned_session(tmp_path, "v07-test-r1")
    assert first.resolved_id == "v07-test"
    assert second.resolved_id == "v07-test-r1"
    assert third.resolved_id == "v07-test-r2"


def test_validation_id_reserves_existing_compact_result(tmp_path: Path) -> None:
    results = tmp_path / "results"
    (results / "v07-test").mkdir(parents=True)
    session = create_versioned_session(tmp_path / "runs", "v07-test", reserved_roots=(results,))
    assert session.resolved_id == "v07-test-r1"


def test_closure_seed_cannot_change_backbone_data_ownership() -> None:
    template = load_config("gpu_validation/v0_7/configs/gpu_residual_multiseed.yaml")
    backbone = _resolved(template, 47)
    closure = _with_history(backbone, 8, 211)
    assert closure.training.seed == backbone.training.seed == 47
    assert closure.data.split.seed == backbone.data.split.seed == 47
    assert closure.residual_training is not None
    assert closure.residual_training.initialization_seed == 211
    assert closure.data == backbone.data


def test_v0_7_problem_is_explicitly_version_classified() -> None:
    text = Path("gpu_validation/v0_7/problems/v0_7_synthetic_latent_memory.yaml").read_text(
        encoding="utf-8"
    )
    assert "version_owner: v0.7" in text
    assert "scientific_acceptance_source: false" in text


def test_gpu_completion_and_failure_schema() -> None:
    validate_completion_payload(
        {
            "requested_validation_id": "requested",
            "resolved_validation_id": "resolved",
            "git_commit": "abc",
            "backbone_seeds": [47, 53, 59],
            "closure_initialization_seeds": [101, 211, 307],
            "expected_evaluation_records": 144,
            "actual_evaluation_records": 144,
            "all_expected_runs_completed": True,
            "provenance_checks_passed": True,
            "required_reports_produced": True,
            "status": "PASS",
        }
    )
    validate_failure_payload(
        {
            "validation_id": "failed",
            "status": "FAILED_INCOMPLETE",
            "failed_stage": "train_and_evaluate",
            "failed_run": {"seed": 47},
            "error": "RuntimeError: example",
            "completed_record_count": 12,
            "expected_record_count": 144,
            "last_valid_checkpoint": "best.pt",
            "git_commit": "abc",
        }
    )


def test_mandatory_v0_7_docs_and_gpu_wrappers_exist() -> None:
    docs = {
        "README.md",
        "implementation_checklist.md",
        "architecture.md",
        "residual_semantics.md",
        "mori_zwanzig_semantics.md",
        "training.md",
        "evaluation.md",
        "testing.md",
        "latest_tech_review.md",
        "novelty_positioning.md",
        "references.md",
        "technology_adoption_declaration.md",
        "code_walkthrough.md",
        "status.md",
        "residual_decision_report.md",
        "v0_8_route_recommendation.md",
        "revised_addendum_v2_audit.md",
    }
    wrappers = {
        "gpu_preflight.py",
        "gpu_smoke.py",
        "gpu_build_residual_cache.py",
        "gpu_train.py",
        "gpu_evaluate.py",
        "gpu_compare.py",
        "gpu_profile.py",
        "gpu_validate_all.py",
    }
    assert docs <= {path.name for path in Path("docs/v0_7").glob("*.md")}
    assert wrappers <= {path.name for path in Path("gpu_validation/v0_7/scripts").glob("*.py")}
    assert Path("scripts/compare_residual_memory_v0_7.py").is_file()
    assert Path("scripts/explain_v0_7.py").is_file()


def test_residual_router_is_problem_agnostic() -> None:
    text = Path("src/jka_model/residual/assessment.py").read_text(encoding="utf-8").lower()
    assert "advection" not in text
    assert "diffusion" not in text
    assert "cylinder" not in text
