from __future__ import annotations

from pathlib import Path

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
    assert config.v0_7_evaluation is not None
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


def test_v0_7_problem_is_explicitly_version_classified() -> None:
    text = Path("gpu_validation/v0_7/problems/v0_7_synthetic_latent_memory.yaml").read_text(
        encoding="utf-8"
    )
    assert "version_owner: v0.7" in text
    assert "scientific_acceptance_source: false" in text


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
