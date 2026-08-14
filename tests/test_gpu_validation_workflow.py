"""CPU-only tests for the complete GPU validation report builder."""

from __future__ import annotations

import json
from pathlib import Path

from gpu_validation.v0_5.scripts.gpu_validate_all import (
    EVALUATION_CHECKPOINTS,
    _build_final_report,
    _write_final_markdown,
)
from gpu_validation.v0_5.scripts.gpu_validate_science import _build_report as build_science_report


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _evaluation(*, frequency_pass: bool, rmse_scale: float) -> dict[str, object]:
    return {
        "status": "PASS",
        "frequency_gate_pass": frequency_pass,
        "frequency_relative_error": 0.97,
        "frequency_threshold": 0.05,
        "decay_gate_pass": True,
        "decay_relative_error": 0.1,
        "decay_threshold": 0.2,
        "stability_gate_pass": True,
        "spectral_abscissa": -0.01,
        "spectral_abscissa_threshold": 0.001,
        "beats_persistence": {name: True for name in ("short", "medium", "long")},
        "mass_gate_pass": {name: True for name in ("short", "medium", "long")},
        "operator_gate_pass": {name: True for name in ("short", "medium", "long")},
        "relative_mass_drift_threshold": 0.01,
        "operator_mse_threshold": 0.0001,
        "ablation_skill_degradation_threshold": 0.05,
        "ablation_constraint_degradation_threshold": 0.10,
        "long_beats_persistence": True,
        "reconstruction_rmse": 0.5,
        "max_samples_per_second": 100.0,
        "peak_gpu_memory_bytes": 1024.0,
        "forecast": {
            horizon: {
                "rmse": rmse_scale * multiplier,
                "relative_l2": rmse_scale * multiplier / 4.0,
                "persistence_rmse": 2.0,
                "mass_drift": rmse_scale * multiplier * 2.0,
                "operator": rmse_scale * multiplier * 3.0,
            }
            for horizon, multiplier in (("short", 1.0), ("medium", 2.0), ("long", 3.0))
        },
    }


def test_final_report_separates_workflow_pass_from_scientific_failure(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    full = tmp_path / "physics"
    resumed = tmp_path / "resumed"
    ablation = tmp_path / "no_physics"
    for run in (full, resumed, ablation):
        _write_json(run / "metadata/run_manifest.json", {"completed_epochs": 150})
    _write_json(
        artifacts / "preflight.json",
        {
            "status": "PASS",
            "git_dirty": False,
            "cuda_version": "12.8",
            "device": "GPU",
            "matrix_exp_device": "cuda:0",
        },
    )
    _write_json(artifacts / "smoke.json", {"status": "PASS"})
    _write_json(artifacts / "profile.json", {"status": "PASS"})
    _write_json(
        artifacts / f"resume_check_{full.name}_vs_{resumed.name}.json",
        {"status": "PASS", "weights_bitwise_equal": True},
    )
    for run, scale in ((full, 1.0), (ablation, 2.0)):
        for checkpoint in EVALUATION_CHECKPOINTS:
            suffix = "" if checkpoint == "best_forecast" else f"_{checkpoint}"
            _write_json(
                artifacts / f"{run.name}{suffix}_metrics.json",
                _evaluation(frequency_pass=False, rmse_scale=scale),
            )
    state = {"steps": {"every_gate": {"status": "PASS"}}}

    report = _build_final_report(
        validation_id="test-validation",
        state=state,
        artifacts_dir=artifacts,
        full_run=full,
        resumed_run=resumed,
        no_physics_run=ablation,
    )

    assert report["workflow_status"] == "PASS"
    assert report["scientific_status"] == "FAIL"
    assert report["overall_acceptance"] == "NOT_ACCEPTED"
    assert report["all_checklist_items_complete"] is True
    comparison = report["physics_vs_no_physics"]
    assert comparison["long"]["rmse"]["relative_change"] == -0.5
    output = tmp_path / "final.md"
    _write_final_markdown(output, report)
    rendered = output.read_text(encoding="utf-8")
    assert "workflow status: **PASS**" in rendered
    assert "scientific status: **FAIL**" in rendered


def test_three_seed_science_report_requires_every_seed_and_consistent_ablation() -> None:
    seeds = [47, 53, 59]
    results: dict[int, dict[str, dict[str, object]]] = {}
    for seed in seeds:
        physics = _evaluation(frequency_pass=True, rmse_scale=0.5)
        no_physics = _evaluation(frequency_pass=True, rmse_scale=1.0)
        # A tiny raw RMSE regression is acceptable when it is negligible relative to
        # the persistence baseline; this avoids unstable ratios near the error floor.
        physics["forecast"]["short"]["rmse"] = 1.02
        no_physics["forecast"]["short"]["rmse"] = 1.0
        physics["run_dir"] = f"physics-{seed}"
        no_physics["run_dir"] = f"no-physics-{seed}"
        results[seed] = {"physics": physics, "no_physics": no_physics}
    report = build_science_report(validation_id="test", seeds=seeds, results=results)
    assert report["scientific_status"] == "PENDING_REVIEW"
    assert report["all_seed_gates_pass"] is True
    assert report["physics_ablation_consistent"] is True
    results[53]["physics"]["frequency_gate_pass"] = False
    failed = build_science_report(validation_id="test", seeds=seeds, results=results)
    assert failed["scientific_status"] == "FAIL"
    results[53]["physics"]["frequency_gate_pass"] = True
    for seed in seeds:
        results[seed]["physics"]["forecast"]["short"]["operator"] = 4.0
    constraint_failed = build_science_report(validation_id="test", seeds=seeds, results=results)
    assert constraint_failed["scientific_status"] == "FAIL"
