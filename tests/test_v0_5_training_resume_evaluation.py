"""End-to-end CPU training, exact resume, evaluation, and run-record checks."""

from __future__ import annotations

import csv
import json
from dataclasses import replace

import torch

from eval.evaluate_v0_5 import evaluate_v0_5
from jka_model.config import load_config
from jka_model.utils import load_checkpoint
from train.train_v0_5 import train_v0_5


def test_v0_5_training_resume_is_exact_and_records_are_complete(tmp_path) -> None:
    config = load_config("configs/v0_5/advection_diffusion_2d_cpu_smoke.yaml")
    config = replace(config, training=replace(config.training, run_root=str(tmp_path)))
    uninterrupted = train_v0_5(
        config,
        device="cpu",
        run_name="uninterrupted",
        checkpoint_epochs={1},
    )
    resumed = train_v0_5(
        config,
        device="cpu",
        resume_from=uninterrupted.run_dir / "checkpoints" / "epoch_0001.pt",
        run_name="resumed",
        checkpoint_epochs=set(),
    )
    assert (uninterrupted.run_dir / "checkpoints" / "epoch_0001.pt").is_file()
    assert not (uninterrupted.run_dir / "checkpoints" / "epoch_0002.pt").exists()
    assert not list((resumed.run_dir / "checkpoints").glob("epoch_*.pt"))
    left = load_checkpoint(uninterrupted.latest_checkpoint)
    right = load_checkpoint(resumed.latest_checkpoint)
    assert left.epoch == right.epoch == 2
    assert left.global_step == right.global_step
    assert left.online_model_state is not None and right.online_model_state is not None
    assert all(
        torch.equal(left.online_model_state[name], right.online_model_state[name])
        for name in left.online_model_state
    )
    with (resumed.run_dir / "logs" / "epoch_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    step_lines = (resumed.run_dir / "logs" / "step_metrics.jsonl").read_text().splitlines()
    assert len(step_lines) == right.global_step
    for relative in (
        "config/resolved_config.yaml",
        "metadata/run_manifest.json",
        "metadata/environment.json",
        "metadata/git_state.json",
        "metadata/data_manifest.json",
        "metadata/model_summary.txt",
        "logs/epoch_metrics.csv",
        "logs/step_metrics.jsonl",
        "checkpoints/last.pt",
        "checkpoints/best_forecast.pt",
        "checkpoints/best_forecast_post_warmup.pt",
        "checkpoints/best_physics.pt",
        "checkpoints/best_physics_post_warmup.pt",
        "evaluation/final_metrics.json",
        "evaluation/rollout_by_horizon.csv",
        "evaluation/spectrum.json",
        "evaluation/physics_metrics.json",
        "evaluation/baseline_metrics.json",
        "plots/training_losses.png",
        "plots/rollout_error.png",
        "plots/physics_metrics.png",
        "reports/training_record.md",
        "reports/test_record.md",
        "reports/final_report.md",
    ):
        assert (resumed.run_dir / relative).is_file()
    run_manifest = json.loads((resumed.run_dir / "metadata/run_manifest.json").read_text())
    assert run_manifest["git_branch"] is not None
    assert isinstance(run_manifest["git_dirty"], bool)
    assert run_manifest["split_manifest"] == right.split_manifest
    assert run_manifest["end_time"]
    assert run_manifest["environment"]["device"] == "cpu"
    metrics = evaluate_v0_5(
        config, checkpoint=resumed.best_checkpoint, device="cpu", run_dir=resumed.run_dir
    )
    assert metrics["split"] == "test"
    assert metrics["finite"]
    assert metrics["scientific_acceptance"] == "PENDING_GPU"
    assert metrics["rollout"]["long"]["persistence_rmse"] >= 0
    assert metrics["learned_frequency_hz"] >= 0
    assert torch.isfinite(torch.tensor(metrics["learned_decay_rate"]))
    assert metrics["latent_max_std"] >= metrics["latent_min_std"] >= 0
