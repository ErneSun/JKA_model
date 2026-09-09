from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eval import audit_v0_9_bottleneck as audit
from gpu_validation.v0_9.scripts import gpu_audit_bottleneck as workflow
from jka_model.adaptive import AdaptiveCache, AdaptiveKoopmanModel, AdaptiveTrajectory
from jka_model.adaptive.models import FactorizedAdaptiveOperator
from jka_model.config import ProjectConfig, load_config, stable_config_hash
from jka_model.context.models import HistoryMLPContextEncoder
from jka_model.data import ChannelStandardizer, TrajectoryDataset, TrajectoryRecord
from jka_model.manifold.bottleneck import (
    diagnostic_priority,
    error_decomposition,
    evaluate_gauge,
    field_views,
    fit_gauge,
    latent_spectrum,
    summarize_errors,
    transition_contract,
)
from jka_model.residual.cache import file_sha256


def _config(mode: str = "latent_inferred") -> ProjectConfig:
    payload = load_config("gpu_validation/v0_9/configs/gpu_adaptive_koopman.yaml").to_dict()
    payload["v0_9_adaptive"].update(condition_mode=mode, rank=2)
    payload["v0_9_phase3"].update(observer_admission_enabled=True, raw_field_rollout_stride=1)
    payload["v0_9_phase2"].update(paired_horizon=4, static_rank=1, dynamic_rank=1)
    payload["v0_9_training"].update(
        rollout_horizons=[1, 2, 4],
        rollout_weights=[1.0, 1.0, 1.0],
        rollout_start_fractions=[0.0, 0.2, 0.5],
        observable_horizons=[1, 2, 4],
        observable_horizon_weights=[1.0, 1.0, 1.0],
        observable_horizon_probabilities=[0.2, 0.3, 0.5],
        physics_horizon=2,
    )
    payload["v0_9_evaluation"]["rollout_horizons"] = [1, 2, 4]
    return ProjectConfig.from_dict(payload)


def test_exact_decomposition_does_not_mistake_cancellation_for_error_floor() -> None:
    y = torch.tensor([1.0, 2.0], dtype=torch.float64)
    perfect = error_decomposition(y, y + 10, y)
    assert perfect["total_mse"] == 0
    assert perfect["reference_decode_mse"] == perfect["propagation_mse"] == 100
    assert perfect["cross_term"] == -200
    assert perfect["closure_relative"] == 0
    assert diagnostic_priority(perfect) == "COUPLED_CROSS_TERM"
    assert diagnostic_priority(error_decomposition(y + 1, y, y)) == "PROPAGATION_PRIORITY"
    assert (
        diagnostic_priority(error_decomposition(y + 1, y + 1, y))
        == "REPRESENTATION_DECODE_PRIORITY"
    )


def test_decomposition_is_scale_consistent_and_rejects_invalid_data() -> None:
    torch.manual_seed(43)
    x, r, y = torch.randn(3, 100, dtype=torch.float64)
    result = error_decomposition(x, r, y)
    scaled = error_decomposition(7 * x, 7 * r, 7 * y)
    assert scaled["total_mse"] == pytest.approx(49 * result["total_mse"])
    assert scaled["relative_l2"] == pytest.approx(result["relative_l2"])
    assert error_decomposition(x, r, y * 0)["relative_l2"] is None
    with pytest.raises(ValueError, match="finite"):
        error_decomposition(x * torch.nan, r, y)


def test_channel_pressure_gauge_and_valid_vorticity_stencils() -> None:
    field = torch.zeros(3, 8, 8)
    field[1] = torch.arange(8)[:, None] * 0.25
    field[2] = 100
    mask = torch.ones(8, 8, dtype=torch.bool)
    mask[3, 3] = False
    field[:, 3, 3] = 1e8
    result = field_views(field, mask, dx=0.25, dy=0.25)
    assert torch.equal(result["vorticity_interior"], torch.ones_like(result["vorticity_interior"]))
    assert result["pressure_demeaned_fluid"].abs().max() == 0
    assert result["pressure_fluid"].mean() == 100
    assert result["field_all"].max() == 1e8
    assert result["field_fluid"].max() == 100


def test_gauge_fits_train_only_and_detects_heldout_shift() -> None:
    torch.manual_seed(29)
    train = torch.randn(100, 3, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    fit = fit_gauge(train, train @ q)
    held = torch.randn(40, 3, dtype=torch.float64)
    result = evaluate_gauge(fit, held, held @ q, -torch.eye(3))
    assert result["aligned_nrmse"] < 1e-12
    assert result["full_commutator"] < 1e-12
    shifted = evaluate_gauge(fit, held, held @ q + 10, -torch.eye(3))
    assert shifted["aligned_nrmse"] > 0.9
    assert shifted["heldout_mean_shift"] > 10
    shifted_fit = fit_gauge(train, train @ q + 10)
    affine = evaluate_gauge(shifted_fit, held, held @ q + 10, -torch.eye(3))
    assert affine["aligned_nrmse"] < 1e-12
    assert affine["full_commutator"] < 1e-12
    assert affine["affine_dynamical_action_defect"] > 0.9
    asymmetric = torch.diag(torch.tensor([-1.0, -2.0, -3.0]))
    assert evaluate_gauge(fit, held, held @ q, asymmetric)["data_action_commutator"] > 0.01


def test_spectrum_reports_collapsed_and_low_rank_without_false_rank_one() -> None:
    assert latent_spectrum(torch.zeros(20, 4))["effective_rank"] == 0
    x = torch.arange(20.0)[:, None] * torch.ones(1, 4)
    result = latent_spectrum(x)
    assert result["effective_rank"] == pytest.approx(1.0)
    assert result["rank_99"] == 1
    with pytest.raises(ValueError, match="constant"):
        fit_gauge(x * 0, x)


def test_causal_transition_index_excludes_unavailable_future_input() -> None:
    inputs = torch.zeros(10, 2)
    inputs[4:, 0] = 3
    before = transition_contract(inputs, 3, 2)
    origin = transition_contract(inputs, 4, 2)
    assert not before["changed_at_origin"] and before["changes_after_origin"] == 1
    assert origin["changed_at_origin"] and origin["changes_after_origin"] == 0
    assert origin["input_indices"] == [4, 5]
    assert origin["target_state_indices"] == [5, 6]


def test_trajectory_balancing_preserves_exact_identity() -> None:
    base = {
        "split": "validation",
        "predictor": "adaptive",
        "reference": "online",
        "horizon": 1,
        "channel": "field_all",
        "input_group": "all",
    }
    one = error_decomposition(torch.ones(3), torch.zeros(3), torch.zeros(3))
    four = error_decomposition(torch.ones(3) * 2, torch.zeros(3), torch.zeros(3))
    rows = [{**base, "trajectory_id": "a", **one} for _ in range(9)]
    rows.append({**base, "trajectory_id": "b", **four})
    result = summarize_errors(rows)[0]
    assert result["total_mse"] == 2.5  # NOT window-weighted 1.3
    assert result["relative_l2"] is None
    assert result["propagation_mse"] + result["reference_decode_mse"] + result["cross_term"] == 2.5


def test_initial_observer_admission_not_later_test_verdict_controls_replay() -> None:
    config = _config()
    payload = {
        "observer_admission": {"admitted": False},
        "locked_test_metrics": {"observer_admitted": 1},
    }
    state = audit.replay_state(config, payload)
    assert state.active_components == "dynamic" and not state.condition_admitted
    payload["observer_admission"]["admitted"] = True
    payload["locked_test_metrics"]["observer_admitted"] = 0
    assert audit.replay_state(config, payload).active_components == "full"
    assert audit.replay_state(_config("known"), {}).active_components == "full"
    with pytest.raises(ValueError, match="initial observer"):
        audit.replay_state(config, {})


def test_checkpoint_hash_checked_before_runtime_defaults_and_relocation(tmp_path: Path) -> None:
    config = _config()
    raw = config.to_dict()
    raw.pop("tags")  # historical omission: hash it BEFORE any defaults are filled
    row = {
        "route": "joint",
        "seed": config.training.seed,
        "operator_seed": 701,
        "condition_mode": "latent_inferred",
    }
    payload = {
        "schema_version": "v0.9-phase3-route-2",
        "route": "joint",
        "config": raw,
        "config_hash": stable_config_hash(raw),
    }
    assert audit.validate_route_payload(payload, row).training.seed == row["seed"]
    raw["training"]["seed"] = 987
    with pytest.raises(ValueError, match="raw config hash"):
        audit.validate_route_payload(payload, row)
    target = tmp_path / "runs" / "v0_9" / "source" / "checkpoints" / "best.pt"
    target.parent.mkdir(parents=True)
    target.touch()
    assert (
        audit.relocate_run_path("/different/server/runs/v0_9/source/checkpoints/best.pt", tmp_path)
        == target
    )
    with pytest.raises(ValueError):
        audit.relocate_run_path("runs/../../outside.pt", tmp_path)
    with pytest.raises(FileNotFoundError, match="no retraining"):
        audit.relocate_run_path("runs/v0_9/missing.pt", tmp_path)


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.mean(dim=(-1, -2)))


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)[..., None, None].expand(*z.shape[:-1], 3, 8, 8)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.online_encoder = _Encoder()
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.training_decoder = _Decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(x)

    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.training_decoder(z)


@pytest.mark.parametrize("mode", ["known", "latent_inferred"])
def test_checkpoint_audit_replays_real_operator_without_training(
    tmp_path: Path, monkeypatch, mode: str
) -> None:
    torch.manual_seed(22)
    config = _config(mode)
    backbone = _Backbone()
    ref_encoder, ref_decoder = (
        copy.deepcopy(backbone.online_encoder),
        copy.deepcopy(backbone.training_decoder),
    )
    nominal = -0.01 * torch.eye(8)
    encoder = HistoryMLPContextEncoder(8, 2, 2, 1, 8)
    operator = FactorizedAdaptiveOperator(nominal, 2, config.v0_9_adaptive, config.v0_9_phase2)
    model = AdaptiveKoopmanModel(encoder, operator)
    normalizer = ChannelStandardizer()
    normalizer.mean, normalizer.scale = torch.zeros(3), torch.ones(3)
    normalizer.spatial_dim, normalizer.layout = 2, "channels_first"
    records, trajectories = [], []
    for split in ("train", "validation", "test"):
        raw = torch.randn(8, 3, 8, 8) * 0.1 + 1
        dt = torch.full((7,), 0.1)
        records.append(
            TrajectoryRecord(split, raw, dt, valid_mask=torch.ones(8, 8, dtype=torch.bool))
        )
        trajectories.append(
            AdaptiveTrajectory(
                split,
                split,
                "smooth_ramp",
                3,
                torch.zeros(8, 8),
                dt,
                torch.ones(1),
                torch.ones(7, 2),
                torch.zeros(7, 8),
            )
        )
    source = tmp_path / "runs" / "v0_9" / "source"
    source.mkdir(parents=True)
    for name in ("context.pt", "backbone.pt"):
        torch.save({}, source / name)
    cache = AdaptiveCache(
        tuple(trajectories),
        file_sha256(source / "backbone.pt"),
        "config",
        file_sha256(source / "context.pt"),
        "data",
        {s: [s] for s in ("train", "validation", "test")},
        normalizer.state_dict(),
        nominal,
    )
    payload = {
        "schema_version": "v0.9-phase3-route-2",
        "route": "joint",
        "config": config.to_dict(),
        "config_hash": config.stable_hash,
        "backbone_state": backbone.state_dict(),
        "adaptive_state": model.state_dict(),
        "reference_encoder_state": ref_encoder.state_dict(),
        "reference_decoder_state": ref_decoder.state_dict(),
        "normalizer_state": normalizer.state_dict(),
        "source_backbone_sha256": cache.backbone_checkpoint_sha256,
        "source_context_sha256": cache.context_checkpoint_sha256,
        "adaptive_cache_fingerprint": cache.fingerprint,
        "observer_admission": {"admitted": False},
        "condition_mean": torch.zeros(3),
        "condition_std": torch.ones(3),
    }
    checkpoint = source / "best.pt"
    torch.save(payload, checkpoint)
    before = file_sha256(checkpoint)
    monkeypatch.setattr(audit, "load_adaptive_cache", lambda _: cache)
    monkeypatch.setattr(
        audit,
        "load_cylinder_wake_dataset",
        lambda *a: SimpleNamespace(records=TrajectoryDataset(records), problem_spec=None),
    )
    monkeypatch.setattr(audit, "data_fingerprint", lambda *a: "data")
    monkeypatch.setattr(
        audit,
        "_build_phase3_models",
        lambda *a, **kw: (
            backbone,
            ref_encoder,
            ref_decoder,
            model,
            normalizer,
            {"history_length_steps": 2},
        ),
    )
    row = {
        "route": "joint",
        "seed": config.training.seed,
        "operator_seed": 701,
        "condition_mode": mode,
        "checkpoint": str(checkpoint),
        "validation": {},
        "locked_test": {},
    }
    # Independent closed-form reference: all adaptive heads are zero initialized,
    # hence the real FactorizedAdaptiveOperator must equal exp(A0*t) on this fixture.
    with torch.no_grad():
        for record in records[1:]:
            metrics = row["validation" if record.trajectory_id == "validation" else "locked_test"]
            for h in (1, 2, 4):
                values = []
                for origin in (1, 2, 3):
                    initial = backbone.encode(record.states_raw[origin][None])
                    prediction = backbone.decode(initial @ torch.matrix_exp(nominal * (0.1 * h)).T)[
                        0
                    ]
                    truth = record.states_raw[origin + h]
                    values.append(float((prediction - truth).norm() / truth.norm()))
                metrics[f"decoded_field_relative_l2_h{h}"] = sum(values) / len(values)
    result = audit.audit_checkpoint(
        root=tmp_path,
        row=row,
        phase2_id="phase2",
        device="cpu",
        output=tmp_path / "audit",
        handoff={
            "context_checkpoint": str(source / "context.pt"),
            "backbone_checkpoint": str(source / "backbone.pt"),
        },
    )
    assert result["window_count"] == 6
    assert result["scientific_acceptance"] == "NOT_EVALUATED"
    assert result["condition_admitted"] == (mode == "known")
    assert result["summary"] and result["validation_priorities"]
    assert file_sha256(checkpoint) == before
    assert all(p.grad is None for p in model.parameters())
    # A second replay checks comparison against source metrics, not just missing keys.
    for key, values in result["source_replay"].items():
        split, metric = key.split(":")
        row["validation" if split == "validation" else "locked_test"][metric] = values["replayed"]
    second = audit.audit_checkpoint(
        root=tmp_path,
        row=row,
        phase2_id="phase2",
        device="cpu",
        output=tmp_path / "audit2",
        handoff={
            "context_checkpoint": str(source / "context.pt"),
            "backbone_checkpoint": str(source / "backbone.pt"),
        },
    )
    assert all(v["absolute_difference"] == 0 for v in second["source_replay"].values())
    row["locked_test"]["decoded_field_relative_l2_h4"] += 0.1
    with pytest.raises(ValueError, match="disagrees with source"):
        audit.audit_checkpoint(
            root=tmp_path,
            row=row,
            phase2_id="phase2",
            device="cpu",
            output=tmp_path / "bad-replay",
            handoff={
                "context_checkpoint": str(source / "context.pt"),
                "backbone_checkpoint": str(source / "backbone.pt"),
            },
        )
    audit.write_audit_report(tmp_path / "report.md", [result], source_id="source")
    assert "not fresh independent validation" in (tmp_path / "report.md").read_text()


def test_workflow_failure_report_and_revision_ids(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workflow, "get_git_commit", lambda _: "test-commit")
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **kw: "")
    args = argparse.Namespace(
        validation_id="audit", source_id="missing", device="cpu", skip_tests=True
    )
    for suffix in ("audit", "audit-r1"):
        with pytest.raises(FileNotFoundError):
            workflow.run_audit(args, root=tmp_path)
        compact = tmp_path / "gpu_validation" / "v0_9" / "results" / suffix
        failure = json.loads((compact / "failure.json").read_text())
        assert failure["status"] == "FAILED_INCOMPLETE"
        assert failure["training_performed"] is False
        assert (compact / "report.md").is_file()
        assert (tmp_path / "runs" / "v0_9" / suffix / "logs" / "audit.log").is_file()


@pytest.mark.parametrize("mismatched_windows", [False, True])
def test_complete_workflow_retains_paired_cells_and_rejects_window_drift(
    tmp_path: Path,
    monkeypatch,
    mismatched_windows: bool,
) -> None:
    monkeypatch.setattr(workflow, "get_git_commit", lambda _: "test-commit")
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **kw: "")
    raw_source = tmp_path / "runs" / "v0_9" / "source"
    rows = []
    for mode in ("known", "latent_inferred"):
        path = raw_source / mode / "best.pt"
        path.parent.mkdir(parents=True)
        path.touch()
        rows.append(
            dict(
                seed=47, condition_mode=mode, operator_seed=701, checkpoint=str(path), route="joint"
            )
        )
    compact = tmp_path / "gpu_validation" / "v0_9" / "results" / "source"
    audit._write_json(compact / "completion.json", {"status": "PASS"})
    audit._write_json(
        compact / "evaluation" / "joint_summary.json",
        {"source_phase2_result": "phase2", "formal_run_count": 2, "runs": rows},
    )
    phase2 = tmp_path / "runs" / "v0_9" / "phase2"
    for suffix in (
        "context.pt",
        "backbone.pt",
        "data/controlled_cylinder_seed_47.pt",
        "seeds/seed_47/cache/adaptive_cache.pt",
    ):
        path = phase2 / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    audit._write_json(
        phase2 / "v0_8_handoff_audit.json",
        {
            "seeds": [
                {
                    "backbone_seed": 47,
                    "context_checkpoint": str(phase2 / "context.pt"),
                    "backbone_checkpoint": str(phase2 / "backbone.pt"),
                }
            ]
        },
    )

    def fake_audit(**kwargs):
        output, row = kwargs["output"], kwargs["row"]
        output.mkdir(parents=True)
        origin = 2 if mismatched_windows and row["condition_mode"] == "latent_inferred" else 1
        audit._write_json(output / "window_manifest.json", [{"origin": origin}])
        return {
            "cell": row,
            "window_count": 1,
            "horizons": [1],
            "summary": [],
            "validation_priorities": [
                dict(horizon=1, channel=c, reference=r, priority="MIXED")
                for c in ("field_all", "velocity_fluid", "pressure_fluid")
                for r in ("online", "teacher")
            ],
        }

    monkeypatch.setattr(workflow, "audit_checkpoint", fake_audit)
    args = argparse.Namespace(
        validation_id="audit", source_id="source", device="cpu", skip_tests=True
    )
    if mismatched_windows:
        with pytest.raises(ValueError, match="identical per-seed"):
            workflow.run_audit(args, root=tmp_path)
        result_path = compact.parent / "audit" / "partial_audit.json"
        result = json.loads(result_path.read_text())
        assert result["completed_cells"] == 1 and len(result["runs"]) == 1
    else:
        result = workflow.run_audit(args, root=tmp_path)
        assert result["status"] == "PASS" and result["completed_cells"] == 2
        assert result["scientific_acceptance"] == "NOT_EVALUATED"
        for name in ("audit.json", "completion.json", "report.md"):
            assert (compact.parent / "audit" / name).is_file()
