from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.optim import Adam

from jka_model.config import ProjectConfig, load_config, stable_config_hash
from jka_model.data import ChannelStandardizer, make_split_manifest
from jka_model.problems import create_problem_adapter
from jka_model.residual import (
    ResidualKoopmanModel,
    ResidualWindowDataset,
    analytical_mlp_parameter_count,
    build_closure,
    build_residual_cache,
    classify_memory_sweep,
    closure_metrics,
    compare_residual_memory_v0_7,
    corrected_latent_rollout,
    load_residual_cache,
    make_v0_7_synthetic_memory_cache,
    save_residual_cache,
    solve_parameter_matched_width,
    validate_sweep_provenance,
)
from jka_model.residual.checkpoint import load_residual_checkpoint, save_residual_checkpoint
from jka_model.training import (
    TrainStage,
    assert_optimizer_matches_trainable_params,
    configure_train_stage,
)
from jka_model.utils import Checkpoint, load_checkpoint
from train.train_v0_6 import initialize_v0_6_model
from train.train_v0_7 import _residual_scale_fingerprint, _training_residual_scale


def _cache_case():
    config = load_config("configs/v0_7/advection_diffusion_2d_cpu_smoke.yaml")
    adapter = create_problem_adapter(config)
    records = adapter.build_dataset(seed=config.training.seed)
    spec = adapter.build_problem_spec()
    manifest = make_split_manifest(records, config.data.split)
    normalizer = ChannelStandardizer(eps=config.data.normalization.eps).fit(records, manifest, spec)
    model = initialize_v0_6_model(config, device="cpu")
    model.requires_grad_(False)
    cache = build_residual_cache(
        ResidualKoopmanModel(
            model,
            build_closure(
                "zero",
                latent_dim=4,
                history=4,
                parameter_dim=3,
                hidden_dim=16,
                depth=2,
            ),
        ),
        records,
        normalizer,
        manifest,
        backbone_checkpoint_sha256="a" * 64,
        backbone_config_hash="b" * 64,
        data_fingerprint="c" * 64,
    )
    return config, model, cache


def test_residual_target_is_exact_online_koopman_difference_and_graph_free() -> None:
    _, model, cache = _cache_case()
    item = cache.trajectories[0]
    expected = item.latents[1:] - model.koopman_core.step(item.latents[:-1], item.dts)
    torch.testing.assert_close(item.residuals, expected)
    assert not item.residuals.requires_grad


def test_residual_target_never_uses_ema_target_encoder() -> None:
    config, model, before = _cache_case()
    adapter = create_problem_adapter(config)
    records = adapter.build_dataset(seed=config.training.seed)
    spec = adapter.build_problem_spec()
    manifest = make_split_manifest(records, config.data.split)
    normalizer = ChannelStandardizer(eps=config.data.normalization.eps).fit(records, manifest, spec)
    with torch.no_grad():
        model.target_encoder.projection.bias.add_(100.0)
    model.requires_grad_(False)
    after = build_residual_cache(
        ResidualKoopmanModel(
            model,
            build_closure("zero", latent_dim=4, history=4, parameter_dim=3, hidden_dim=16, depth=2),
        ),
        records,
        normalizer,
        manifest,
        backbone_checkpoint_sha256="a" * 64,
        backbone_config_hash="b" * 64,
        data_fingerprint="c" * 64,
    )
    for left, right in zip(before.trajectories, after.trajectories, strict=True):
        torch.testing.assert_close(left.residuals, right.residuals)


def test_residual_training_scale_uses_train_split_only_and_is_positive() -> None:
    _, _, cache = _cache_case()
    scale = _training_residual_scale(cache)
    train_residuals = torch.cat([item.residuals.float() for item in cache.select("train")])
    expected = train_residuals.square().mean(dim=0).sqrt()
    floor = max(
        float(train_residuals.square().mean().sqrt()) * 1e-3,
        torch.finfo(expected.dtype).eps,
    )
    torch.testing.assert_close(scale, expected.clamp_min(floor))
    assert torch.all(scale > 0)


def test_standardized_residual_metrics_use_supplied_train_scale() -> None:
    prediction = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
    target = torch.zeros_like(prediction)
    scale = torch.tensor([2.0, 1.0])
    metrics = closure_metrics(prediction, target, scale)
    expected = ((prediction - target) / scale).square().mean()
    assert metrics["standardized_mse"] == pytest.approx(float(expected))
    assert metrics["mse"] == pytest.approx(float((prediction - target).square().mean()))


def test_residual_stage_freezes_backbone_and_optimizer_owns_only_closure() -> None:
    config, backbone, _ = _cache_case()
    assert config.residual_closure
    closure = build_closure(
        "history", latent_dim=4, history=4, parameter_dim=3, hidden_dim=16, depth=2
    )
    model = ResidualKoopmanModel(backbone, closure)
    groups = configure_train_stage(model, TrainStage.RESIDUAL)
    assert groups["residual_head"] and not any(
        groups[name]
        for name in ("online_encoder", "koopman_core", "training_decoder", "target_encoder")
    )
    optimizer = Adam(model.residual_head.parameters())
    assert_optimizer_matches_trainable_params(model, optimizer)


def test_residual_windows_preserve_dt_alignment_and_never_cross_trajectories() -> None:
    _, _, cache = _cache_case()
    dataset = ResidualWindowDataset(cache, "train", history=4)
    sample = dataset[0]
    trajectory = next(
        item for item in cache.trajectories if item.trajectory_id == sample["trajectory_id"]
    )
    index = sample["target_index"]
    assert isinstance(index, int)
    torch.testing.assert_close(sample["history_z"][-1], trajectory.latents[index])
    torch.testing.assert_close(sample["next_dt"], trajectory.dts[index : index + 1])
    torch.testing.assert_close(sample["target"], trajectory.residuals[index])


def test_h1_matches_markovian_semantics() -> None:
    torch.manual_seed(19)
    history = build_closure(
        "history", latent_dim=4, history=1, parameter_dim=3, hidden_dim=16, depth=2
    )
    torch.manual_seed(19)
    instantaneous = build_closure(
        "instantaneous", latent_dim=4, history=1, parameter_dim=3, hidden_dim=16, depth=2
    )
    for left, right in zip(history.parameters(), instantaneous.parameters(), strict=True):
        torch.testing.assert_close(left, right)
    z = torch.randn(3, 1, 4)
    history_dts = torch.empty(3, 0)
    next_dt = torch.full((3, 1), 0.1)
    parameters = torch.randn(3, 3)
    torch.testing.assert_close(
        history(z, history_dts, next_dt, parameters),
        instantaneous(z, history_dts, next_dt, parameters),
    )


def test_history_window_alignment() -> None:
    _, _, cache = _cache_case()
    dataset = ResidualWindowDataset(cache, "train", history=4)
    sample = dataset[0]
    trajectory = next(
        item for item in cache.trajectories if item.trajectory_id == sample["trajectory_id"]
    )
    index = int(sample["target_index"])
    torch.testing.assert_close(sample["history_z"], trajectory.latents[index - 3 : index + 1])
    torch.testing.assert_close(sample["history_dts"], trajectory.dts[index - 3 : index])


def test_history_next_dt_alignment() -> None:
    _, _, cache = _cache_case()
    sample = ResidualWindowDataset(cache, "train", history=2)[0]
    trajectory = next(
        item for item in cache.trajectories if item.trajectory_id == sample["trajectory_id"]
    )
    index = int(sample["target_index"])
    torch.testing.assert_close(sample["next_dt"], trajectory.dts[index : index + 1])


def test_history_no_future_leakage() -> None:
    _, _, cache = _cache_case()
    dataset = ResidualWindowDataset(cache, "train", history=2)
    before = dataset[0]
    trajectory = next(
        item for item in cache.trajectories if item.trajectory_id == before["trajectory_id"]
    )
    index = int(before["target_index"])
    saved = trajectory.latents[index + 1].clone()
    trajectory.latents[index + 1].add_(1000)
    after = dataset[0]
    torch.testing.assert_close(before["history_z"], after["history_z"])
    trajectory.latents[index + 1].copy_(saved)


@pytest.mark.parametrize(
    "variant", ["zero", "linear", "instantaneous", "history", "shuffled_history"]
)
def test_all_closures_have_direct_delta_shape_and_are_deterministic(variant: str) -> None:
    torch.manual_seed(7)
    closure = build_closure(
        variant, latent_dim=4, history=4, parameter_dim=3, hidden_dim=16, depth=2
    )
    z = torch.randn(5, 4, 4)
    history_dts = torch.full((5, 3), 0.1)
    next_dt = torch.full((5, 1), 0.11)
    parameters = torch.randn(5, 3)
    first = closure(z, history_dts, next_dt, parameters)
    second = closure(z, history_dts, next_dt, parameters)
    assert first.shape == (5, 4)
    torch.testing.assert_close(first, second)


def test_correction_is_applied_after_exact_koopman_step() -> None:
    _, backbone, _ = _cache_case()
    closure = build_closure(
        "linear", latent_dim=4, history=4, parameter_dim=3, hidden_dim=16, depth=2
    )
    with torch.no_grad():
        closure.linear.weight.zero_()
        closure.linear.bias.fill_(0.25)
    model = ResidualKoopmanModel(backbone, closure)
    z = torch.randn(2, 4, 4)
    history_dts = torch.full((2, 3), 0.05)
    next_dt = torch.full((2, 1), 0.04)
    parameters = torch.randn(2, 3)
    corrected, base, correction = model.corrected_step(z, history_dts, next_dt, parameters)
    torch.testing.assert_close(base, model.koopman_core.step(z[:, -1], next_dt[:, 0]))
    torch.testing.assert_close(correction, torch.full_like(correction, 0.25))
    torch.testing.assert_close(corrected, base + correction)


def test_zero_initialized_closure_matches_koopman_prediction() -> None:
    _, backbone, _ = _cache_case()
    closure = build_closure(
        "history", latent_dim=4, history=2, parameter_dim=3, hidden_dim=16, depth=2
    )
    model = ResidualKoopmanModel(backbone, closure)
    history = torch.randn(2, 2, 4)
    corrected, base, correction = model.corrected_step(
        history,
        torch.full((2, 1), 0.1),
        torch.full((2, 1), 0.1),
        torch.randn(2, 3),
    )
    torch.testing.assert_close(correction, torch.zeros_like(correction))
    torch.testing.assert_close(corrected, base)


def test_closed_loop_feeds_predicted_state_back_into_history() -> None:
    _, backbone, _ = _cache_case()
    closure = build_closure(
        "linear", latent_dim=4, history=4, parameter_dim=3, hidden_dim=16, depth=2
    )
    with torch.no_grad():
        closure.linear.weight.zero_()
        closure.linear.bias.fill_(0.1)
    model = ResidualKoopmanModel(backbone, closure)
    history = torch.randn(1, 4, 4)
    history_dts = torch.full((1, 3), 0.05)
    future_dts = torch.full((1, 3), 0.04)
    parameters = torch.randn(1, 3)
    rollout, _, _ = corrected_latent_rollout(model, history, history_dts, future_dts, parameters)
    manual = history[:, -1]
    for step in range(3):
        manual = model.koopman_core.step(manual, future_dts[:, step]) + 0.1
        torch.testing.assert_close(rollout[:, step + 1], manual)


def test_history_rollout_uses_predicted_history() -> None:
    test_closed_loop_feeds_predicted_state_back_into_history()


def test_history_length_config_roundtrip() -> None:
    config = load_config("configs/v0_7/advection_diffusion_2d_cpu_smoke.yaml")
    restored = ProjectConfig.from_dict(config.to_dict())
    assert restored.memory_sweep is not None
    assert restored.memory_sweep.history_lengths == (1, 2, 4, 8)
    assert restored.memory_sweep.initialization_seeds == (101, 211, 307)


@pytest.mark.parametrize("variant", ["linear", "instantaneous", "history"])
def test_learned_closures_start_as_zero_correction(variant: str) -> None:
    closure = build_closure(
        variant, latent_dim=4, history=2, parameter_dim=3, hidden_dim=16, depth=2
    )
    prediction = closure(
        torch.randn(3, 2, 4),
        torch.full((3, 1), 0.1),
        torch.full((3, 1), 0.1),
        torch.randn(3, 3),
    )
    torch.testing.assert_close(prediction, torch.zeros_like(prediction))


def test_parameter_count_report() -> None:
    for history in (1, 2, 4, 8, 16):
        ordered = build_closure(
            "history", latent_dim=8, history=history, parameter_dim=3, hidden_dim=32, depth=2
        )
        control = build_closure(
            "instantaneous",
            latent_dim=8,
            history=history,
            parameter_dim=3,
            hidden_dim=32,
            depth=2,
        )
        ordered_count = sum(parameter.numel() for parameter in ordered.parameters())
        control_count = sum(parameter.numel() for parameter in control.parameters())
        assert abs(ordered_count - control_count) / ordered_count <= 0.05


def test_analytical_parameter_count_and_width_solver_do_not_consume_rng() -> None:
    before = torch.random.get_rng_state().clone()
    count = analytical_mlp_parameter_count(11, 16, 2, 4)
    width = solve_parameter_matched_width(
        instantaneous_input_dim=8,
        history_input_dim=11,
        history_hidden_dim=16,
        depth=2,
        output_dim=4,
    )
    after = torch.random.get_rng_state()
    assert count > 0 and width > 0
    torch.testing.assert_close(before, after)


def test_shuffled_history_preserves_current_state() -> None:
    _, _, cache = _cache_case()
    plain = ResidualWindowDataset(cache, "train", history=4)
    shuffled = ResidualWindowDataset(cache, "train", history=4, shuffle_history=True)
    torch.testing.assert_close(plain[0]["history_z"][-1], shuffled[0]["history_z"][-1])
    torch.testing.assert_close(plain[0]["next_dt"], shuffled[0]["next_dt"])
    torch.testing.assert_close(plain[0]["target"], shuffled[0]["target"])


def test_residual_cache_roundtrip_and_tamper_guard(tmp_path: Path) -> None:
    _, _, cache = _cache_case()
    path = tmp_path / "cache.pt"
    save_residual_cache(cache, path)
    restored = load_residual_cache(path)
    assert restored.fingerprint == cache.fingerprint
    payload = torch.load(path, weights_only=False)
    payload["trajectories"][0]["residuals"][0, 0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="fingerprint"):
        load_residual_cache(path)


def test_v0_7_standalone_checkpoint_roundtrip(tmp_path: Path) -> None:
    config, backbone, cache = _cache_case()
    closure = build_closure(
        "history", latent_dim=4, history=4, parameter_dim=3, hidden_dim=16, depth=2
    )
    model = ResidualKoopmanModel(backbone, closure)
    residual_scale = _training_residual_scale(cache)
    payload = {
        "schema_version": 7,
        "architecture_revision": "2.2",
        "project_version": "0.7.0",
        "train_stage": "residual",
        "epoch": 1,
        "global_step": 2,
        "closure_variant": "history",
        "backbone_data_seed": config.training.seed,
        "closure_init_seed": config.residual_training.initialization_seed,
        "history_length_steps": 4,
        "backbone_state": model.backbone_state_dict(),
        "closure_state": closure.state_dict(),
        "optimizer_state": {},
        "scheduler_state": {},
        "amp_scaler_state": None,
        "rng_state": {},
        "normalizer_state": cache.normalizer_state,
        "problem_spec": None,
        "config": config.to_dict(),
        "config_hash": config.stable_hash,
        "data_fingerprint": cache.data_fingerprint,
        "split_manifest": cache.split_manifest,
        "backbone_checkpoint_sha256": cache.backbone_checkpoint_sha256,
        "cache_fingerprint": cache.fingerprint,
        "residual_training_scale": residual_scale.tolist(),
        "residual_scale_fingerprint": _residual_scale_fingerprint(residual_scale),
        "git_commit": None,
    }
    path = tmp_path / "v07.pt"
    save_residual_checkpoint(payload, path)
    restored = load_residual_checkpoint(path)
    assert restored["closure_variant"] == "history"
    assert restored["cache_fingerprint"] == cache.fingerprint


def test_historical_schema_six_checkpoint_migrates_to_v0_7_runtime(tmp_path: Path) -> None:
    config = load_config("configs/v0_6/advection_diffusion_2d_cpu_smoke.yaml")
    payload = Checkpoint(
        train_stage=TrainStage.JEPA,
        epoch=1,
        global_step=1,
        online_model_state={"weight": torch.tensor([1.0])},
        target_model_state={"weight": torch.tensor([2.0])},
        config=config,
    ).to_payload()
    payload["schema_version"] = 6
    payload["project_version"] = "0.6.0"
    for key in ("residual_closure", "residual_training", "memory_sweep", "v0_7_evaluation"):
        payload["config"].pop(key)
    payload["config_hash"] = stable_config_hash(payload["config"])
    path = tmp_path / "historical-v06.pt"
    torch.save(payload, path)
    restored = load_checkpoint(path)
    assert restored.schema_version == 7
    assert restored.project_version == "0.7.0"
    assert restored.config == config


def test_version_scoped_synthetic_problem_exhibits_ordered_history_signal() -> None:
    cache = make_v0_7_synthetic_memory_cache()

    def matrices(split: str, history: bool):
        dataset = ResidualWindowDataset(cache, split, history=3)
        rows, targets = [], []
        for index in range(len(dataset)):
            sample = dataset[index]
            z = sample["history_z"]
            rows.append(z.flatten() if history else z[-1])
            targets.append(sample["target"])
        x = torch.stack(rows)
        x = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1)
        return x, torch.stack(targets)

    train_history, train_target = matrices("train", True)
    test_history, test_target = matrices("test", True)
    train_instant, _ = matrices("train", False)
    test_instant, _ = matrices("test", False)
    history_weight = torch.linalg.lstsq(train_history, train_target).solution
    instant_weight = torch.linalg.lstsq(train_instant, train_target).solution
    history_mse = (test_history @ history_weight - test_target).square().mean()
    instant_mse = (test_instant @ instant_weight - test_target).square().mean()
    assert history_mse < 1e-10
    assert history_mse < instant_mse * 1e-4


def _sweep_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    config = {
        "history_lengths": [1, 2, 4, 8],
        "effective_gain_fraction": 0.95,
        "material_relative_gain": 0.02,
        "plateau_relative_gain": 0.01,
        "parameter_match_tolerance": 0.05,
        "seed_consistency_fraction": 2 / 3,
        "initialization_seeds": [101, 211, 307],
        "strong_r2": 0.75,
        "moderate_r2": 0.4,
        "weak_r2": 0.05,
    }
    evaluation = {
        "min_residual_rms": 1e-6,
        "min_residual_significance": 0.01,
        "max_physics_degradation": 0.1,
        "max_closure_burden": 0.25,
        "formal_record_count": 117,
    }
    for seed in (47, 53, 59):
        provenance = {
            "backbone_checkpoint_sha256": f"backbone-{seed}",
            "backbone_config_hash": "config",
            "cache_fingerprint": f"cache-{seed}",
            "data_fingerprint": f"data-{seed}",
            "split_fingerprint": f"split-{seed}",
            "normalizer_fingerprint": f"normalizer-{seed}",
            "evaluation_trajectory_ids": [f"test-{seed}"],
        }
        for closure_seed in config["initialization_seeds"]:
            for history in (1, 2, 4, 8):
                variants = ["history", "instantaneous"]
                if history == 1:
                    variants += ["zero", "linear"]
                else:
                    variants += ["shuffled_history"]
                for variant in variants:
                    gain = 0.05 * min(history - 1, 3) if variant == "history" else 0.0
                    teacher = {
                        "mse": 0.2 - gain,
                        "standardized_mse": 0.2 - gain,
                        "normalized_rmse": 0.5 - gain,
                        "r2": 0.6 + gain,
                        "target_rms": 0.1,
                    }
                    records.append(
                        {
                            "phase": "v0.7",
                            "run_id": (f"seed_{seed}-closure_{closure_seed}-h{history}-{variant}"),
                            "git_commit": "test-commit",
                            "seed": seed,
                            "backbone_data_seed": seed,
                            "closure_initialization_seed": closure_seed,
                            "closure_init_seed": closure_seed,
                            "variant": variant,
                            "closure_family": variant,
                            "history_length_steps": history,
                            "history_length_physical_time": {"mean": 0.1 * (history - 1)},
                            "parameter_count": 1000,
                            "parameter_matched_control": variant == "instantaneous",
                            "history_shuffled": variant == "shuffled_history",
                            "teacher_forced": teacher,
                            "teacher_forced_validation": dict(teacher),
                            "residual_scale": [1.0],
                            "residual_scale_fingerprint": "scale-fingerprint",
                            "residual_structure": {
                                "validation": {
                                    "residual_significance": 0.2,
                                    "residual_rms": 0.1,
                                    "true_increment_rms": 0.22360679775,
                                },
                                "test": {
                                    "residual_significance": 0.2,
                                    "residual_rms": 0.1,
                                    "true_increment_rms": 0.22360679775,
                                },
                            },
                            "closed_loop": {
                                "8": {
                                    "latent_rmse": 0.4 - gain,
                                    "field_rmse": 0.5 - gain,
                                    "relative_l2": 0.3,
                                    "mass_drift": 0.001,
                                    "operator_mse": 1e-5,
                                    "closure_burden": 0.2,
                                    "closure_burden_by_step": [0.2] * 8,
                                }
                            },
                            "provenance": provenance,
                            "memory_sweep_config": config,
                            "v0_7_evaluation_config": evaluation,
                            "physics_limits": {
                                "max_relative_mass_drift": 0.01,
                                "max_operator_mse": 1e-4,
                            },
                            "target_encoder_used": False,
                            "rollout_uses_predicted_history": True,
                            "physics_used_for_training": False,
                            "source_file": (
                                f"seed-{seed}-init-{closure_seed}-{history}-{variant}.json"
                            ),
                        }
                    )
    return records


def _set_metrics(record: dict[str, object], *, mse: float, nrmse: float, r2: float) -> None:
    for name in ("teacher_forced_validation", "teacher_forced"):
        metrics = record[name]
        assert isinstance(metrics, dict)
        metrics.update(
            mse=mse,
            standardized_mse=mse,
            normalized_rmse=nrmse,
            r2=r2,
        )


def _r2_records() -> list[dict[str, object]]:
    records = _sweep_records()
    for record in records:
        variant = record["variant"]
        history = record["history_length_steps"]
        if history == 1 and variant in {"instantaneous", "history"}:
            _set_metrics(record, mse=0.02, nrmse=0.2, r2=0.85)
        elif variant != "zero":
            _set_metrics(record, mse=0.1, nrmse=0.5, r2=0.5)
    return records


def _r3_records() -> list[dict[str, object]]:
    records = _r2_records()
    for record in records:
        if record["variant"] == "history" and record["history_length_steps"] == 2:
            _set_metrics(record, mse=0.005, nrmse=0.1, r2=0.95)
            closed = record["closed_loop"]["8"]
            closed["field_rmse"] = 0.2
    return records


def test_history_sweep_same_backbone() -> None:
    result = validate_sweep_provenance(_sweep_records())
    assert result["same_backbone_data_split_normalizer_and_trajectories"]


def test_history_sweep_same_split() -> None:
    records = _sweep_records()
    records[1]["provenance"] = dict(records[1]["provenance"], split_fingerprint="changed")
    with pytest.raises(ValueError, match="split_fingerprint"):
        validate_sweep_provenance(records)


def test_memory_classification_schema() -> None:
    records = _sweep_records()
    provenance = validate_sweep_provenance(records)
    result = classify_memory_sweep(records, provenance)
    assert result["residual_learnability"] in {"STRONG", "MODERATE", "WEAK", "NONE"}
    assert result["closed_loop_utility"] in {"POSITIVE", "NEUTRAL", "NEGATIVE"}
    assert result["memory_class"] in {
        "MARKOVIAN",
        "SHORT_MEMORY",
        "LONG_MEMORY_CANDIDATE",
        "INCONCLUSIVE",
    }
    assert result["residual_route"] in {"R0", "R1", "R2", "R3", "INCONCLUSIVE"}
    assert result["physics_acceptance"] in {"PASS", "FAIL"}


def test_r0_route_for_negligible_residual() -> None:
    records = _sweep_records()
    for record in records:
        record["residual_structure"]["validation"]["residual_significance"] = 0.001
        record["residual_structure"]["test"]["residual_significance"] = 0.001
    result = classify_memory_sweep(records, validate_sweep_provenance(records))
    assert result["residual_route"] == "R0"


def test_r1_route_for_significant_unlearnable_residual() -> None:
    records = _sweep_records()
    result = classify_memory_sweep(records, validate_sweep_provenance(records))
    assert result["residual_route"] == "R1"


def test_r2_route_for_learnable_markovian_residual() -> None:
    records = _r2_records()
    result = classify_memory_sweep(records, validate_sweep_provenance(records))
    assert result["conditional_history_gain"] == "ABSENT"
    assert result["residual_route"] == "R2"


def test_r3_route_for_stable_conditional_history_gain() -> None:
    records = _r3_records()
    result = classify_memory_sweep(records, validate_sweep_provenance(records))
    assert result["conditional_history_gain"] == "PRESENT"
    assert result["locked_history_steps"] == 2
    assert result["residual_route"] == "R3"


def test_inconclusive_route_for_seed_disagreement() -> None:
    records = _r2_records()
    for record in records:
        seed = record["seed"]
        initialization = record["closure_initialization_seed"]
        mixed_seed_gain = seed == 59 and (
            (initialization == 101 and record["history_length_steps"] == 2)
            or (initialization == 211 and record["history_length_steps"] == 4)
        )
        if record["variant"] == "history" and (
            (seed == 47 and record["history_length_steps"] == 2) or mixed_seed_gain
        ):
            _set_metrics(record, mse=0.005, nrmse=0.1, r2=0.95)
    result = classify_memory_sweep(records, validate_sweep_provenance(records))
    assert result["conditional_history_gain"] == "INCONCLUSIVE"
    assert result["residual_route"] == "INCONCLUSIVE"


def test_test_metrics_cannot_change_validation_route_selection() -> None:
    records = _r2_records()
    before = classify_memory_sweep(records, validate_sweep_provenance(records))
    for record in records:
        metrics = record["teacher_forced"]
        metrics["mse"] = 1000.0 if record["variant"] != "zero" else 0.01
        metrics["standardized_mse"] = 1000.0 if record["variant"] != "zero" else 0.01
    after = classify_memory_sweep(records, validate_sweep_provenance(records))
    assert before["validation_residual_route"] == "R2"
    assert after["validation_residual_route"] == "R2"
    assert after["residual_route"] == "INCONCLUSIVE"


def test_primary_route_selection_uses_standardized_not_raw_mse() -> None:
    records = _r2_records()
    for record in records:
        if record["variant"] == "linear" and record["history_length_steps"] == 1:
            record["teacher_forced_validation"]["mse"] = 1e-12
    result = classify_memory_sweep(records, validate_sweep_provenance(records))
    assert result["validation_residual_route"] == "R2"
    assert all(
        selected["variant"] == "instantaneous"
        for selected in result["evidence"]["residual_structure"]["route_selected_models"].values()
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("mass_drift", 0.02, "physics_absolute_pass"),
        ("mass_drift", 0.005, "physics_noninferiority_pass"),
        ("closure_burden", 0.3, "closure_burden_pass"),
    ],
)
def test_physics_acceptance_requires_each_gate(field: str, value: float, expected: str) -> None:
    records = _r2_records()
    for record in records:
        if record["variant"] == "instantaneous" and record["history_length_steps"] == 1:
            record["closed_loop"]["8"][field] = value
    result = classify_memory_sweep(records, validate_sweep_provenance(records))
    assert result[expected] is False
    assert result["physics_acceptance"] == "FAIL"
    assert result["residual_route"] == "INCONCLUSIVE"


def test_duplicate_formal_run_identity_is_rejected() -> None:
    records = _sweep_records()
    records.append(dict(records[0]))
    with pytest.raises(ValueError, match="duplicated run identity"):
        validate_sweep_provenance(records)


def test_formal_matrix_rejects_unexpected_identity_even_at_exact_count() -> None:
    records = _sweep_records()
    records[0]["variant"] = "unexpected_control"
    with pytest.raises(ValueError, match="formal experiment matrix mismatch"):
        validate_sweep_provenance(records)


def test_model_selection_uses_validation_not_test_oracle() -> None:
    records = _sweep_records()
    target = next(
        record
        for record in records
        if record["seed"] == 47
        and record["closure_initialization_seed"] == 101
        and record["variant"] == "history"
        and record["history_length_steps"] == 8
    )
    target["teacher_forced_validation"]["normalized_rmse"] = 10.0
    target["teacher_forced"]["r2"] = 0.999
    target["closed_loop"]["8"]["field_rmse"] = 1e-9
    provenance = validate_sweep_provenance(records)
    result = classify_memory_sweep(records, provenance)
    selected = result["evidence"]["selected_models"]["seed_47_init_101"]
    assert selected["history_steps"] != 8
    memory_candidate = result["evidence"]["validation_selected_memory_candidate_by_repeat"][
        "seed_47_init_101"
    ]
    assert memory_candidate["history_steps"] != 8


def test_physics_failure_prevents_positive_utility() -> None:
    records = _r2_records()
    for record in records:
        if record["variant"] == "instantaneous" and record["history_length_steps"] == 1:
            record["closed_loop"]["8"]["field_rmse"] = 0.2
            record["closed_loop"]["8"]["mass_drift"] = 0.02
    provenance = validate_sweep_provenance(records)
    result = classify_memory_sweep(records, provenance)
    assert result["closed_loop_utility"] == "NEGATIVE"


def test_memory_comparison_writes_required_outputs(tmp_path: Path) -> None:
    for index, record in enumerate(_sweep_records()):
        destination = (
            tmp_path
            / "session"
            / "seeds"
            / f"seed_{record['seed']}"
            / "evaluation"
            / f"record_{index}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record), encoding="utf-8")
    for seed in (47, 53, 59):
        diagnostics = (
            tmp_path / "session" / "seeds" / f"seed_{seed}" / "cache" / "residual_diagnostics.json"
        )
        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.write_text(
            json.dumps(
                {
                    "splits": {
                        "validation": {
                            "rms": 0.1,
                            "normalized_rms_by_true_increment": 0.2,
                            "acf": {"0": [0.4, 0.2, 0.1], "1": [0.3, 0.1, 0.0]},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
    compare_residual_memory_v0_7(tmp_path / "session", tmp_path / "results")
    assert (tmp_path / "results/evaluation/history_sweep.csv").is_file()
    assert (tmp_path / "results/evaluation/memory_classification.json").is_file()
    assert (tmp_path / "results/evaluation/residual_structure_assessment.json").is_file()
    assert (tmp_path / "results/plots/residual_magnitude.png").is_file()
    assert (tmp_path / "results/plots/history_length_residual_error.png").is_file()
    assert (tmp_path / "results/plots/history_length_rollout_error.png").is_file()
    assert (tmp_path / "results/plots/history_gain_by_backbone_seed.png").is_file()
    assert (tmp_path / "results/plots/residual_autocorrelation_vs_lag.png").is_file()
    assert (tmp_path / "results/reports/residual_decision_report.md").is_file()
    assert (tmp_path / "results/reports/v0_8_route_recommendation.md").is_file()
