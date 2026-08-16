from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch.optim import SGD, Adam

from jka_model.config import JEPALossConfig, load_config, stable_config_hash
from jka_model.data import (
    ChannelStandardizer,
    TrajectoryWindowDataset,
    collate_problem_batches,
    make_split_manifest,
    select_split,
)
from jka_model.evaluation import near_identity_diagnostic
from jka_model.losses import compute_field_jepa_loss
from jka_model.models import FieldJEPAKoopmanModel, normalized_parameter_distance
from jka_model.problems import create_problem_adapter
from jka_model.training import TrainStage, configure_train_stage
from jka_model.training.ema import EMATracker
from jka_model.utils import Checkpoint, load_checkpoint, save_checkpoint
from train.train_v0_6 import (
    initialize_v0_6_model,
    load_v0_5_initialization,
    update_ema_after_optimizer_result,
)


def _case() -> tuple[object, object, object, object, object]:
    config = load_config("configs/v0_6/advection_diffusion_2d_cpu_smoke.yaml")
    adapter = create_problem_adapter(config)
    records = adapter.build_dataset(seed=config.training.seed)
    spec = adapter.build_problem_spec()
    manifest = make_split_manifest(records, config.data.split)
    normalizer = ChannelStandardizer(eps=config.data.normalization.eps).fit(records, manifest, spec)
    dataset = TrajectoryWindowDataset(
        select_split(records, manifest, "train"),
        history=config.data.history,
        horizon=config.data.horizon,
        normalizer=normalizer,
    )
    batch = collate_problem_batches([dataset[0], dataset[1]]).to(dtype=torch.float32)
    return config, adapter, spec, normalizer, batch


def _loss(model: object, jepa: JEPALossConfig | None = None) -> object:
    config, adapter, spec, normalizer, batch = _case()
    assert config.field_loss and config.jepa_loss
    return compute_field_jepa_loss(
        model,
        batch,
        normalizer,
        spec,
        config.field_loss,
        config.jepa_loss if jepa is None else jepa,
        adapter.build_physics_constraints(),
        physics_scale=1.0,
    )


def test_target_hard_sync() -> None:
    config, *_ = _case()
    model = initialize_v0_6_model(config, device="cpu")
    assert normalized_parameter_distance(model) == 0.0
    with torch.no_grad():
        next(model.online_encoder.parameters()).add_(1.0)
    assert normalized_parameter_distance(model) > 0
    model.hard_sync_target()
    assert normalized_parameter_distance(model) == 0.0


def test_target_frozen() -> None:
    config, *_ = _case()
    model = initialize_v0_6_model(config, device="cpu")
    model.train()
    assert not model.target_encoder.training
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())


def test_target_not_optimizer() -> None:
    config, *_ = _case()
    model = initialize_v0_6_model(config, device="cpu")
    optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert not optimizer_ids.intersection(id(p) for p in model.target_encoder.parameters())


def test_target_no_grad_graph() -> None:
    config, _, _, _, batch = _case()
    model = initialize_v0_6_model(config, device="cpu")
    target = model.encode_target(batch.future_states_model)
    assert not target.requires_grad and target.grad_fn is None


def test_ema_formula() -> None:
    config, *_ = _case()
    model = initialize_v0_6_model(config, device="cpu")
    old_target = [p.detach().clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        for parameter in model.online_encoder.parameters():
            parameter.add_(0.25)
    online = [p.detach().clone() for p in model.online_encoder.parameters()]
    tracker = EMATracker(config.ema, total_updates=2)
    tau = tracker.update_after_optimizer(model)
    for before, source, after in zip(
        old_target, online, model.target_encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(after, tau * before + (1 - tau) * source)


def test_ema_not_before_optimizer() -> None:
    config, *_ = _case()
    model = initialize_v0_6_model(config, device="cpu")
    tracker = EMATracker(config.ema, total_updates=2)
    before = copy.deepcopy(model.target_encoder.state_dict())
    next(model.online_encoder.parameters()).sum().backward()
    assert tracker.update_count == 0
    for name, value in model.target_encoder.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_ema_update_after_optimizer() -> None:
    config, *_ = _case()
    model = initialize_v0_6_model(config, device="cpu")
    tracker = EMATracker(config.ema, total_updates=2)
    optimizer = SGD(model.online_encoder.parameters(), lr=0.1)
    next(model.online_encoder.parameters()).sum().backward()
    optimizer.step()
    tracker.update_after_optimizer(model)
    assert tracker.update_count == 1 and normalized_parameter_distance(model) > 0


def test_ema_once_per_optimizer_step() -> None:
    config, *_ = _case()
    model = initialize_v0_6_model(config, device="cpu")
    tracker = EMATracker(config.ema, total_updates=2)
    tracker.update_after_optimizer(model)
    tracker.update_after_optimizer(model)
    with pytest.raises(RuntimeError):
        tracker.update_after_optimizer(model)


def test_amp_skipped_optimizer_skips_ema() -> None:
    config, *_ = _case()
    model = initialize_v0_6_model(config, device="cpu")
    tracker = EMATracker(config.ema, total_updates=2)
    before = copy.deepcopy(model.target_encoder.state_dict())
    tau = update_ema_after_optimizer_result(model, tracker, optimizer_updated=False)
    assert tau is None and tracker.update_count == 0
    for name, value in model.target_encoder.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_koopman_loss_uses_online_future() -> None:
    config, *_ = _case()
    model = initialize_v0_6_model(config, device="cpu")
    before = _loss(model)
    with torch.no_grad():
        model.target_encoder.projection.bias.add_(3.0)
    target_changed = _loss(model)
    torch.testing.assert_close(target_changed.v0_5.total, before.v0_5.total)


def test_jepa_loss_uses_ema_future() -> None:
    config, *_ = _case()
    model = initialize_v0_6_model(config, device="cpu")
    before = _loss(model)
    with torch.no_grad():
        model.target_encoder.projection.bias.add_(3.0)
    target_changed = _loss(model)
    assert not torch.isclose(target_changed.jepa_one_step, before.jepa_one_step)
    with torch.no_grad():
        model.online_encoder.projection.bias.add_(1.0)
    online_changed = _loss(model)
    assert not torch.isclose(online_changed.v0_5.koopman_one_step, before.v0_5.koopman_one_step)


def _backward_case() -> FieldJEPAKoopmanModel:
    config, *_ = _case()
    model = initialize_v0_6_model(config, device="cpu")
    losses = _loss(model)
    losses.total.backward()
    return model


def test_jepa_target_stop_gradient() -> None:
    model = _backward_case()
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_jepa_gradient_to_encoder() -> None:
    model = _backward_case()
    assert any(p.grad is not None for p in model.online_encoder.parameters())


def test_jepa_gradient_to_A() -> None:
    model = _backward_case()
    assert model.koopman_core.A.grad is not None


def test_jepa_no_gradient_to_target() -> None:
    model = _backward_case()
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_multistep_jepa_closed_loop() -> None:
    config, _, _, _, batch = _case()
    model = initialize_v0_6_model(config, device="cpu")
    losses = _loss(model)
    predicted = model.koopman_core.rollout(
        model.encode(batch.context_states_model[:, -1]), batch.future_dts
    )[:, 1:]
    target = model.encode_target(batch.future_states_model)
    expected = (predicted[:, 1:] - target[:, 1:]).square().mean()
    torch.testing.assert_close(losses.jepa_multi_step, expected)


def test_near_identity_diagnostic() -> None:
    generator = torch.zeros(3, 3)
    diagnostic = near_identity_diagnostic(generator, torch.tensor([0.01, 0.1, 1.0]))
    assert all(item["relative_frobenius"] == 0.0 for item in diagnostic.values())


def test_no_jepa_control() -> None:
    config, *_ = _case()
    model = initialize_v0_6_model(config, device="cpu")
    losses = _loss(model, JEPALossConfig(0.0, 0.0))
    torch.testing.assert_close(losses.total, losses.v0_5.total, rtol=0, atol=0)
    assert losses.jepa_one_step.item() == losses.jepa_multi_step.item() == 0.0


def test_v0_6_resume_preserves_target(tmp_path: Path) -> None:
    config, *_ = _case()
    model = initialize_v0_6_model(config, device="cpu")
    with torch.no_grad():
        model.target_encoder.projection.bias.add_(0.5)
    tracker = EMATracker(config.ema, total_updates=4)
    tracker.update_after_optimizer(model)
    path = tmp_path / "v06.pt"
    save_checkpoint(
        Checkpoint(
            train_stage=TrainStage.JEPA,
            epoch=1,
            global_step=1,
            optimizer_update_step=1,
            online_model_state=model.online_state_dict(),
            target_model_state=model.target_encoder.state_dict(),
            ema_state=tracker.state_dict(),
            config=config,
        ),
        path,
    )
    restored = load_checkpoint(path)
    reloaded = initialize_v0_6_model(config, device="cpu")
    reloaded.load_online_state_dict(restored.online_model_state)
    reloaded.target_encoder.load_state_dict(restored.target_model_state)
    for left, right in zip(
        model.target_encoder.parameters(), reloaded.target_encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(left, right)
    assert EMATracker.from_state_dict(restored.ema_state).update_count == 1


def _deterministic_step(model: FieldJEPAKoopmanModel, optimizer: Adam, tracker: EMATracker) -> None:
    optimizer.zero_grad(set_to_none=True)
    objective = sum(
        parameter.square().sum() for parameter in model.parameters() if parameter.requires_grad
    )
    objective.backward()
    optimizer.step()
    tracker.update_after_optimizer(model)


def test_v0_6_resume_one_more_step_exact(tmp_path: Path) -> None:
    config, *_ = _case()
    continuous = initialize_v0_6_model(config, device="cpu")
    continuous_optimizer = Adam((p for p in continuous.parameters() if p.requires_grad), lr=1.0e-3)
    continuous_ema = EMATracker(config.ema, total_updates=3)
    _deterministic_step(continuous, continuous_optimizer, continuous_ema)
    path = tmp_path / "exact-resume.pt"
    save_checkpoint(
        Checkpoint(
            train_stage=TrainStage.JEPA,
            epoch=1,
            global_step=1,
            optimizer_update_step=1,
            online_model_state=continuous.online_state_dict(),
            target_model_state=continuous.target_encoder.state_dict(),
            optimizer_state=continuous_optimizer.state_dict(),
            ema_state=continuous_ema.state_dict(),
            config=config,
        ),
        path,
    )
    restored = load_checkpoint(path)
    resumed = initialize_v0_6_model(config, device="cpu")
    resumed.load_online_state_dict(restored.online_model_state)
    resumed.target_encoder.load_state_dict(restored.target_model_state)
    resumed_optimizer = Adam((p for p in resumed.parameters() if p.requires_grad), lr=1.0e-3)
    resumed_optimizer.load_state_dict(restored.optimizer_state)
    resumed_ema = EMATracker.from_state_dict(restored.ema_state)
    _deterministic_step(continuous, continuous_optimizer, continuous_ema)
    _deterministic_step(resumed, resumed_optimizer, resumed_ema)
    for name, value in continuous.online_state_dict().items():
        torch.testing.assert_close(value, resumed.online_state_dict()[name], rtol=0, atol=0)
    for name, value in continuous.target_encoder.state_dict().items():
        torch.testing.assert_close(value, resumed.target_encoder.state_dict()[name], rtol=0, atol=0)
    assert continuous_ema.state_dict() == resumed_ema.state_dict()


def test_v0_5_init_hard_syncs_target(tmp_path: Path) -> None:
    config, *_ = _case()
    source_model = initialize_v0_6_model(config, device="cpu")
    old_config = config.to_dict()
    old_config["project_version"] = "0.5.0"
    for name in ("jepa_loss", "ema", "v0_6_evaluation"):
        old_config.pop(name)
    path = tmp_path / "legacy-v05.pt"
    torch.save(
        {
            "schema_version": 5,
            "project_version": "0.5.0",
            "architecture_revision": "2.2",
            "train_stage": "koopman",
            "config": old_config,
            "config_hash": stable_config_hash(old_config),
            "online_model_state": source_model.online_state_dict(),
        },
        path,
    )
    payload = load_v0_5_initialization(path, config)
    target = initialize_v0_6_model(config, device="cpu")
    target.load_online_state_dict(payload["online_model_state"])
    target.hard_sync_target()
    assert normalized_parameter_distance(target) == 0.0


def test_stage_contract_excludes_target() -> None:
    config, *_ = _case()
    model = initialize_v0_6_model(config, device="cpu")
    ownership = configure_train_stage(model, TrainStage.JEPA)
    assert ownership == {
        "online_encoder": True,
        "koopman_core": True,
        "training_decoder": True,
        "target_encoder": False,
    }
