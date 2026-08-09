from __future__ import annotations

import inspect

import pytest
import torch

from jka_model.config import KoopmanAutoencoderConfig, RepresentationLossConfig
from jka_model.contracts import ProblemBatch
from jka_model.losses import (
    compute_representation_loss,
    koopman_multi_step_loss,
    koopman_one_step_loss,
    reconstruction_loss,
    variance_loss,
)
from jka_model.models import (
    ContinuousKoopmanCore,
    KoopmanAutoencoder,
    KoopmanEncoder,
    TrainingDecoder,
)
from jka_model.training import (
    TrainStage,
    assert_optimizer_matches_trainable_params,
    configure_train_stage,
)
from jka_model.training.koopman_representation import initialize_koopman_autoencoder


def _model() -> KoopmanAutoencoder:
    return initialize_koopman_autoencoder(
        KoopmanAutoencoderConfig(
            observation_dim=5,
            latent_dim=2,
            hidden_dim=12,
            encoder_hidden_layers=0,
            decoder_hidden_layers=2,
        ),
        seed=3,
        init_scale=0.05,
        dtype=torch.float64,
    )


def _batch() -> ProblemBatch:
    random = torch.Generator().manual_seed(4)
    context = torch.randn((6, 2, 5), generator=random, dtype=torch.float64)
    future = torch.randn((6, 3, 5), generator=random, dtype=torch.float64)
    return ProblemBatch(
        context_states_raw=context,
        future_states_raw=future,
        context_states_model=context.clone(),
        future_states_model=future.clone(),
        history_dts=torch.full((6, 1), 0.1, dtype=torch.float64),
        future_dts=torch.full((6, 3), 0.1, dtype=torch.float64),
    )


def test_encoder_output_shape() -> None:
    encoder = KoopmanEncoder(5, 2, hidden_layers=1, dtype=torch.float64)
    assert encoder(torch.zeros((4, 5), dtype=torch.float64)).shape == (4, 2)


def test_latent_dim_validation() -> None:
    encoder = KoopmanEncoder(5, 2, hidden_layers=1, dtype=torch.float64)
    with pytest.raises(ValueError, match="positive"):
        KoopmanEncoder(5, 0)
    with pytest.raises(ValueError, match="input"):
        encoder(torch.zeros((4, 4), dtype=torch.float64))


def test_decoder_output_shape() -> None:
    decoder = TrainingDecoder(2, 5, hidden_dim=8, dtype=torch.float64)
    assert decoder(torch.zeros((4, 2), dtype=torch.float64)).shape == (4, 5)


def test_decoder_has_no_time_dynamics() -> None:
    decoder = TrainingDecoder(2, 5, hidden_dim=8, dtype=torch.float64)
    assert tuple(inspect.signature(TrainingDecoder.forward).parameters) == ("self", "z_k")
    assert not hasattr(decoder, "step") and not hasattr(decoder, "rollout")


def test_encoder_decoder_gradient_flow() -> None:
    model = _model()
    loss = model.reconstruct(torch.randn((8, 5), dtype=torch.float64)).square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.encoder.parameters())
    assert all(parameter.grad is not None for parameter in model.decoder.parameters())


def test_encoder_decoder_reject_dtype_mismatch_and_nonfinite_input() -> None:
    model = _model()
    with pytest.raises(ValueError, match="dtype/device"):
        model.encode(torch.zeros((2, 5), dtype=torch.float32))
    with pytest.raises(ValueError, match="finite"):
        model.encode(torch.full((2, 5), float("nan"), dtype=torch.float64))
    with pytest.raises(ValueError, match="dtype/device"):
        model.decode(torch.zeros((2, 2), dtype=torch.float32))
    with pytest.raises(ValueError, match="finite"):
        model.decode(torch.full((2, 2), float("inf"), dtype=torch.float64))


def _exact_latent_case() -> tuple[ContinuousKoopmanCore, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.tensor([[-0.1, -1.2], [1.2, -0.1]], dtype=torch.float64)
    core = ContinuousKoopmanCore(2, generator=generator, trainable=False, dtype=torch.float64)
    z0 = torch.tensor([[1.0, 0.0], [0.2, -0.7]], dtype=torch.float64)
    dts = torch.tensor([[0.1, 0.2, 0.15], [0.05, 0.12, 0.2]], dtype=torch.float64)
    exact = core.rollout(z0, dts)[:, 1:]
    return core, z0, dts, exact


def test_koopman_one_step_loss_zero_on_exact_case() -> None:
    core, z0, dts, exact = _exact_latent_case()
    assert koopman_one_step_loss(core, z0, exact[:, 0], dts[:, 0]) < 1e-28


def test_multi_step_loss_zero_on_exact_case() -> None:
    core, z0, dts, exact = _exact_latent_case()
    assert koopman_multi_step_loss(core, z0, exact, dts) < 1e-28


def test_multi_step_prediction_is_closed_loop() -> None:
    core, z0, dts, exact = _exact_latent_case()
    corrupted = exact.clone()
    corrupted[:, 0] += 10.0
    expected = (core.rollout(z0, dts)[:, 1:] - corrupted).square().mean()
    torch.testing.assert_close(
        koopman_multi_step_loss(core, z0, corrupted, dts), expected
    )


def test_reconstruction_loss_zero_on_identity_case() -> None:
    state = torch.randn((5, 3), dtype=torch.float64)
    assert reconstruction_loss(state, state) == 0


def test_variance_loss_penalizes_collapsed_latent() -> None:
    collapsed = torch.ones((8, 2), dtype=torch.float64)
    assert variance_loss(collapsed, min_std=0.2) > 0


def test_variance_loss_small_for_noncollapsed_latent() -> None:
    noncollapsed = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    assert variance_loss(noncollapsed, min_std=0.2) == 0


def test_koopman_stage_optimizer_parameter_ownership() -> None:
    model = _model()
    ownership = configure_train_stage(model, TrainStage.KOOPMAN)
    assert ownership == {
        "koopman_encoder": True,
        "koopman_core": True,
        "training_decoder": True,
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    assert_optimizer_matches_trainable_params(model, optimizer)


def test_backward_updates_encoder_decoder_and_A() -> None:
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    losses = compute_representation_loss(
        model, _batch(), RepresentationLossConfig(lambda_spec=0.0)
    )
    losses.total.backward()
    groups = {
        "encoder": list(model.encoder.parameters()),
        "decoder": list(model.decoder.parameters()),
        "generator": [model.core.A],
    }
    for parameters in groups.values():
        gradients = [parameter.grad for parameter in parameters]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
        finite_gradients = [gradient for gradient in gradients if gradient is not None]
        assert torch.stack([gradient.norm() for gradient in finite_gradients]).norm() > 0
    optimizer.step()
    after = dict(model.named_parameters())
    for group_name, parameters in groups.items():
        parameter_ids = {id(parameter) for parameter in parameters}
        names = {
            name for name, parameter in after.items() if id(parameter) in parameter_ids
        }
        assert names, f"missing names for {group_name}"
        assert any(not torch.equal(before[name], after[name]) for name in names)


def _variable_dt_rollout() -> tuple[KoopmanAutoencoder, torch.Tensor, torch.Tensor]:
    model = _model()
    initial = torch.randn((3, 5), dtype=torch.float64)
    dts = torch.tensor(
        [[0.1, 0.2, 0.05], [0.05, 0.1, 0.2], [0.2, 0.05, 0.1]],
        dtype=torch.float64,
    )
    return model, initial, dts


def test_learned_model_rollout_shape() -> None:
    model, initial, dts = _variable_dt_rollout()
    latent, decoded = model.rollout_decoded(initial, dts)
    assert latent.shape == (3, 4, 2)
    assert decoded.shape == (3, 4, 5)


def test_learned_model_rollout_uses_prediction_history() -> None:
    model, initial, dts = _variable_dt_rollout()
    latent, _ = model.rollout_decoded(initial, dts)
    torch.testing.assert_close(latent, model.core.rollout(model.encode(initial), dts))


def test_decoded_rollout_finite() -> None:
    model, initial, dts = _variable_dt_rollout()
    _, decoded = model.rollout_decoded(initial, dts)
    assert torch.isfinite(decoded).all()


def test_variable_dt_learned_rollout() -> None:
    model, initial, dts = _variable_dt_rollout()
    actual = model.rollout_latent(initial, dts)
    constant_dt = model.rollout_latent(initial, torch.full_like(dts, 0.1))
    assert not torch.allclose(actual, constant_dt)
