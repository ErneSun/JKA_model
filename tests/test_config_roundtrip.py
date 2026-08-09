from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jka_model.config import (
    DataConfig,
    ProjectConfig,
    SplitConfig,
    ToyAdvectionDiffusionConfig,
    load_config,
    save_config,
    stable_config_hash,
)
from jka_model.contracts import DtMode


def test_config_roundtrip_and_stable_hash(tmp_path, toy_config: ProjectConfig) -> None:
    destination = tmp_path / "resolved.yaml"
    save_config(toy_config, destination)
    restored = load_config(destination)
    assert restored == toy_config
    assert restored.stable_hash == toy_config.stable_hash
    assert stable_config_hash(restored.to_dict()) == toy_config.stable_hash


def test_config_hash_is_key_order_independent(toy_config: ProjectConfig) -> None:
    payload = toy_config.to_dict()
    reversed_payload = dict(reversed(list(payload.items())))
    assert stable_config_hash(payload) == stable_config_hash(reversed_payload)


def test_config_rejects_unknown_fields(tmp_path, toy_config: ProjectConfig) -> None:
    payload = toy_config.to_dict()
    payload["training"]["silent_default"] = True
    destination = tmp_path / "invalid.yaml"
    destination.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown training config"):
        load_config(destination)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"history": 1}, "history"),
        ({"horizon": 0}, "horizon"),
    ],
)
def test_v0_2_config_rejects_invalid_window_lengths(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DataConfig(
            problem_name="test",
            dt_mode=DtMode.CONSTANT,
            constant_dt=0.1,
            **kwargs,
        )


def test_v0_2_config_rejects_invalid_split_and_toy_physics() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        SplitConfig(train=0.8, validation=0.2, test=0.2)
    with pytest.raises(ValueError, match="diffusivity"):
        ToyAdvectionDiffusionConfig(nu_min=-0.1)
    with pytest.raises(ValueError, match="nx"):
        ToyAdvectionDiffusionConfig(nx=4)


def test_config_roundtrip_v0_3(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "v0_3_smoke.yaml")
    destination = tmp_path / "v0_3.yaml"
    save_config(config, destination)
    restored = load_config(destination)
    assert restored == config
    assert restored.koopman is not None and restored.koopman.state_dim == 2
    assert restored.oscillator is not None and restored.oscillator.variable_dt


def test_v0_3_config_rejects_data_oscillator_mismatch(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "v0_3_smoke.yaml")
    payload = config.to_dict()
    payload["data"]["dt_mode"] = "constant"
    payload["data"]["constant_dt"] = config.oscillator.base_dt if config.oscillator else 0.04
    destination = tmp_path / "mismatched-v0-3.yaml"
    destination.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="dt_mode must match"):
        load_config(destination)


def test_config_roundtrip_v0_4(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "v0_4_smoke.yaml")
    destination = tmp_path / "v0_4.yaml"
    save_config(config, destination)
    restored = load_config(destination)
    assert restored == config
    assert restored.autoencoder is not None
    assert restored.autoencoder.observation_dim == 5
    assert restored.autoencoder.latent_dim == 2
    assert restored.representation_evaluation is not None
    assert restored.representation_evaluation.min_alignment_r2 == 0.98


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("autoencoder", "observation_dim", 4, "observation_dim=5"),
        ("data", "normalization", {"kind": "external", "eps": 1e-6}, "standard"),
        ("known_latent", "num_steps", 5, "too short"),
        ("representation_evaluation", "rollout_horizon", 81, "exceeds"),
    ],
)
def test_v0_4_config_rejects_cross_section_mismatches(
    tmp_path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    payload = load_config(root / "configs" / "v0_4_smoke.yaml").to_dict()
    payload[section][field] = value
    destination = tmp_path / f"bad-{section}-{field}.yaml"
    destination.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(destination)


def test_v0_4_constant_dt_must_match_generator_config(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    payload = load_config(root / "configs" / "v0_4_smoke.yaml").to_dict()
    payload["known_latent"]["variable_dt"] = False
    payload["known_latent"]["dt_jitter"] = 0.0
    payload["data"]["dt_mode"] = "constant"
    payload["data"]["constant_dt"] = 0.2
    destination = tmp_path / "bad-v0-4-constant-dt.yaml"
    destination.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="constant_dt must match"):
        load_config(destination)
