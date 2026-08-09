from __future__ import annotations

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
