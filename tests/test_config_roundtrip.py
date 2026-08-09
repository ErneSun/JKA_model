from __future__ import annotations

import pytest
import yaml

from jka_model.config import ProjectConfig, load_config, save_config, stable_config_hash


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

