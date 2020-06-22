from __future__ import annotations

import typing as T

import pytest

from Gat.config import Config


class GrandChildConfig(Config):
    def __init__(self, prop1: T.Optional[float] = None) -> None:
        self.prop1 = prop1


class ChildConfig(Config):
    def __init__(
        self, grand_child_config: GrandChildConfig, prop1: T.Optional[float] = None
    ) -> None:
        self.grand_child_config = grand_child_config
        self.prop1 = prop1


class ParentConfig(Config):
    def __init__(self, prop1: int, prop2: str, child_config: ChildConfig):
        self.prop1 = prop1
        self.prop2 = prop2
        self.child_config = child_config


@pytest.fixture
def parent_config() -> ParentConfig:
    return ParentConfig(999, "999", ChildConfig(GrandChildConfig(0.999)))


def test_it(parent_config: ParentConfig) -> None:
    assert parent_config.as_dict() == {
        "prop1": 999,
        "prop2": "999",
        "child_config": {"grand_child_config": {"prop1": 0.999,}, "prop1": None},
    }
    assert parent_config.as_flat_dict() == {
        "prop1": 999,
        "prop2": "999",
        "child_config.grand_child_config.prop1": 0.999,
        "child_config.prop1": None,
    }

    with pytest.raises(Exception):
        parent_config.asdfadsf = 4
