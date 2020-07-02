from __future__ import annotations

import typing as T
import unittest

from Gat.configs import Config
from Gat.testing_utils import debug_on


class GrandChildConfig(Config):
    def __init__(self, prop1: T.Optional[float]) -> None:
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


class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._parent = ParentConfig(999, "999", ChildConfig(GrandChildConfig(0.999)))
        self._parent_expected_flat_dict = {
            "prop1": 999,
            "prop2": "999",
            "child_config.grand_child_config.prop1": 0.999,
            "child_config.prop1": None,
        }

    def tearDown(self) -> None:
        super().tearDown()

    def test_as_dict(self) -> None:
        self.assertDictEqual(
            self._parent.as_dict(),
            {
                "prop1": 999,
                "prop2": "999",
                "child_config": {
                    "grand_child_config": {"prop1": 0.999,},
                    "prop1": None,
                },
            },
        )
        self.assertDictEqual(
            self._parent.as_flat_dict(), self._parent_expected_flat_dict,
        )

    @debug_on()
    def test_from_flat_dict(self) -> None:
        assert (
            ParentConfig.from_flat_dict(self._parent_expected_flat_dict) == self._parent
        )


if __name__ == "__main__":
    unittest.main()
