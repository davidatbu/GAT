from __future__ import annotations

import typing as T
import unittest

from Gat.configs import Config


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


class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._parent = ParentConfig(999, "999", ChildConfig(GrandChildConfig(0.999)))

    def tearDown(self) -> None:
        super().tearDown()

    def test_it(self) -> None:
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
            self._parent.as_flat_dict(),
            {
                "prop1": 999,
                "prop2": "999",
                "child_config.grand_child_config.prop1": 0.999,
                "child_config.prop1": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
