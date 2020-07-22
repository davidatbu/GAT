from __future__ import annotations

import dataclasses
import logging
import sys
import tempfile
import typing as T
import unittest
from pathlib import Path

from Gat.data.cacheable import Cacheable
from Gat.data.cacheable import CacheableSettings
from Gat.data.cacheable import ThingCacher

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


class DigitsOfPi(Cacheable):
    @staticmethod
    def thing_cachers() -> T.Tuple[ThingCacher]:
        return (ThingCacher("digits"),)

    @dataclasses.dataclass
    class Settings(CacheableSettings):
        start_digit: int
        end_digit: int

        def validate(self) -> None:
            assert self.end_digit > self.start_digit
            assert self.start_digit >= 0

    def __init__(
        self, settings: DigitsOfPi.Settings, things: T.Tuple[T.Tuple[int, ...]]
    ) -> None:
        self.settings = settings
        (self.digits,) = things

    @staticmethod
    def make_cached_things(settings: DigitsOfPi.Settings) -> T.Tuple[T.Tuple[int, ...]]:  # type: ignore[override]
        # Apprently, all the digits of pi are just 0
        settings.validate()
        digits = (0,) * (settings.end_digit - settings.start_digit)
        things = (digits,)
        return things

    def say_digits(self) -> None:
        print(
            f"The digits of pi, from {self.settings.start_digit}-th place to the {self.settings.end_digit}-th"
            f" place are: {self.digits}"
        )


class TestCacheable(unittest.TestCase):
    def setUp(self) -> None:
        self._settings = DigitsOfPi.Settings(1, 10)
        self._digits = (0,) * (self._settings.end_digit - self._settings.start_digit)
        self._temp_dir = Path(tempfile.mkdtemp())
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    def test_creation(self) -> None:
        digits_of_pi = DigitsOfPi(self._settings, (self._digits,))
        digits_of_pi.say_digits()
        print(__name__)

    def test_caching(self) -> None:
        digits_of_pi = DigitsOfPi.try_from_cache(self._settings, self._temp_dir)
        digits_of_pi.say_digits()

        digits_of_pi_again = DigitsOfPi.try_from_cache(self._settings, self._temp_dir)
        digits_of_pi_again.say_digits()
