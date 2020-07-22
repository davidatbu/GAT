from __future__ import annotations

import abc
import dataclasses
import hashlib
import logging
import typing as T
from pathlib import Path

import torch  # type: ignore

logger = logging.getLogger()


__all__ = ["CachingTool", "TorchCachingTool", "Cacheable"]


class CachingTool(abc.ABC):
    """A class to abstract away any caching tool."""

    @staticmethod
    @abc.abstractmethod
    def load(file_: Path) -> T.Any:
        pass

    @staticmethod
    @abc.abstractmethod
    def save(obj: T.Any, file_: Path) -> None:
        pass


class TorchCachingTool(CachingTool):
    """Still generic."""

    @staticmethod
    def load(file_: Path) -> T.Any:
        with file_.open("rb") as fb:
            obj = torch.load(fb)  # type: ignore
        return obj

    @staticmethod
    def save(obj: T.Any, file_: Path) -> None:
        with file_.open("wb") as fb:
            torch.save(obj, fb)  # type: ignore


@dataclasses.dataclass
class CacheableSettings:
    """subclasses will define what makes a cache unique."""

    def __repr__(self) -> str:
        raise NotImplementedError()


Thing = T.Any
# Thing = T.TypeVar("Thing", bound=T.Any) # can't do that because _Things won't work


@dataclasses.dataclass
class ThingCacher:
    """A name, and a caching tool."""

    name: str
    caching_tool: CachingTool = TorchCachingTool()

    def from_cache(self, cache_dir: Path) -> Thing:
        thing = self.caching_tool.load(self._cached_file_path(cache_dir))
        return T.cast(Thing, thing)

    def to_cache(self, cache_dir: Path, thing: Thing) -> None:
        self.caching_tool.save(thing, self._cached_file_path(cache_dir))

    def _cached_file_path(self, cache_dir: Path) -> Path:
        return cache_dir / (self.name + ".cached")


_Things = T.TypeVar("_Things", bound=T.Tuple[Thing, ...])
_Settings = T.TypeVar("_Settings", bound=CacheableSettings)


class Cacheable(T.Generic[_Settings, _Things]):
    """Support caching anything.

    Look at the abstract methods defined below to understand how to use this.
    """

    _Cacheable = T.TypeVar("_Cacheable", bound="Cacheable[_Settings, _Things]")

    @staticmethod
    @abc.abstractmethod
    def thing_cachers() -> T.Tuple[ThingCacher, ...]:
        pass

    @abc.abstractmethod
    def __init__(self, settings: CacheableSettings, things: _Things) -> None:
        """Initialize from post processed results."""
        pass

    @classmethod
    @abc.abstractmethod
    def make_cached_things(cls: T.Type[_Cacheable], settings: _Settings) -> _Things:
        pass

    @classmethod
    def try_from_cache(
        cls: T.Type[_Cacheable], settings: _Settings, cache_dir: T.Optional[Path],
    ) -> _Cacheable:
        """Check if a cached version is available, otherwise, do processing and cache it.

        Args:
            cache_dir: If None, no caching will be performed, but
                self.make_cached_things() will be called anyways.
        """
        if cache_dir is None:
            things = cls.make_cached_things(settings)
            logger.info(
                f"{cls.__name__}.try_from_cache: Not checking cache, cache_dir=None."
            )
        else:
            # Use the  repr to create a cache dir
            settings_hash = hashlib.sha1(repr(settings).encode()).hexdigest()
            specific_cache_dir = cache_dir / settings_hash
            specific_cache_dir.mkdir(exist_ok=True)

            if all(  # Check if all cached things exist
                thing_cacher._cached_file_path(specific_cache_dir).exists()
                for thing_cacher in cls.thing_cachers()
            ):
                logger.info(f"{cls.__name__}.try_from_cache: Found cached.")

                things = T.cast(
                    _Things,
                    tuple(
                        thing_cacher.from_cache(specific_cache_dir)
                        for thing_cacher in cls.thing_cachers()
                    ),
                )
            else:  # Else, make the cached things and save them
                logger.info(
                    f"{cls.__name__}.try_from_cache: No cache found. Making things, and caching them."
                )
                things = cls.make_cached_things(settings)
                thing: Thing
                for thing_cacher, thing in zip(cls.thing_cachers(), things):
                    thing_cacher.to_cache(specific_cache_dir, thing)

        return cls(settings, things)
