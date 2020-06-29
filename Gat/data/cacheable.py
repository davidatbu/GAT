import abc
import hashlib
import logging
import typing as T
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class Cacheable(abc.ABC):
    """Support caching anything.

    Look at the abstract methods defined below to understand how to use this.
    """

    def __init__(self, cache_dir: Path, ignore_cache: bool) -> None:
        """Check if a cached version is available."""
        # Use the  repr to create a cache dir
        ignore_cache = True
        obj_repr_hash = hashlib.sha1(repr(self).encode()).hexdigest()
        self.specific_cache_dir = cache_dir / obj_repr_hash
        self.specific_cache_dir.mkdir(exist_ok=True)

        if self._cached_exists() and not (ignore_cache):
            logger.info(f"{obj_repr_hash} found cached.")
            self._from_cache()
        else:
            logger.info(f"{obj_repr_hash} not found cached. Processing ...")
            self.process()
            self.to_cache()

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Return a unique representation of the settings for the class.

        What is returned must be:
            1. The same across instances that should share the same cached attributes.
            2. Reproducible across multiple Python runs(so `hash()` doesn't work).
        """
        pass

    @abc.abstractproperty
    def _cached_attrs(self) -> T.List[str]:
        """List of attributes that will be cached/restored from cache."""
        pass

    @abc.abstractmethod
    def process(self) -> None:
        """Do the processing that will set the _cached_attrs.

        This function will not be called if a cached version is found.
        After this is called, every attribute in self._cached_attrs must be set.
        """
        pass

    def _cached_exists(self) -> bool:
        """Check if a cached version of the cached attributes exist."""
        return all(
            [
                self._cache_fp_for_attr(attr_name).exists()
                for attr_name in self._cached_attrs
            ]
        )

    def _from_cache(self) -> None:
        """Restore cached attributes."""
        for attr_name in self._cached_attrs:
            fp = self._cache_fp_for_attr(attr_name)

            with fp.open("rb") as fb:
                obj = torch.load(fb)  # type: ignore
            setattr(self, attr_name, obj)

    def _cache_fp_for_attr(self, attr_name: str) -> Path:
        """Return the cache file name for a specific attribute."""
        return self.specific_cache_dir / f"{attr_name}.torch"

    def to_cache(self) -> None:
        """Save cached attributes to cache."""
        for attr_name in self._cached_attrs:

            fp = self._cache_fp_for_attr(attr_name)
            obj = getattr(self, attr_name)
            with fp.open("wb") as fb:
                torch.save(obj, fb)  # type: ignore
