import typing as T

from ..typing import F

def skip(reason: T.Optional[str]) -> T.Callable[[F], F]: ...
def skipif(do_skip: bool, reason: T.Optional[str]) -> T.Callable[[F], F]: ...

__all__ = ["skip", "skipif"]
