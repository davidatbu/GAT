import typing as T

from . import mark
from .typing import F
@T.overload
def fixture(func: F) -> F: ...
@T.overload
def fixture(scope: T.Literal["module", "session"]) -> T.Callable[[F], F]: ...

__all__ = ["mark", "fixture"]
