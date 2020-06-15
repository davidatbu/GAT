import typing as T

from ..typing import F

def skip(reason: T.Optional[str]) -> T.Callable[[F], F]: ...
