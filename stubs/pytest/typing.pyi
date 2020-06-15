import typing as T

# From Mypy doc on how to annotate function decorators that don't change the signature
F = T.TypeVar("F", bound=T.Callable[..., T.Any])
