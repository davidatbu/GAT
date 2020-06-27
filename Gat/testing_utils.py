import functools
import pdb
import shutil
import sys
import tempfile
import traceback
import typing as T
from pathlib import Path


F = T.TypeVar("F", bound=T.Callable[[T.Any], T.Any])


def debug_on(*exceptions: T.Type[Exception]) -> T.Callable[[F], F]:
    if not exceptions:
        exceptions = (Exception,)

    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: T.Any, **kwargs: T.Any) -> T.Any:
            try:
                return f(*args, **kwargs)  # type: ignore
            except exceptions:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])

        return wrapper  # type: ignore

    return decorator


class TempDirMixin:
    """
    Sets:
        self._temp_dir
    """

    def setUp(self) -> None:
        self._temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self._temp_dir)
