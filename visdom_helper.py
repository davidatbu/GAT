from pathlib import Path
from time import time
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set

import torch
from torch import Tensor
from visdom import Visdom  # type: ignore


class CachedVisdom:
    def __init__(
        self,
        save_dir: Path,
        save_secs: float = 60,
        server: str = "http://0.0.0.0",
        port: int = 5432,
        base_url: str = "/",
        offline: bool = False,
    ) -> None:

        self._save_secs = save_secs
        self._save_dir = save_dir

        self._viz = Visdom(
            server=server,
            port=port,
            base_url=base_url,
            offline=offline,
            log_to_filename=save_dir / "visdom.log",
        )
        self._last_saved_time: float = 0
        self._envs: Set[str] = set()

    def line(
        self,
        Y: Tensor,
        win: str,
        env: str,
        X: Optional[Tensor] = None,
        opts: Optional[Dict[str, Any]] = None,
        update: Optional[str] = "update",
        name: Optional[str] = None,
    ) -> None:
        self._viz.line(Y, X, win=win, env=env, opts=opts, update=update, name=name)

        self._save_if_needed(last_modified_env=env)

    def _save_if_needed(self, last_modified_env: str) -> None:
        self._envs.add(last_modified_env)
        cur_time = time()

        if cur_time - self._last_saved_time > self._save_secs:
            self._viz.save(list(sorted(self._envs)))
            self._last_saved_time = cur_time


def _test() -> None:
    viz = CachedVisdom(save_dir=Path("data/"))
    Y = torch.tensor([12, 10, 9, 6, 4, 3, 2, 2, 2, 1, 1, 1])
    X = torch.arange(len(Y))
    viz.line(
        Y=Y, X=X, env="initial", win="loss",
    )


if __name__ == "__main__":
    _test()
