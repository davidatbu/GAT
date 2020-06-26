from __future__ import annotations

import typing as T
from pathlib import Path

from Gat.data.tokenizers import Tokenizer

from bpemb import BPEmb


class BPETokenizer(Tokenizer):
    def __init__(self, BPEemb) -> None:
        """.

        Args:
            vocab_size: For now, we are doing only 25000, no reason we can't change
            that.
        """
        self._vocab_size = vocab_size


    def tokenize(self, txt: str) -> T.List[str]:
        pass

    def __repr__(self) -> str:
        return f"{self._class__}-model_file_{str(self._model_file)}"
