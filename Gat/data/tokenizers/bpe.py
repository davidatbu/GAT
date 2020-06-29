from __future__ import annotations

import typing as T

from bpemb import BPEmb  # type: ignore

from .base import Tokenizer


class BPETokenizer(Tokenizer):
    def __init__(self, bpemb_en: BPEmb) -> None:
        """. """
        assert bpemb_en.lang == "en"
        self._pbemb_en = bpemb_en
        self._vocab_size = bpemb_en.vocab_size

    def _tokenize(self, txt: str) -> T.List[str]:
        return self._pbemb_en.encode(txt)  # type: ignore

    def __repr__(self) -> str:
        return f"{self.__class__}-vocab_size_{str(self._vocab_size)}"
