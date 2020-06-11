import bisect
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import torch
from tqdm import tqdm  # type: ignore


FASTTEXT_F = Path(
    "/projectnb/llamagrp/davidat/projects/graphs/fasttext/wiki-news-300d-1M-subword.vec"
)

logging.basicConfig()
logger = logging.getLogger("embeddings")
logger.setLevel(logging.INFO)


class Phrase(Tuple[str]):
    def __repr__(self) -> str:
        return "Phrase[" + " ".join(self) + "]"


class WordToVec:
    name: Optional[str] = None
    dim: int

    def __init__(self, do_sif: bool):
        pass

    def for_lsword(self, sent: Iterable[str]) -> torch.Tensor:
        raise NotImplementedError()

    def for_lslsword(self, sents: List[List[str]]) -> torch.Tensor:
        raise NotImplementedError()

    def for_word(self, word: str) -> torch.Tensor:
        raise NotImplementedError()

    def for_lsword_averaged(self, sent: Iterable[str]) -> torch.Tensor:
        """
        Returns
        -------
            (D,)
        """
        vecs: torch.Tensor = self.for_lsword(sent)  # (W, D)
        return torch.mean(vecs, dim=0)

    def prefetch_lsword(self, lsword: List[str]) -> None:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"WordToVec_{self.name}_{self.dim}"

    def set_unk_as_avg(self) -> None:
        raise NotImplementedError()

    def for_unk(self) -> torch.Tensor:
        raise NotImplementedError()


class LargeFileCacher(Dict[str, Any]):
    def __init__(self, file_p: Path, initial_keys: List[str] = []):
        self.file_p = file_p
        self.cache_keys(initial_keys)
        if not self.file_p.exists():
            raise IOError(f"{self.file_p} must exist")

    def cache_keys(self, keys: List[str]) -> None:
        if not len(keys):
            return
        logger.info(
            f"Caching lines for {len(keys)} keys. Be warned that not all may be found."
        )
        keys = sorted(set(keys))
        with open(self.file_p) as f:
            f = tqdm(f, total=1e6)
            for line in f:
                key_value_sep: int = line.find(" ")
                key = line[:key_value_sep]
                possible_pos = bisect.bisect_left(keys, key)
                if possible_pos < len(keys) and keys[possible_pos] == key:
                    value = self.get_value_from_line(line[key_value_sep:])
                    self[key] = value
                    keys.pop(possible_pos)
                if not keys:  # Did all the keys
                    break
        if keys:
            logger.warning(f"Could not find embeddings for {len(keys)} lsword: {keys}")

    def get_value_from_line(self, value: str) -> Any:
        return value


class TextVectorsFileCacher(LargeFileCacher):
    def __init__(
        self, dim: int, file_p: Path, *args: List[Any], **kwargs: Dict[str, Any]
    ):
        super().__init__(file_p=file_p, *args, **kwargs)  # type: ignore
        self.dim = dim

    def get_value_from_line(self, value: str) -> torch.Tensor:
        vals = list(map(float, value.split()))
        return torch.tensor(vals, dtype=torch.float).view(1, -1)


class TextBasedWordToVec(WordToVec):
    def __init__(
        self,
        name: str,
        dim: int,
        file_p: Path,
        initial_sentences: List[List[str]] = [],
    ):
        self.dim = dim
        self.name = name
        prefetch_lsword = list(
            set([word for sent in initial_sentences for word in sent])
        )
        self.cacher = TextVectorsFileCacher(file_p=file_p, dim=dim)
        self.cacher.cache_keys(prefetch_lsword)
        self._unk_emb: Optional[torch.Tensor] = None

    def for_word(self, word: str) -> torch.Tensor:
        return self.cacher.get(word, self._unk_emb)

    def for_lsword(self, sent: Iterable[str]) -> torch.Tensor:
        return torch.cat([self.for_word(word) for word in sent], dim=0)

    def prefetch_lsword(self, lsword: List[str]) -> None:
        self.cacher.cache_keys(lsword)

    def set_unk_as_avg(self) -> None:
        self._unk_emb = (
            torch.cat(list(self.cacher.values()), dim=0).mean(dim=0).view(1, self.dim)
        )

    def for_unk(self) -> torch.Tensor:
        assert self._unk_emb is not None
        return self._unk_emb


class FastTextWordToVec(TextBasedWordToVec):
    def __init__(
        self, file_p: Path = FASTTEXT_F, initial_sentences: List[List[str]] = [],
    ):
        super().__init__(
            name="fasttext", file_p=file_p, dim=300, initial_sentences=initial_sentences
        )
