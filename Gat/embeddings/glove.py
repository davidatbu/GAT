import logging
from pathlib import Path
from typing import List

import torch

from .base import TextBasedWordToVec

GLOVE_F = Path(
    "/projectnb/llamagrp/davidat/pretrained_models/glove_6b/glove.6B.300d.txt"
)
logging.basicConfig()
logger = logging.getLogger("embeddings")
logger.setLevel(logging.INFO)


class GloveWordToVec(TextBasedWordToVec):
    def __init__(
        self, file_p: Path = GLOVE_F, initial_sentences: List[List[str]] = [],
    ):
        super().__init__(
            name="glove", file_p=file_p, dim=300, initial_sentences=initial_sentences
        )


def _test() -> None:
    ex1 = ["i", "cry", "you"]
    ex2 = ["gentrification", "interestingly", "enough"]
    w2v = GloveWordToVec(initial_sentences=[ex1, ex2])
    w2v.set_unk_as_avg()
    assert w2v.for_word(ex1[0]).shape == (1, w2v.dim)
    a = w2v.for_lsword(ex1)
    b = w2v.for_lsword(ex2)
    assert 3 == len(a) == len(b)
    all_dims = set(map(lambda x: len(x), a + b))
    assert (len(all_dims)) == 1
    assert all_dims.pop() == w2v.dim

    # Make sure the UNK got set properly
    all_words_embs = w2v.for_lsword(ex1 + ex2).mean(dim=0)
    unk_emb = w2v.for_word("ADFDFGKLJE")
    assert torch.allclose(all_words_embs, unk_emb)


if __name__ == "__main__":
    _test()
