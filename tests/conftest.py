"""PyTest fixtures that are shared across tests files.

https://docs.pytest.org/en/stable/fixture.html#conftest-py-sharing-fixture-functions

    "If during implementing your tests you realize that you want to use a fixture
     function from multiple test files you can move it to a conftest.py file. You don’t
     need to import the fixture you want to use in a test, it automatically gets
     discovered by pytest. The discovery of fixture functions starts at test classes,
     then test modules, then conftest.py files and finally builtin and third party
     plugins."

"""
from __future__ import annotations

import shutil
import tempfile
import typing as T
from pathlib import Path

import pytest

from Gat import data
from Gat.neural import layers


@pytest.fixture(scope="session")
def temp_dir() -> T.Iterator[Path]:
    """A temporary directory."""
    dir_ = Path(tempfile.mkdtemp("gat_testing"))
    yield dir_
    shutil.rmtree(dir_)


class VocabSetup(T.NamedTuple):
    basic_vocab: data.BasicVocab
    bert_vocab: data.BertVocab
    lslstok_id: T.List[T.List[int]]


@pytest.fixture(scope="session")
def vocab_setup(temp_dir: Path) -> VocabSetup:
    txt_src = data.FromIterableTextSource(
        [
            (["guard your heart"], "yes"),
            (["eat from the tree of the knowledge of good and evil"], "no"),
        ]
    )

    basic_vocab = data.BasicVocab(
        txt_src=txt_src,
        tokenizer=data.tokenizers.spacy.WrappedSpacyTokenizer(),
        cache_dir=temp_dir,
        unk_thres=2,
    )

    bert_vocab = data.BertVocab(
        txt_src=txt_src,
        tokenizer=data.tokenizers.bert.WrappedBertTokenizer(),
        cache_dir=temp_dir,
    )

    lslstok_id = [
        [1, 0],
        [2],
    ]
    return VocabSetup(
        basic_vocab=basic_vocab, bert_vocab=bert_vocab, lslstok_id=lslstok_id
    )


@pytest.fixture(scope="session")
def bert_embedder(vocab_setup: VocabSetup) -> layers.BertEmbedder:
    embedder = layers.BertEmbedder(vocab=vocab_setup.bert_vocab, last_how_many_layers=4)
    return embedder
