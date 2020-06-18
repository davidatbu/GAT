"""Test pooling over BERT subword."""
import pytest
import torch

from Gat.data.tokenizers.reconciler import TokenizingReconciler
from Gat.neural import layers
from tests.conftest import VocabSetup


@pytest.fixture
def reconciler_fixture(
    vocab_setup: VocabSetup, bert_embedder: layers.BertEmbedder
) -> TokenizingReconciler:
    reconciler = TokenizingReconciler(
        vocab_setup.bert_vocab, vocab_setup.basic_vocab.tokenizer, bert_embedder
    )
    return reconciler


def test_one_subword_per_word(
    reconciler_fixture: TokenizingReconciler,
    vocab_setup: VocabSetup,
    bert_embedder: layers.BertEmbedder,
) -> None:
    lstxt = ["i love you."]

    from_rec = reconciler_fixture.forward(lstxt)
    without_rec = bert_embedder.forward(
        vocab_setup.bert_vocab.batch_tokenize_and_get_tok_ids(lstxt)
    )
    torch.testing.assert_allclose(  # type: ignore
        without_rec.rename(None), from_rec.rename(None)
    )


def test_two_subwords_per_word(
    reconciler_fixture: TokenizingReconciler,
    vocab_setup: VocabSetup,
    bert_embedder: layers.BertEmbedder,
) -> None:
    lstxt = ["pacification"]  # tokenizes into ["pacific", "##ation"]

    from_rec = reconciler_fixture.forward(lstxt)
    without_rec = bert_embedder.forward(
        vocab_setup.bert_vocab.batch_tokenize_and_get_tok_ids(lstxt)
    )
    without_rec_pooled = without_rec.mean(dim="L").align_to(  # type: ignore
        "B", "L", "E"
    )
    torch.testing.assert_allclose(  # type: ignore
        without_rec_pooled.rename(None), from_rec.rename(None)
    )
