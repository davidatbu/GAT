"""Tests for BertEmbedder, BasicEmbedder, ReconcilingEmbedder."""
from __future__ import annotations

import torch

from Gat.neural import layers
from Gat.utils import Device
from tests.conftest import VocabSetup


def test_basic(vocab_setup: VocabSetup, device: torch.device) -> None:
    basic_embedder = layers.BasicEmbedder(
        vocab=vocab_setup.basic_vocab,
        num_embeddings=3,
        embedding_dim=768,
        padding_idx=vocab_setup.basic_vocab.padding_tok_id,
    )

    lssent = ["huh hoh", "something"]
    lslstok_id = vocab_setup.basic_vocab.batch_tokenize_and_get_tok_ids(lssent)
    tok_ids = vocab_setup.basic_vocab.prepare_for_embedder(
        lslstok_id, basic_embedder, device=device
    )
    embs = basic_embedder(tok_ids)
    embs = vocab_setup.basic_vocab.strip_after_embedder(embs)

    embs_size = tuple(embs.size())
    assert embs_size == (2, 2, basic_embedder.embedding_dim)


def test_pos(vocab_setup: VocabSetup, device: torch.device) -> None:
    lssent = ["huh hoh", "something"]
    lslstok_id = vocab_setup.basic_vocab.batch_tokenize_and_get_tok_ids(lssent)
    pos_embedder = layers.PositionalEmbedder(embedding_dim=999)
    tok_ids = vocab_setup.basic_vocab.prepare_for_embedder(
        lslstok_id, pos_embedder, device=device
    )

    embs = pos_embedder(tok_ids)
    embs = vocab_setup.basic_vocab.strip_after_embedder(embs)
    embs_size = tuple(embs.size())
    assert embs_size == (2, 2, pos_embedder.embedding_dim)


def test_bert(
    vocab_setup: VocabSetup, bert_embedder: layers.BertEmbedder, device: torch.device
) -> None:

    lssent = ["pacification", "something"]
    lslstok_id = vocab_setup.bert_vocab.batch_tokenize_and_get_tok_ids(lssent)
    breakpoint()
    tok_ids = vocab_setup.bert_vocab.prepare_for_embedder(
        lslstok_id, bert_embedder, device=device
    )
    embs = bert_embedder(tok_ids)
    embs = vocab_setup.bert_vocab.strip_after_embedder(embs)

    max_seq_len = max(map(len, lslstok_id))
    assert embs.size("L") == max_seq_len + 1
    # The +1 for the [SEP] token, not removed
    # by strip_after_embedder


def test_reconciler(
    reconciler_embedder: layers.ReconcilingEmbedder,
    vocab_setup: VocabSetup,
    bert_embedder: layers.BertEmbedder,
    device: torch.device,
) -> None:
    lstxt = ["pacification"]  # Bert tokenization and basic tokeinzation different
    bert_lslstok_id = vocab_setup.bert_vocab.batch_tokenize_and_get_tok_ids(lstxt)
    bert_tok_ids = vocab_setup.bert_vocab.prepare_for_embedder(
        bert_lslstok_id, bert_embedder, device=device
    )
    without_rec = bert_embedder(bert_tok_ids)
    without_rec = vocab_setup.bert_vocab.strip_after_embedder(without_rec)

    basic_lslstok_id = vocab_setup.basic_vocab.batch_tokenize_and_get_tok_ids(lstxt)
    basic_tok_ids = vocab_setup.basic_vocab.prepare_for_embedder(
        basic_lslstok_id, reconciler_embedder, device=device
    )
    with_rec = reconciler_embedder(basic_tok_ids)
    with_rec = vocab_setup.basic_vocab.strip_after_embedder(with_rec)

    without_rec_pooled = (
        without_rec[:, :-1]  # The [SEP] token is not yet removed
        .mean(dim="L")  # type: ignore
        .align_to("B", "L", "E")
    )
    torch.testing.assert_allclose(  # type: ignore
        without_rec_pooled.rename(None), with_rec.rename(None)
    )
