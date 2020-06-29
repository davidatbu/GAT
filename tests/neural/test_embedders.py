"""Tests for BertEmbedder, BasicEmbedder, ReconcilingEmbedder."""
from __future__ import annotations

import unittest

import torch

from Gat import data
from Gat import testing_utils
from Gat.neural import layers
from Gat.testing_utils import debug_on


class TestEmbs(testing_utils.TempDirMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        txt_src = data.FromIterableTextSource(
            [(["guard your heart"], "yes"), (["pacification"], "no")]
        )
        self._basic_vocab = data.BasicVocab(
            txt_src=txt_src,
            tokenizer=data.tokenizers.spacy.WrappedSpacyTokenizer(),
            cache_dir=self._temp_dir,
            unk_thres=1,
        )

        self._bert_vocab = data.BertVocab()

        self._bert_embedder = layers.BertEmbedder()

        self._reconciler_embedder = layers.ReconcilingEmbedder(
            self._bert_vocab, self._basic_vocab, self._bert_embedder
        )

    def test_basic(self) -> None:
        basic_embedder = layers.BasicEmbedder(
            num_embeddings=3,
            embedding_dim=768,
            padding_idx=self._basic_vocab.get_tok_id(self._basic_vocab.padding_tok),
        )

        lssent = ["huh hoh", "something"]
        lslstok_id = self._basic_vocab.batch_tokenize_and_get_lstok_id(lssent)
        tok_ids = self._basic_vocab.prepare_for_embedder(lslstok_id, basic_embedder)
        embs = basic_embedder(tok_ids)
        embs = self._basic_vocab.strip_after_embedder(embs)

        embs_size = tuple(embs.size())
        assert embs_size == (2, 2, basic_embedder.embedding_dim)

    def test_pos(self) -> None:
        lssent = ["huh hoh", "something"]
        lslstok_id = self._basic_vocab.batch_tokenize_and_get_lstok_id(lssent)
        pos_embedder = layers.PositionalEmbedder(embedding_dim=999)
        tok_ids = self._basic_vocab.prepare_for_embedder(lslstok_id, pos_embedder)

        embs = pos_embedder(tok_ids)
        embs = self._basic_vocab.strip_after_embedder(embs)
        embs_size = tuple(embs.size())
        assert embs_size == (2, 2, pos_embedder.embedding_dim)

    def test_bert(self,) -> None:

        lssent = ["pacification", "something"]
        lslstok_id = self._bert_vocab.batch_tokenize_and_get_lstok_id(lssent)
        tok_ids = self._bert_vocab.prepare_for_embedder(lslstok_id, self._bert_embedder)
        # (B, L)
        embs = self._bert_embedder(tok_ids)
        # (B, L, E)
        embs = self._bert_vocab.strip_after_embedder(embs)
        # (B, L, E)

        max_seq_len = max(map(len, lslstok_id))
        assert embs.size(1) == max_seq_len + 1
        # The +1 for the [sep] token, not removed by strip_after_embedder.

    @debug_on()
    def test_reconciler(self,) -> None:
        lstxt = ["pacification"]  # Bert tokenization and basic tokeinzation different
        bert_lslstok_id = self._bert_vocab.batch_tokenize_and_get_lstok_id(lstxt)
        bert_tok_ids = self._bert_vocab.prepare_for_embedder(
            bert_lslstok_id, self._bert_embedder
        )
        without_rec = self._bert_embedder(bert_tok_ids)
        without_rec = self._bert_vocab.strip_after_embedder(without_rec)

        assert list(without_rec.size()) == [1, 3, 768]  # There's a SEP token

        basic_lslstok_id = self._basic_vocab.batch_tokenize_and_get_lstok_id(lstxt)
        basic_tok_ids = self._basic_vocab.prepare_for_embedder(
            basic_lslstok_id, self._reconciler_embedder
        )
        with_rec = self._reconciler_embedder(basic_tok_ids)
        with_rec = self._basic_vocab.strip_after_embedder(with_rec)

        assert list(with_rec.size()) == [1, 1, 768]

        without_rec_pooled = without_rec[:, :-1].mean(dim=1, keepdim=True)
        # (B, 1, E)
        torch.testing.assert_allclose(without_rec_pooled, with_rec)  # type: ignore


if __name__ == "__main__":
    unittest.main()
