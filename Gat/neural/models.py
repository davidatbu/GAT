"""THings that produce losses."""
import typing as T
from functools import lru_cache

import torch
import torch.nn as nn
from torch import Tensor

from ..config.base import GATForSequenceClassificationConfig
from Gat import data
from Gat.neural import layers


class GATForSequenceClassification(nn.Module):  # type: ignore
    def __init__(
        self,
        config: GATForSequenceClassificationConfig,
        word_vocab: data.BasicVocab,
        sub_word_vocab: T.Optional[data.Vocab] = None,
    ):
        """

        Args:
            cls_tok_id: We need to assert that word_ids[:, 0] in `self.forward()` is
                indeed equal to it.
        """

        self._cls_tok_id = word_vocab.cls_tok_id

        positional_embedder = layers.PositionalEmbedder(config.embedding_dim)
        if config.node_embedding_type == "basic":
            word_embedder: layers.Embedder = layers.BasicEmbedder(
                num_embeddings=word_vocab.vocab_size,
                embedding_dim=config.embedding_dim,
                padding_idx=word_vocab.padding_tok_id,
            )
        else:
            assert sub_word_vocab is not None
            sub_word_embedder = layers.BertEmbedder()
            word_embedder = layers.ReconcilingEmbedder(
                sub_word_vocab, word_vocab, sub_word_embedder
            )

        if config.use_edge_features:
            key_edge_feature_embedder: T.Optional[
                layers.BasicEmbedder
            ] = layers.BasicEmbedder(
                num_embeddings=config.num_edge_types,
                embedding_dim=config.embedding_dim,
                padding_idx=word_vocab.padding_tok_id,
            )
        else:
            key_edge_feature_embedder = None

        lsnode_feature_embedder = [word_embedder, positional_embedder]

        self._gat_layered = layers.GATLayered(
            config.gat_layered,
            lsnode_feature_embedder,
            key_edge_feature_embedder,
            value_edge_feature_embedder=None,
        )

        self.dropout = nn.Dropout(config.gat_layered.feat_dropout_p)
        self.linear = nn.Linear(config.embedding_dim, config.num_classes)

    def forward(
        self,
        word_ids: torch.LongTensor,
        adj: torch.LongTensor,
        edge_types: T.Optional[torch.LongTensor],
    ) -> Tensor:

        h = self._gat_layered(word_ids=word_ids, adj=adj, edge_types=edge_types)

        assert word_ids[:, 0].equal(
            torch.tensor(self._cls_tok_id, dtype=torch.long)
            .align_as(word_ids)
            .expand_as(word_ids)
        )
        cls_id_h = h[:, 0]

        cls_id_h = self.dropout(cls_id_h)
        logits = self.linear(cls_id_h)

        return logits

    def __call__(
        self,
        word_ids: torch.LongTensor,
        adj: torch.LongTensor,
        edge_types: T.Optional[torch.LongTensor],
    ) -> Tensor:

        return super().__call__(word_ids, adj, edge_types)  # type: ignore
