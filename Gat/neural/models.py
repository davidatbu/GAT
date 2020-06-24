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
        assert config.dataset_dep is not None
        super().__init__()

        self._cls_tok_id = word_vocab.cls_tok_id

        positional_embedder = layers.PositionalEmbedder(config.embedding_dim)

        setattr(positional_embedder, "debug_name", "positional_embedder")
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

        setattr(word_embedder, "debug_name", "word_embedder")
        self._word_embedder = word_embedder

        if config.use_edge_features:
            head_size = config.embedding_dim // config.gat_layered.num_heads
            key_edge_feature_embedder: T.Optional[
                layers.BasicEmbedder
            ] = layers.BasicEmbedder(
                num_embeddings=config.dataset_dep.num_edge_types,
                embedding_dim=head_size,
                padding_idx=word_vocab.padding_tok_id,
            )
            setattr(
                key_edge_feature_embedder, "debug_name", "key_edge_feature_embedder"
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

        self._dropout = nn.Dropout(config.gat_layered.feat_dropout_p)
        self._linear = nn.Linear(config.embedding_dim, config.dataset_dep.num_classes)

        setattr(self._linear, "debug_name", "linear")

    @property
    def word_embedder(self) -> layers.Embedder:
        """Used in train.py's LitGatForSequenceClassification._collate_fn"""
        return self._word_embedder

    def forward(
        self,
        word_ids: torch.LongTensor,
        batched_adj: torch.BoolTensor,
        edge_types: T.Optional[torch.LongTensor],
    ) -> Tensor:

        word_ids.rename_("B", "L")
        batched_adj.rename_("B", "L_left", "L_right")
        edge_types.rename_("B", "L_left", "L_right")

        h = self._gat_layered(
            node_ids=word_ids, batched_adj=batched_adj, edge_types=edge_types
        )

        assert torch.all(
            word_ids[:, 0] == torch.tensor(self._cls_tok_id, dtype=torch.long)
        )
        cls_id_h = h[:, 0]

        cls_id_h = self._dropout(cls_id_h)
        logits = self._linear(cls_id_h)

        return logits

    def __call__(
        self,
        word_ids: torch.LongTensor,
        batched_adj: torch.BoolTensor,
        edge_types: T.Optional[torch.LongTensor],
    ) -> Tensor:

        return super().__call__(word_ids, batched_adj, edge_types)  # type: ignore
