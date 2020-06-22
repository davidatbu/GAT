"""THings that produce losses."""
import typing as T
from functools import lru_cache

import torch
import torch.nn as nn
from torch import Tensor

from ..config.base import GATConfig
from ..config.base import GATForSeqClsfConfig
from Gat import data
from Gat.neural import layers


class GATForSequenceClassification(nn.Module):  # type: ignore
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        edge_dropout_p: float,
        rezero_or_residual: T.Literal["rezero", "residual"],
        intermediate_dim: int,
        num_layers: int,
        num_classes: int,
        padding_tok_id: int,
        cls_tok_id: int,
        feat_dropout_p: float,
        node_embedding_type: T.Literal["pooled_bert", "basic"],
        use_edge_features: bool,
        num_edge_types: int,
        word_vocab: data.BasicVocab,
        sub_word_vocab: T.Optional[data.Vocab] = None,
    ):
        """

        Args:
            cls_tok_id: We need to assert that word_ids[:, 0] in `self.forward()` is
                indeed equal to it.
        """

        self._cls_tok_id = cls_tok_id

        positional_embedder = layers.PositionalEmbedder(embedding_dim)
        if node_embedding_type == "basic":
            word_embedder: layers.Embedder = layers.BasicEmbedder(
                num_embeddings=word_vocab.vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_tok_id,
            )
        else:
            assert sub_word_vocab is not None
            sub_word_embedder = layers.BertEmbedder()
            word_embedder = layers.ReconcilingEmbedder(
                sub_word_vocab, word_vocab, sub_word_embedder
            )

        if use_edge_features:
            key_edge_feature_embedder: T.Optional[
                layers.BasicEmbedder
            ] = layers.BasicEmbedder(
                num_embeddings=num_edge_types,
                embedding_dim=embedding_dim,
                padding_idx=word_vocab.padding_tok_id,
            )
        else:
            key_edge_feature_embedder = None

        lsnode_feature_embedder = [word_embedder, positional_embedder]

        self._gat_layered = layers.GATLayered(
            num_heads,
            edge_dropout_p,
            rezero_or_residual,
            intermediate_dim,
            num_layers,
            feat_dropout_p,
            lsnode_feature_embedder,
            key_edge_feature_embedder,
            None,
        )

        self.dropout = nn.Dropout(feat_dropout_p)
        self.linear = nn.Linear(embedding_dim, num_classes)
        self._crs_entrpy = nn.CrossEntropyLoss()

    @T.overload
    def forward(
        self,
        word_ids: torch.LongTensor,
        adj: torch.LongTensor,
        edge_types: torch.LongTensor,
    ) -> T.Tuple[Tensor]:
        ...

    @T.overload
    def forward(
        self,
        word_ids: torch.LongTensor,
        adj: torch.LongTensor,
        edge_types: torch.LongTensor,
        labels: torch.LongTensor,
    ) -> T.Tuple[Tensor, Tensor]:
        ...

    def forward(
        self,
        word_ids: torch.LongTensor,
        adj: torch.LongTensor,
        edge_types: torch.LongTensor,
        labels: T.Optional[torch.LongTensor] = None,
    ) -> T.Tuple[Tensor, ...]:
        h = self._gat_layered(word_ids=word_ids, adj=adj, edge_types=edge_types)

        assert word_ids[:, 0].equal(
            torch.tensor(self._cls_tok_id, dtype=torch.long)
            .align_as(word_ids)
            .expand_as(word_ids)
        )
        cls_id_h = h[:, 0]

        cls_id_h = self.dropout(cls_id_h)
        logits = self.linear(cls_id_h)

        if labels is not None:
            loss = self._crs_entrpy(logits, labels)

            return (logits, loss)
        return (logits,)
