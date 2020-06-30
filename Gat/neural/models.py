"""THings that produce losses."""
import typing as T

import torch
import torch.nn as nn
from torch import Tensor

from Gat import configs
from Gat import utils
from Gat.data import vocabs
from Gat.neural import layers


class GATForSequenceClassification(nn.Module):  # type: ignore
    def __init__(
        self,
        config: configs.GATForSequenceClassificationConfig,
        word_vocab: vocabs.BasicVocab,
        sub_word_vocab: T.Optional[vocabs.Vocab] = None,
    ):
        """

        Args:
            cls_tok_id: We need to assert that word_ids[:, 0] in `self.forward()` is
                indeed equal to it.
        """
        assert config.dataset_dep is not None
        super().__init__()

        self._cls_tok_id = word_vocab.get_tok_id(word_vocab.cls_tok)

        positional_embedder = layers.PositionalEmbedder(config.embedding_dim)

        setattr(positional_embedder, "debug_name", "positional_embedder")
        if config.node_embedding_type == "basic":
            word_embedder: layers.Embedder = layers.BasicEmbedder(
                num_embeddings=word_vocab.vocab_size,
                embedding_dim=config.embedding_dim,
                padding_idx=word_vocab.get_tok_id(word_vocab.padding_tok),
            )
        else:
            sub_word_embedder: layers.Embedder
            assert sub_word_vocab is not None
            if config.node_embedding_type == "pooled_bert":
                sub_word_embedder = layers.BertEmbedder()
                word_embedder = layers.ReconcilingEmbedder(
                    sub_word_vocab, word_vocab, sub_word_embedder
                )
            elif config.node_embedding_type == "bpe":
                if config.use_pretrained_embs:
                    assert (
                        sub_word_vocab.has_pretrained_embs
                    ), "use_pretrained_embs=True, but sub_word_vocab.has_pretrained_embs=False"
                    sub_word_embedder = layers.BasicEmbedder(
                        padding_idx=sub_word_vocab.get_tok_id(
                            sub_word_vocab.padding_tok
                        ),
                        pretrained_embs=sub_word_vocab.pretrained_embs,
                    )
                else:
                    sub_word_embedder = layers.BasicEmbedder(
                        padding_idx=sub_word_vocab.get_tok_id(
                            sub_word_vocab.padding_tok
                        ),
                        num_embeddings=sub_word_vocab.vocab_size,
                        embedding_dim=config.embedding_dim,
                    )
            word_embedder = layers.ReconcilingEmbedder(
                sub_word_vocab, word_vocab, sub_word_embedder
            )

        setattr(word_embedder, "debug_name", "word_embedder")
        # If we don't do this, our TensorBoard graph won't look nice
        # because we word embedder to GatLayered as well
        self._word_embedder_container = [word_embedder]

        if config.use_edge_features:
            head_size = config.embedding_dim // config.gat_layered.num_heads
            key_edge_feature_embedder: T.Optional[
                layers.BasicEmbedder
            ] = layers.BasicEmbedder(
                num_embeddings=config.dataset_dep.num_edge_types,
                embedding_dim=head_size,
                padding_idx=word_vocab.get_tok_id(word_vocab.padding_tok),
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
        return self._word_embedder_container[0]

    def forward(
        self,
        word_ids: torch.Tensor,
        batched_adj: torch.Tensor,
        edge_types: T.Optional[torch.Tensor],
    ) -> Tensor:
        """
        
        Args:
            word_ids: (B, L)
            batched_adj: (B, L, L)
            edge_types: (B, L, L)
        """

        h = self._gat_layered(
            node_ids=word_ids, batched_adj=batched_adj, edge_types=edge_types
        )
        # (B, L, E)

        assert torch.all(
            word_ids[:, 0] == torch.tensor(self._cls_tok_id, dtype=torch.long)
        )
        cls_id_h = h[:, 0]

        cls_id_h = self._dropout(cls_id_h)
        logits = self._linear(cls_id_h)

        return logits

    def __call__(
        self,
        word_ids: torch.Tensor,
        batched_adj: torch.Tensor,
        edge_types: T.Optional[torch.Tensor],
    ) -> Tensor:

        return super().__call__(word_ids, batched_adj, edge_types)  # type: ignore
