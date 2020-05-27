import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..config.base import GATConfig

logger = logging.getLogger("__main__")


class DotProductAttHead(nn.Module):  # type: ignore
    def __init__(
        self,
        config: GATConfig,
        edge_k_embedding: nn.Embedding,
        edge_v_embedding: nn.Embedding,
    ) -> None:
        super().__init__()
        edge_dropout_p = config.edge_dropout_p
        embedding_dim = config.embedding_dim
        nhid = config.nhid

        self.W_q = nn.Linear(embedding_dim, nhid)
        self.W_k = nn.Linear(embedding_dim, nhid)
        self.W_v = nn.Linear(
            embedding_dim, nhid, bias=False
        )  # Why have bias if the layer normalization will add?

        self.edge_k_embedding = edge_k_embedding
        self.edge_v_embedding = edge_v_embedding
        self.nhid = nhid
        self.softmax = nn.Softmax(dim=1)
        self.edge_dropout = nn.Dropout(p=edge_dropout_p)

    def forward(self, input: Tensor, adj: Tensor, edge_type: Tensor) -> Tensor:

        Q = self.W_q(input)
        K = self.W_k(input)
        V = self.W_v(input)

        edge_k = self.edge_k_embedding(edge_type)

        att_scores = Q @ K.t()

        # This is such terrible coding practice, because it's so not obvious what's going on here.
        # adjnonzero(as_tuple=True) should hopefully be of the same length as edge_k
        att_scores[adj.nonzero(as_tuple=True)] += edge_k.view(-1)  # type: ignore

        att_scores /= self.nhid

        zero_vec = -9e15 * torch.ones_like(att_scores)
        att_scores = torch.where(adj > 0, att_scores, zero_vec)
        att_scores /= math.sqrt(self.nhid)
        att_probs = self.softmax(att_scores)

        h_prime = att_probs @ V

        return h_prime


class GATLayer(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig, concat: bool = True):
        nhid = config.nhid
        nheads = config.nheads
        embedding_dim = config.embedding_dim
        nedge_type = config.nedge_type

        if not nhid * nheads == embedding_dim:
            raise Exception("nhid * nheads != out_features")
        super(GATLayer, self).__init__()

        out_dim = embedding_dim if concat else nhid

        self.W_o = nn.Linear(out_dim, out_dim)

        # THe +1 is because we might add an additional edge type
        edge_k_embedding = nn.Embedding(nedge_type + 1, 1)
        edge_v_embedding = nn.Embedding(nedge_type + 1, nhid)
        self.attentions = nn.ModuleList(
            [
                DotProductAttHead(
                    config,
                    edge_k_embedding=edge_k_embedding,
                    edge_v_embedding=edge_v_embedding,
                )
                for _ in range(nheads)
            ]
        )
        self.concat = concat

    def forward(self, h: Tensor, adj: Tensor, edge_type: Tensor) -> Tensor:
        lsatt_res = [att(h, adj, edge_type) for att in self.attentions]
        if self.concat:
            h = torch.cat(lsatt_res, dim=1)
        else:
            h = torch.stack(lsatt_res, dim=0).mean(dim=0)
        h = self.W_o(h)
        return h


class GATLayerWrapper(nn.Module):  # type: ignore
    def __init__(
        self, config: GATConfig, do_residual: bool = True, concat: bool = True
    ):
        if do_residual and not concat:
            raise Exception("Can't do residual connection when not concatting")
        super().__init__()
        do_layer_norm = config.do_layer_norm
        feat_dropout_p = config.feat_dropout_p

        self.layer = GATLayer(config, concat=concat)

        if concat:
            out_features = config.embedding_dim
        else:
            out_features = config.nhid
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(feat_dropout_p)

        self.do_residual = do_residual
        self.do_layer_norm = do_layer_norm

    def forward(self, h: Tensor, adj: Tensor, edge_type: Tensor) -> Tensor:
        h = self.dropout(h)
        h_new = self.layer(h, adj, edge_type)
        if self.do_residual:
            h_new = h_new + h  # Learnt not to do += for autograd
        if self.do_layer_norm:
            h_new = self.layer_norm(h_new)
        return h_new  # type: ignore


class FeedForwardWrapper(nn.Module):  # type: ignore
    def __init__(
        self, config: GATConfig, do_residual: bool = True, concat: bool = True
    ):
        if do_residual and not concat:
            raise Exception("Can't do residual connection when not concatting")
        super().__init__()
        do_layer_norm = config.do_layer_norm
        embedding_dim = config.embedding_dim
        nhid = config.nhid
        feat_dropout_p = config.feat_dropout_p

        w2_bias = True
        if do_layer_norm:
            w2_bias = False

        self.W1 = nn.Linear(embedding_dim, embedding_dim // 4, bias=True)
        if concat:
            out_features = embedding_dim
        else:
            out_features = nhid
        self.relu = nn.ReLU()
        self.W2 = nn.Linear(embedding_dim // 4, out_features, bias=w2_bias)
        self.layer_norm = nn.LayerNorm(out_features)

        self.do_residual = do_residual
        self.do_layer_norm = do_layer_norm
        self.dropout = nn.Dropout(feat_dropout_p)

    def forward(self, h: Tensor) -> Tensor:
        h = self.dropout(h)
        h_new = self.W2(self.relu(self.W1(h)))
        if self.do_residual:
            h_new = h_new + h  # Learnt not to do += for autograd
        if self.do_layer_norm:
            h_new = self.layer_norm(h_new)
        return h_new


# TODO: Add positional embeddings here
class EmbeddingWrapper(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig, emb_init: Optional[Tensor]):
        super().__init__()
        do_layer_norm = config.do_layer_norm
        vocab_size = config.vocab_size
        embedding_dim = config.embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )

        self.position_embedding = PositionEmbedding(config)

        if emb_init is not None:
            logger.info("Initializing embeddings with pretrained embeddings ...")
            self.embedding.from_pretrained(emb_init)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.do_layer_norm = do_layer_norm

    def forward(self, tcword_id: Tensor, position_ids: Tensor) -> Tensor:
        assert len(tcword_id) == len(position_ids)
        h = self.embedding(tcword_id)
        h = h + self.position_embedding(position_ids)

        if self.do_layer_norm:
            h = self.layer_norm(h)
        return h


class PositionEmbedding(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig) -> None:
        super().__init__()
        initial_max_length = 100
        self.embedding_dim = config.embedding_dim

        self.embs: Tensor
        self.register_buffer(
            "embs", self.create_embs(initial_max_length, self.embedding_dim)
        )

    @staticmethod
    def create_embs(max_length: int, embedding_dim: int) -> Tensor:
        embs = torch.zeros(max_length, embedding_dim)
        position_enc = np.array(
            [
                [
                    pos / np.power(10000, 2 * (j // 2) / embedding_dim)
                    for j in range(embedding_dim)
                ]
                for pos in range(max_length)
            ]
        )
        embs[:, 0::2] = torch.from_numpy(np.sin(position_enc[:, 0::2]))
        embs[:, 1::2] = torch.from_numpy(np.cos(position_enc[:, 1::2]))
        embs.detach_()
        embs.requires_grad = False
        return embs

    def forward(self, position_ids: Tensor) -> Tensor:
        cur_max = int(position_ids.max().item())
        if cur_max > self.embs.size(0):
            logger.info(f"Increasing max position embedding to {cur_max}")
            self.register_buffer("embs", self.create_embs(cur_max, self.embedding_dim))
        return self.embs[position_ids]


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
