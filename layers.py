import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from config import GATConfig

logger = logging.getLogger("__main__")


class AttHead(nn.Module):  # type: ignore
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, config: GATConfig) -> None:
        super().__init__()
        edge_dropout_p = config.edge_dropout_p
        embedding_dim = config.embedding_dim
        alpha = config.alpha
        nhid = config.nhid

        self.W = nn.Linear(embedding_dim, nhid, bias=False)

        self.a = nn.Parameter(torch.zeros(2 * nhid, 1, dtype=torch.float))  # type: ignore
        nn.init.xavier_uniform_(
            self.a, gain=nn.init.calculate_gain("leaky_relu", alpha)  # type: ignore
        )
        # self.a2 = torch.empty(out_features, 1, dtype=torch.float)
        # nn.init.xavier_uniform_(
        # self.a, gain=nn.init.calculate_gain("leaky_relu", alpha)  # type: ignore
        # )
        self.nhid = nhid
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim=1)
        self.edge_dropout = nn.Dropout(p=edge_dropout_p)

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:  # type: ignore

        h = self.W(input)
        N = h.size(0)

        a_input = torch.cat(
            [h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1
        ).view(N, -1, 2 * self.nhid)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # f_1 = h @ self.a1
        # f_2 = h @ self.a2
        # e = self.leakyrelu(f_1 + f_2.transpose(0, 1))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = self.softmax(attention)
        attention = self.edge_dropout(attention)
        h_prime = torch.matmul(attention, h)

        return h_prime

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + " ("
            + str(self.embedding_dim)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GATLayer(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig, concat: bool = True):
        nhid = config.nhid
        nheads = config.nheads
        embedding_dim = config.embedding_dim

        if not nhid * nheads == embedding_dim:
            raise Exception("nhid * nheads != out_features")
        super(GATLayer, self).__init__()

        self.attentions = nn.ModuleList([AttHead(config) for _ in range(nheads)])
        self.concat = concat
        self.elu = nn.ELU()

    def forward(self, h: Tensor, adj: Tensor) -> torch.Tensor:  # type: ignore
        lsatt_res = [att(h, adj) for att in self.attentions]
        if self.concat:
            h = torch.cat(lsatt_res, dim=1)
        else:
            h = torch.stack(lsatt_res, dim=0).mean(dim=0)
        h = self.elu(h)
        return h


class GATLayerWrapper(nn.Module):  # type: ignore
    def __init__(
        self, config: GATConfig, do_residual: bool = True, concat: bool = True
    ):
        if do_residual and not concat:
            raise Exception("Can't do residual connection when not concatting")
        super().__init__()
        do_layer_norm = config.do_layer_norm

        self.layer = GATLayer(config, concat=concat)

        if concat:
            out_features = config.embedding_dim
        else:
            out_features = config.nhid
        self.layer_norm = nn.LayerNorm(out_features)

        self.do_residual = do_residual
        self.do_layer_norm = do_layer_norm

    def forward(self, h: Tensor, adj: Tensor) -> torch.Tensor:  # type: ignore
        h_new = self.layer(h, adj)
        if self.do_residual:
            h_new = h_new + h  # Learnt not to do += for autograd
        if self.do_layer_norm:
            h_new = self.layer_norm(h_new)
        return h_new  # type: ignore


# TODO: Add positional embeddings here
class EmbeddingWrapper(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig, emb_init: Optional[torch.Tensor]):
        super().__init__()
        do_layer_norm = config.do_layer_norm
        vocab_size = config.vocab_size
        embedding_dim = config.embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        if emb_init is not None:
            logger.info(f"Initializing embeddings with pretrained embeddings ...")
            self.embedding.from_pretrained(emb_init)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.do_layer_norm = do_layer_norm

    def forward(self, tcword_id: torch.Tensor) -> torch.Tensor:  # type: ignore
        h = self.embedding(tcword_id)

        if self.do_layer_norm:
            h = self.layer_norm(h)
        return h


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
