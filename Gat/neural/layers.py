import logging
import math
import typing as T

import numpy as np
import torch
from torch import nn

from ..config.base import GATConfig

logger = logging.getLogger("__main__")


# Look here: https://mypy.readthedocs.io/en/stable/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
# for why I have to do this
if T.TYPE_CHECKING:
    Module = nn.Module[torch.Tensor]
else:
    Module = nn.Module


class GraphMultiHeadSelfAttention(Module):
    def __init__(
        self, embed_dim: int, num_heads: int, num_edge_types: int, dropout: float = 0.0,
    ):
        """Why not use nn.MultiHeadAttention?
            1. Because it doesn't support graph like inputs.  Ie, it assumes
               every node/token is connected with every other token. We don't
            2. Because we are doing edge aware attention.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        if not self.head_size * num_heads == embed_dim:
            raise AssertionError("num_heads must be a divisor of embed_dim")

        self.W_key = nn.Linear(embed_dim, embed_dim)
        self.W_query = nn.Linear(embed_dim, embed_dim)
        self.W_value = nn.Linear(embed_dim, embed_dim)
        self.W_out = nn.Linear(embed_dim, embed_dim)
        # Take advantage of PyTorch's state of the art initialization techniques
        # by taking weights from a linear layer
        self.edge_values = nn.Parameter(
            torch.zeros(  # type: ignore
                num_edge_types, embed_dim, dtype=torch.float, names=("NumEdges", "E"),
            )
        )
        nn.init.kaiming_uniform_(self.edge_values, nonlinearity="relu")  # type: ignore
        # Used to incoprorate edge type to e_{ij}
        self.edge_queries = nn.Parameter(
            torch.zeros(  # type: ignore
                num_edge_types, embed_dim, dtype=torch.float, names=("NumEdges", "E"),
            )
        )
        nn.init.kaiming_uniform_(self.edge_queries, nonlinearity="relu")  # type: ignore

        self.softmax = nn.Softmax(
            dim="N_right"  # type: ignore
        )  # It will be softmaxing (B, N_left, N_right)

    def forward(
        self, node_features: torch.FloatTensor, adj: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """
        Shape:
            - Inputs:
            - node_features: (B, N, E)
            - adj: (B, N, N)

            - Outputs:
            - result: (B, N, E)
        """

        # Refine names because linear layer erases the E dim name
        Q: torch.FloatTensor = self.W_query(node_features).refine_names(..., "E")  # type: ignore
        K: torch.FloatTensor = self.W_key(node_features).refine_names(..., "E")  # type: ignore
        V: torch.FloatTensor = self.W_value(node_features).refine_names(..., "E")  # type: ignore

        # Reshape using self.num_heads to compute probabilities headwize
        # Rename dim names to avoid N being duplicated in att_scores
        # Look at shape of att_scores below for why each is either left or right
        transed_Q = self._transpose_for_scores(Q).rename(N="N_left")  # type: ignore
        transed_K = self._transpose_for_scores(K).rename(N="N_right")  # type: ignore
        transed_V = self._transpose_for_scores(V).rename(N="N_right")  # type: ignore

        # Compute attnention probabilibby
        att_scores = torch.matmul(
            transed_Q, transed_K.transpose("N_right", "head_size")
        )
        # att_scores: (B, head_size, N_left, N_right)
        att_scores /= math.sqrt(self.head_size)
        # Prepare  adj to broadT.cast to head wize
        adj = T.cast(
            torch.BoolTensor, adj.align_to("B", "num_heads", "N_left", "N_right")  # type: ignore
        )
        # Inject the graph structure by setting non existent edges' scores to negative infinity
        neg_inf = torch.tensor(-float("inf"))
        att_scores_names = T.cast(
            T.Optional[T.List[T.Optional[str]]], att_scores.names
        )  # I'm not sure why mypy needs this
        att_scores = torch.where(
            adj.rename(None), att_scores.rename(None), neg_inf
        ).rename(*att_scores_names)
        att_probs = self.softmax(att_scores)

        # Again, head wize, combine values using attention
        new_node_features = torch.matmul(att_probs, transed_V)
        new_node_features = new_node_features.rename(N_left="N")  # type: ignore

        # Reshape to concatenate the heads again
        new_node_features = new_node_features.transpose("num_heads", "N")
        # new_node_features: (B, N, num_heads, head_size)
        new_node_features = new_node_features.flatten(("num_heads", "head_size"), "E")  # type: ignore
        # new_node_features: (B, N, E)

        return new_node_features  # type: ignore

    def _transpose_for_scores(self, W: torch.FloatTensor) -> torch.FloatTensor:
        W = W.unflatten(  # type: ignore
            "E", [("num_heads", self.num_heads), ("head_size", self.head_size)]
        )

        # Returning  (B, num_heads, N, head_size)
        return W.transpose("N", "num_heads")  # type: ignore


class OldDotProductAttHead(Module):
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

    def forward(
        self, input: torch.Tensor, adj: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:

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


class OldMultiHeadAtt(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig):
        nhid = config.nhid
        nheads = config.nheads
        embedding_dim = config.embedding_dim
        nedge_type = config.nedge_type

        if not nhid * nheads == embedding_dim:
            raise Exception("nhid * nheads != out_features")
        super().__init__()

        self.W_o = nn.Linear(embedding_dim, embedding_dim)

        # THe +1 is because we might add an additional edge type
        edge_k_embedding = nn.Embedding(nedge_type + 1, 1)
        edge_v_embedding = nn.Embedding(nedge_type + 1, nhid)
        self.attentions = nn.ModuleList(
            [
                OldDotProductAttHead(
                    config,
                    edge_k_embedding=edge_k_embedding,
                    edge_v_embedding=edge_v_embedding,
                )
                for _ in range(nheads)
            ]
        )

    def forward(
        self, h: torch.Tensor, adj: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        lsatt_res = [att(h, adj, edge_type) for att in self.attentions]
        h = torch.cat(lsatt_res, dim=1)
        h = self.W_o(h)
        return h


class Rezero(nn.Module):  # type: ignore
    def __init__(
        self, layer: nn.Module  # type: ignore
    ):
        super().__init__()
        self.register_parameter(
            "rezero_weight", nn.Parameter(torch.tensor([0], dtype=torch.float))
        )
        self.layer = layer

    def forward(self, h: torch.Tensor, **kwargs: T.Any) -> torch.Tensor:
        after_rezero = h + self.rezero_weight * self.layer(h, **kwargs)
        return after_rezero  # type: ignore


class ResidualAndNorm(nn.Module):  # type: ignore
    def __init__(
        self, dim: int, layer: nn.Module  # type: ignore
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.layer = layer

    def forward(self, h: torch.Tensor, **kwargs: T.Any) -> torch.Tensor:
        after_residual = h + self.layer(h, **kwargs)
        after_layer_norm = self.layer_norm(after_residual)
        return after_layer_norm


class FeedForward(nn.Module):  # type: ignore
    def __init__(
        self, in_out_dim: int, intermediate_dim: int, out_bias: bool, dropout_p: float
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.W1 = nn.Linear(in_out_dim, intermediate_dim, bias=True)
        self.relu = nn.ReLU()
        self.W2 = nn.Linear(intermediate_dim, in_out_dim, bias=out_bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        after_dropout = self.dropout(h)
        after_ff = self.W2(self.dropout(self.relu(self.W1(after_dropout))))
        return after_ff


class MultiHeadAttWrapper(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig):
        super().__init__()

        multihead_att = OldMultiHeadAtt(config)

        self.wrapper: nn.Module  # type: ignore
        if config.do_rezero:
            self.wrapper = Rezero(layer=multihead_att)
        else:
            self.wrapper = ResidualAndNorm(
                dim=config.embedding_dim, layer=multihead_att
            )

    def forward(self, h: torch.Tensor, **kwargs: T.Any) -> torch.Tensor:
        return self.wrapper(h, **kwargs)  # type: ignore


class FeedForwardWrapper(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig):
        super().__init__()

        out_bias = True
        if config.do_layer_norm:
            out_bias = False
        ff = FeedForward(
            in_out_dim=config.embedding_dim,
            intermediate_dim=config.intermediate_dim,
            out_bias=out_bias,
            dropout_p=config.feat_dropout_p,
        )

        self.wrapper: nn.Module  # type: ignore
        if config.do_rezero:
            self.wrapper = Rezero(layer=ff)
        else:
            self.wrapper = ResidualAndNorm(dim=config.embedding_dim, layer=ff)

    def forward(self, h: torch.Tensor, **kwargs: T.Any) -> torch.Tensor:
        return self.wrapper(h, **kwargs)  # type: ignore


class EmbeddingWrapper(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig, emb_init: T.Optional[torch.Tensor]):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=0,
        )
        self.position_embedding = PositionEmbedding(config)

        if emb_init is not None:
            logger.info("Initializing embeddings with pretrained embeddings ...")
            self.embedding.from_pretrained(emb_init)

        self.wrapper: T.Callable[[torch.Tensor], torch.Tensor]
        if config.do_layer_norm:
            self.wrapper = nn.LayerNorm(normalized_shape=config.embedding_dim)
        elif config.do_rezero:
            self.wrapper = lambda x: x

    def forward(
        self, tcword_id: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        assert len(tcword_id) == len(position_ids)
        h = self.embedding(tcword_id)
        after_pos_embed = h + self.position_embedding(position_ids)

        after_potentially_norm = self.wrapper(after_pos_embed)
        return after_potentially_norm


class PositionEmbedding(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig) -> None:
        super().__init__()
        initial_max_length = 100
        self.embedding_dim = config.embedding_dim

        self.embs: torch.Tensor
        self.register_buffer(
            "embs", self.create_embs(initial_max_length, self.embedding_dim)
        )

    @staticmethod
    def create_embs(max_length: int, embedding_dim: int) -> torch.Tensor:
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

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        cur_max = int(position_ids.max().item())
        if cur_max > self.embs.size(0):
            logger.info(f"Increasing max position embedding to {cur_max}")
            self.register_buffer("embs", self.create_embs(cur_max, self.embedding_dim))
        return self.embs[position_ids]


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
