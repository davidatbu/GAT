import abc
import logging
import math
import typing as T

import numpy as np
import torch
import typing_extensions as TT
from torch import nn
from transformers import AutoConfig  # type: ignore
from transformers import AutoModel

from Gat import utils
from Gat.data import tokenizers


logger = logging.getLogger("__main__")


# Look here:
# https://mypy.readthedocs.io/en/stable/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
# for why.
if T.TYPE_CHECKING:
    # TODO: This is totally not standard practice(not obvious)
    Module = nn.Module[torch.Tensor]
else:
    Module = nn.Module


class GraphMultiHeadAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        include_edge_features: bool,
        edge_dropout_p: float,
    ):
        """Why not use nn.MultiHeadAttention?
            1. Because it doesn't support graph like inputs.  Ie, it assumes
               every node/token is connected with every other token. We don't
            2. Because we are doing edge aware attention.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.include_edge_features = include_edge_features
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

        self.softmax = nn.Softmax(
            dim="N_right"  # type: ignore
        )  # It will be softmaxing (B, N_left, N_right)
        self.dropout = nn.Dropout(p=edge_dropout_p)
        self.register_buffer("neg_infinity", torch.tensor(-float("inf")))

    def forward(
        self,
        node_features: torch.FloatTensor,
        adj: torch.BoolTensor,
        key_edge_features: T.Optional[torch.FloatTensor] = None,
        value_edge_features: T.Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Shape:
            - Inputs:
            - adj: (B, N_left, N_right)
                adj[b, i, j] means there's a directed edge from the j-th node to
                the i-th node of the b-th graph in the batch.

                That means, node features of the j-th node will affect the calculation
                of the node features of the i-th node.
            - node_features: (B, N, E)
            - value_edge_features and key_edge_features: (B, N_left, N_right, head_size)
                edge_features[b, i, j, :] are the features of the directed edge from
                the j-th node to the i-th node of the b-th graph in the batch.

                That means, this E long vector will affect e_{ji} and z_j
            - Outputs:
            - result: (B, N, E)
        """

        if not self.include_edge_features and (
            key_edge_features or value_edge_features
        ):
            raise Exception("Not insantiated to include edge features.")
        if value_edge_features is not None:
            raise Exception("passing in value_edge_features is not yet implemented")

        # Refine names because linear layer erases the E dim name
        Q: torch.FloatTensor = self.W_query(node_features).refine_names(..., "E")  # type: ignore
        K: torch.FloatTensor = self.W_key(node_features).refine_names(..., "E")  # type: ignore
        V: torch.FloatTensor = self.W_value(node_features).refine_names(..., "E")  # type: ignore

        # Reshape using self.num_heads to compute probabilities headwize
        # Rename dim names to avoid N being duplicated in att_scores
        transed_Q = self._transpose_for_scores(Q)
        transed_K = self._transpose_for_scores(K)
        transed_V = self._transpose_for_scores(V)

        # Compute node attention probability
        att_scores = torch.matmul(
            transed_Q.rename(N="N_left"),  # type: ignore
            transed_K.rename(N="N_right").transpose("N_right", "head_size"),  # type: ignore
        )
        # att_scores: (B, head_size, N_left, N_right)

        if key_edge_features is not None:
            # Einstein notation used here .
            # A little complicated because of batching dimension.
            # Just keep in mind that:
            ##############################################
            # For edge_att_scores_{i,j} = dot_product(i-th query vector, with the edge feature of edge (j,i))
            ##############################################
            edge_att_scores = torch.einsum(  # type: ignore
                # b is batch
                # h is head number
                # n is node number
                # m is also node number
                # d is dimension of head (head size). This is the dimension that's being summed over
                # TODO: Look at doing this with BLAS operations, or may be even tensordot works bfaster than einsum
                "bhnd,bnmd->bhnm",
                transed_Q.rename(None),
                key_edge_features.rename(None),
            ).rename("B", "num_heads", "N_left", "N_right")

            att_scores = att_scores + edge_att_scores

        att_scores /= math.sqrt(self.embed_dim)
        # Prepare  adj to broadT.cast to head size
        adj = T.cast(
            torch.BoolTensor, adj.align_to("B", "num_heads", "N_left", "N_right")  # type: ignore
        )
        # Inject the graph structure by setting non existent edges' scores to negative infinity
        att_scores_names = T.cast(
            T.Optional[T.List[T.Optional[str]]], att_scores.names
        )  # I'm not sure why mypy needs this cast
        att_scores = torch.where(  # type: ignore
            adj.rename(None), att_scores.rename(None), self.neg_infinity  # type: ignore
        ).rename(*att_scores_names)
        att_probs = self.softmax(att_scores)
        # att_probs: (B, head_size, N_left, N_right)

        # Apply dropout
        names = att_probs.names
        att_probs = self.dropout(att_probs.rename(None)).rename(*names)  # type: ignore

        # Again combine values using attention
        new_node_features = torch.matmul(att_probs, transed_V)
        new_node_features = new_node_features.rename(N_left="N")  # type: ignore

        if value_edge_features:
            # Not yet implemented
            pass

        # Reshape to concatenate the heads again
        new_node_features = new_node_features.transpose("num_heads", "N")
        # new_node_features: (B, N, num_heads, head_size)
        new_node_features = new_node_features.flatten(("num_heads", "head_size"), "E")  # type: ignore
        # new_node_features: (B, N, E)

        # Pass them through W_o finally
        new_node_features = self.W_out(new_node_features).refine_names(..., "E")  # type: ignore

        return new_node_features  # type: ignore

    def _transpose_for_scores(self, W: torch.FloatTensor) -> torch.FloatTensor:
        W = W.unflatten(  # type: ignore
            "E", [("num_heads", self.num_heads), ("head_size", self.head_size)]
        )

        # Returning  (B, num_heads, N, head_size)
        return W.transpose("N", "num_heads")  # type: ignore


class Rezero(nn.Module):  # type: ignore
    def __init__(
        self, layer: nn.Module  # type: ignore
    ):
        super().__init__()
        self.rezero_weight = nn.Parameter(torch.tensor([0], dtype=torch.float))
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
        self,
        in_out_dim: int,
        intermediate_dim: int,
        out_bias: bool,
        feat_dropout_p: float,
    ):
        super().__init__()
        self.dropout = nn.Dropout(feat_dropout_p)
        self.W1 = nn.Linear(in_out_dim, intermediate_dim, bias=True)
        self.relu = nn.ReLU()
        self.W2 = nn.Linear(intermediate_dim, in_out_dim, bias=out_bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        after_dropout = self.dropout(h)
        after_ff = self.W2(self.dropout(self.relu(self.W1(after_dropout))))
        return after_ff


class GraphMultiHeadAttentionWrapped(nn.Module):  # type: ignore
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        include_edge_features: bool,
        edge_dropout_p: float,
        rezero_or_residual: TT.Literal["rezero", "residual"] = "rezero",
    ):
        "Wrapped with rezero or residual connection"
        super().__init__()

        multihead_att = GraphMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            include_edge_features=True,
            edge_dropout_p=edge_dropout_p,
        )

        self.wrapper: Module
        if rezero_or_residual == "rezero":
            self.wrapper = Rezero(layer=multihead_att)
        elif rezero_or_residual == "residual":
            self.wrapper = ResidualAndNorm(dim=embed_dim, layer=multihead_att)
        else:
            raise Exception('rezero_or_residual must be one of "rezero" or "residual"')

    def forward(
        self,
        node_features: torch.FloatTensor,
        adj: torch.BoolTensor,
        key_edge_features: T.Optional[torch.FloatTensor] = None,
        value_edge_features: T.Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        return self.wrapper(
            node_features,
            adj=adj,
            key_edge_features=key_edge_features,
            value_edge_features=value_edge_features,
        )


class FeedForwardWrapped(nn.Module):  # type: ignore
    def __init__(
        self,
        in_out_dim: int,
        intermediate_dim: int,
        feat_dropout_p: float,
        rezero_or_residual: TT.Literal["rezero", "residual"],
    ):
        super().__init__()

        out_bias = True
        if rezero_or_residual == "residual":
            out_bias = False
        ff = FeedForward(
            in_out_dim=in_out_dim,
            intermediate_dim=intermediate_dim,
            out_bias=out_bias,
            feat_dropout_p=feat_dropout_p,
        )

        self.wrapper: Module
        if rezero_or_residual == "rezero":
            self.wrapper = Rezero(layer=ff)
        elif rezero_or_residual == "residual":
            self.wrapper = ResidualAndNorm(dim=in_out_dim, layer=ff)
        else:
            raise Exception('rezero_or_residual must be one of "rezero" or "residual"')

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.wrapper(node_features)


class Embedder(Module, abc.ABC):
    def __init__(self, tokenizer: T.Optional[tokenizers.Tokenizer]) -> None:
        """Some embedders don't need to know the tokenizer(for example, positional embedding.
           I think, only BertEmbedder needs the tokenizer actually.
        """
        super().__init__()
        if tokenizer is not None:
            self._tokenizer = tokenizer
            self._validate_tokenizer()

    @abc.abstractmethod
    def forward(self, lsls_tok_id: T.List[T.List[int]]) -> torch.Tensor:
        """
        Args:
            token_ids:
        Returns:
            embs: (B, L, E)
        """
        pass

    @abc.abstractproperty
    def embedding_dim(self) -> int:
        pass

    def _validate_tokenizer(self) -> None:
        "Raise exception if self._tokenizer is not correct"
        pass


class BertEmbedder(Embedder):
    _model_name = "bert-base-uncased"

    def __init__(
        self, tokenizer: tokenizers.Tokenizer, last_how_many_layers: int = 4
    ) -> None:
        super().__init__(tokenizer=tokenizer)
        self._embedding_dim = 768 * last_how_many_layers

        # Setup transformers BERT
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self._model_name, output_hidden_states=True
        )
        self._model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self._model_name, config=config
        )

    def forward(self, lslstok_id: T.List[T.List[int]]) -> torch.Tensor:
        """
        Args:
            token_ids: (B, L)
        Returns:
            embs: (B, L, E)
        """
        breakpoint()

        raise NotImplementedError()
        # return self._embedder(token_ids)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _validate_tokenizer(self) -> None:
        acceptable_tokenizer_repr = f"WrappedBertTokenizer-{self._model_name}"
        if repr(self._tokenizer) != acceptable_tokenizer_repr:
            raise Exception(
                f"{repr(self._tokenizer)} is not a valid tokenizer for BertEmbedder."
                f" Only {acceptable_tokenizer_repr} is."
            )


class BasicEmbedder(Embedder):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        tokenizer: tokenizers.Tokenizer,
    ) -> None:
        super().__init__(tokenizer=tokenizer)
        self._embedder = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self._embedding_dim = embedding_dim
        self._padding_idx = padding_idx

    def forward(self, lslstok_id: T.List[T.List[int]]) -> torch.Tensor:
        """
        Args:
            token_ids:
        Returns:
            embs: (B, L, E)
                Where L is the length of the longest lstok_id in lslstok_id
        """

        padded_lslstok_id = utils.pad_lslsid(
            lslstok_id, padding_tok_id=self._padding_idx
        )

        token_ids = torch.tensor(padded_lslstok_id)
        res = self._embedder(token_ids)
        return res

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class PositionalEmbedder(Embedder):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        initial_max_length = 100
        self._embedding_dim = embedding_dim

        self.embs: torch.Tensor
        self.register_buffer("embs", self._create_embs(initial_max_length))

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, lslstok_id: T.List[T.List[int]]) -> torch.Tensor:
        """Only needs the length information from lslstok_id.

        Args:
            token_ids:
        Returns:
            positional_embs: (B, L, E)
                Where L is the size of the longest lstok_id in lslstok_id
        """
        cur_max = max(map(len, lslstok_id))
        if cur_max > self.embs.size("L"):
            logger.info(f"Increasing max position embedding to {cur_max}")
            self.register_buffer("embs", self._create_embs(cur_max))

        batch_size = len(lslstok_id)
        return self.embs[:, :cur_max].expand(batch_size, -1, -1)

    def _create_embs(self, max_length: int) -> torch.Tensor:
        """
        Returns:
            positional_embs: (1, L, E)
        """
        embs = torch.zeros(  # type: ignore
            1, max_length, self.embedding_dim, names=("B", "L", "E")
        )  # TODO: torch should recognize named tensors in their types
        position_enc = np.array(
            [
                [
                    pos / np.power(10000, 2 * (j // 2) / self.embedding_dim)
                    for j in range(self.embedding_dim)
                ]
                for pos in range(max_length)
            ]
        )
        embs[0, :, 0::2] = torch.from_numpy(np.sin(position_enc[:, 0::2]))
        embs[0, :, 1::2] = torch.from_numpy(np.cos(position_enc[:, 1::2]))
        embs.detach_()
        embs.requires_grad = False
        return embs  # type: ignore


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
