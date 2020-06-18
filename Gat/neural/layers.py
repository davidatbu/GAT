"""nn.Module subclasses not "big enough" to be in models.py."""
from __future__ import annotations

import abc
import itertools
import logging
import math
import typing as T

import numpy as np
import torch
import typing_extensions as TT
from torch import nn
from transformers import AutoConfig
from transformers import AutoModel
from transformers import BertModel

from Gat import data
from Gat import utils

logger = logging.getLogger("__main__")


# Look at
# https://mypy.readthedocs.io/en/stable/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
# for why this is necessary.
if T.TYPE_CHECKING:
    nnModule = nn.Module[torch.Tensor]
else:
    nnModule = nn.Module


class GraphMultiHeadAttention(nnModule):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        include_edge_features: bool,
        edge_dropout_p: float,
    ):
        """Multihead attention that supports an arbitrary graph.

        Note that nn.MultiHeadAttention(the PyTorch native):
            1. doesn't support graph like inputs.  Ie, it assumes
               every node/token is connected with every other token.
            2. Has no way to take edge features into attention calculation,
               or calculating node features.
        We do both things.
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
        """.

        Args:
            adj: (B, N_left, N_right)
                adj[b, i, j] means there's a directed edge from the j-th node to
                the i-th node of the b-th graph in the batch.

                That means, node features of the j-th node will affect the calculation
                of the node features of the i-th node.
            node_features: (B, N, E)
            value_edge_features and key_edge_features: (B, N_left, N_right, head_size)
                edge_features[b, i, j, :] are the features of the directed edge from
                the j-th node to the i-th node of the b-th graph in the batch.

                That means, this E long vector will affect e_{ji} and z_j
        Returns:
            result: (B, N, E)
        """
        if not self.include_edge_features and (
            key_edge_features or value_edge_features
        ):
            raise Exception("Not insantiated to include edge features.")
        if value_edge_features is not None:
            raise Exception("passing in value_edge_features is not yet implemented")

        # Refine names because linear layer erases the E dim name
        Q: torch.FloatTensor = self.W_query(node_features).refine_names(..., "E")  # type: ignore # noqa:
        K: torch.FloatTensor = self.W_key(node_features).refine_names(..., "E")  # type: ignore # noqa:
        V: torch.FloatTensor = self.W_value(node_features).refine_names(..., "E")  # type: ignore # noqa:

        # Reshape using self.num_heads to compute probabilities headwize
        # Rename dim names to avoid N being duplicated in att_scores
        transed_Q = self._transpose_for_scores(Q)
        transed_K = self._transpose_for_scores(K)
        transed_V = self._transpose_for_scores(V)

        # Compute node attention probability
        att_scores = torch.matmul(
            transed_Q.rename(N="N_left"),  # type: ignore
            transed_K.rename(N="N_right").transpose("N_right", "head_size"),  # type: ignore # noqa:
        )
        # att_scores: (B, head_size, N_left, N_right)

        if key_edge_features is not None:
            # Einstein notation used here .
            # A little complicated because of batching dimension.
            # Just keep in mind that:
            # edge_att_scores_{i,j} = dot_product(
            #   i-th query vector,
            #   the edge feature of edge (j,i)
            # )
            edge_att_scores = torch.einsum(  # type: ignore
                # b is batch
                # h is head number
                # n is node number
                # m is also node number
                # d is dimension of head (head size).
                #      This is the dimension that's being summed over
                # TODO: Look at doing this with BLAS operations,
                #        or may be tensordot.
                "bhnd,bnmd->bhnm",
                transed_Q.rename(None),
                key_edge_features.rename(None),
            ).rename("B", "num_heads", "N_left", "N_right")

            att_scores = att_scores + edge_att_scores

        att_scores /= math.sqrt(self.embed_dim)
        # Prepare  adj to broadT.cast to head size
        adj = T.cast(
            torch.BoolTensor, adj.align_to("B", "num_heads", "N_left", "N_right")  # type: ignore # noqa:
        )
        # Inject the graph structure by setting non existent edges' scores to
        # negative infinity
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
        new_node_features = new_node_features.flatten(("num_heads", "head_size"), "E")  # type: ignore # noqa:
        # new_node_features: (B, N, E)

        # Pass them through W_o finally
        new_node_features = self.W_out(new_node_features).refine_names(..., "E")  # type: ignore # noqa:

        return new_node_features  # type: ignore

    def _transpose_for_scores(self, W: torch.FloatTensor) -> torch.FloatTensor:
        W = W.unflatten(  # type: ignore
            "E", [("num_heads", self.num_heads), ("head_size", self.head_size)]
        )

        # Returning  (B, num_heads, N, head_size)
        return W.transpose("N", "num_heads")  # type: ignore


class Rezero(nnModule):
    def __init__(self, layer: nn.Module[torch.Tensor]):
        """Wrap a `layer` with a rezero conneciton."""
        super().__init__()
        self.rezero_weight = nn.Parameter(torch.tensor([0], dtype=torch.float))
        self.layer = layer

    def forward(self, h: torch.Tensor, **kwargs: T.Any) -> torch.Tensor:
        after_rezero = h + self.rezero_weight * self.layer(h, **kwargs)
        return after_rezero


class ResidualAndNorm(nnModule):
    def __init__(self, dim: int, layer: nn.Module[torch.Tensor]):
        """Wrap a `layer` with a residual conneciton and layer normalization."""
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.layer = layer

    def forward(self, h: torch.Tensor, **kwargs: T.Any) -> torch.Tensor:
        after_residual = h + self.layer(h, **kwargs)
        after_layer_norm = self.layer_norm(after_residual)
        return after_layer_norm


class FeedForward(nnModule):
    def __init__(
        self,
        in_out_dim: int,
        intermediate_dim: int,
        out_bias: bool,
        feat_dropout_p: float,
    ):
        """The feedforward from the Transformer.

        Alternatively described as a convolution of width and stride 1.
        """
        super().__init__()
        self.dropout = nn.Dropout(feat_dropout_p)
        self.W1 = nn.Linear(in_out_dim, intermediate_dim, bias=True)
        self.relu = nn.ReLU()
        self.W2 = nn.Linear(intermediate_dim, in_out_dim, bias=out_bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        after_dropout = self.dropout(h)
        after_ff = self.W2(self.dropout(self.relu(self.W1(after_dropout))))
        return after_ff


class GraphMultiHeadAttentionWrapped(nnModule):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        include_edge_features: bool,
        edge_dropout_p: float,
        rezero_or_residual: TT.Literal["rezero", "residual"] = "rezero",
    ):
        """Wrapped with rezero or residual connection."""
        super().__init__()

        multihead_att = GraphMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            include_edge_features=True,
            edge_dropout_p=edge_dropout_p,
        )

        self.wrapper: nn.Module[torch.Tensor]
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


class FeedForwardWrapped(nnModule):
    def __init__(
        self,
        in_out_dim: int,
        intermediate_dim: int,
        feat_dropout_p: float,
        rezero_or_residual: TT.Literal["rezero", "residual"],
    ):
        """Wrapped with a rezero or residual connection."""
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

        self.wrapper: nn.Module[torch.Tensor]
        if rezero_or_residual == "rezero":
            self.wrapper = Rezero(layer=ff)
        elif rezero_or_residual == "residual":
            self.wrapper = ResidualAndNorm(dim=in_out_dim, layer=ff)
        else:
            raise Exception('rezero_or_residual must be one of "rezero" or "residual"')

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.wrapper(node_features)


class Embedder(nnModule, abc.ABC):
    """An abstract class.

    The intended behavior for all subclassers is that their `forward()` accepts a
    T.List[T.List[str]], pads each T.List[str] to the maximum T.List[str] present (or
    the maximum that the underlying embedding source supports, ie BERT) and then outputs
    a torch.Tensor.
    """

    def __init__(self, vocab: T.Optional[data.Vocab] = None) -> None:
        """An abstract embedder to specify input and output. # noqa: 

           Args:
               vocab: Needed to access vocab.padding_tok_id, but not needed for "PositionalEmbedder".
        """
        super().__init__()
        if vocab is not None:
            self._vocab = vocab

    @abc.abstractmethod
    def forward(self, lsls_tok_id: T.List[T.List[int]]) -> torch.Tensor:
        """.

        Args:
            token_ids:
        Returns:
            embs: (B, L, E)
        """
        pass

    @abc.abstractproperty
    def embedding_dim(self) -> int:
        pass

    @abc.abstractproperty
    def max_seq_len(self) -> T.Optional[int]:
        """Maximum length of a sequence that can be outputted."""
        pass


class BertEmbedder(Embedder):
    _model_name: T.Literal["bert-base-uncased"] = "bert-base-uncased"

    def __init__(self, vocab: data.BertVocab, last_how_many_layers: int = 4,) -> None:
        """Initialize bert model and so on."""
        super().__init__(vocab=vocab)
        self._embedding_dim = 768 * last_how_many_layers

        # Setup transformers BERT
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self._model_name, output_hidden_states=True
        )
        self._model: BertModel = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self._model_name, config=config
        )

        self._vocab: data.BertVocab

    def forward(self, lslstok_id: T.List[T.List[int]]) -> torch.Tensor:
        """Get the BERT token embeddings.

        This does not return embeddings for the [CLS] token, or the [SEP] token.
        lslstok_id should not contain the [CLS] token, or the [SEP] token, either.

        Args:
            token_ids:
        Returns:
            embs: (B, L, E)
                B = len(token_ids)
                L = self._model.config.max_position_embeddings
                And E is the num of dimensions the BERT vectors
        """
        cls_tok_id = self._vocab.tokenizer.unwrapped_tokenizer.cls_token_id
        sep_tok_id = self._vocab.tokenizer.unwrapped_tokenizer.sep_token_id

        unpadded_with_special_toks_lslstok_id = [
            [cls_tok_id] + lstok_id + [sep_tok_id] for lstok_id in lslstok_id
        ]

        padded_lslstok_id = utils.pad_lslsid(
            unpadded_with_special_toks_lslstok_id,
            padding_tok_id=self._vocab.padding_tok_id,
            max_len=self._model.config.max_position_embeddings,
        )
        input_ids = torch.tensor(
            padded_lslstok_id,
            dtype=torch.long,
            device=next(self._model.parameters()).device,
        )
        outputs = self._model(input_ids)

        last_hidden_outputs = outputs[0]
        last_hidden_outputs = last_hidden_outputs.refine_names(
            "B", "L", "E"  # type: ignore
        )

        # Remove [CLS] and [SEP] embeddings, truncate to max sequence length
        max_length = min(self.max_seq_len, max(map(len, lslstok_id)))
        assert max_length is not None
        last_hidden_outputs = last_hidden_outputs[:, 1 : max_length + 1, :]

        return last_hidden_outputs

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

    @property
    def max_seq_len(self) -> T.Optional[int]:
        """Look at superclass doc."""
        # The -2 here is because [CLS] and [SEP] actually count as tokens themselves for
        # BERT
        return self._model.config.max_position_embeddings - 2


class BasicEmbedder(Embedder):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        vocab: data.BasicVocab,
    ) -> None:
        """Wrapper around `nn.Embedding` that conforms to `Embedder`."""
        super().__init__(vocab=vocab)
        self._embedder = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=vocab.padding_tok_id,
        )
        self._embedding_dim = embedding_dim
        self._padding_idx = padding_idx

    def forward(self, lslstok_id: T.List[T.List[int]]) -> torch.Tensor:
        """.

        Args:
            token_ids:
        Returns:
            embs: (B, L, E)
                  Where L is the length of the longest lstok_id in lslstok_id
        """
        padded_lslstok_id = utils.pad_lslsid(
            lslstok_id, padding_tok_id=self._vocab.padding_tok_id
        )

        token_ids = torch.tensor(padded_lslstok_id)
        res = self._embedder(token_ids)
        return res

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def max_seq_len(self) -> T.Optional[int]:
        """Look at superclass doc."""
        return None


class ReconcilingEmbedder(Embedder):
    def __init__(
        self,
        sub_word_vocab: data.Vocab,
        word_vocab: data.Vocab,
        sub_word_embedder: Embedder,
    ) -> None:
        """Pool over subword embeddings.

        Args:
            sub_word_vocab: We access `sub_word_vocab.tokenizer` and
                `sub_word_vocab.padding_tok_id`.
            word_vocab: We'll use `word_vcab.batch_get_toks()`.
            sub_word_embedder: We use it to get subword embeddings, and access
                `sub_.wrod_embedder.max_seq_len`.
        """
        super().__init__()
        self._sub_word_vocab = sub_word_vocab
        self._word_vocab = word_vocab
        self._sub_word_embedder = sub_word_embedder

    def forward(self, lslsword_id: T.List[T.List[int]]) -> torch.Tensor:
        """Tokenize using the two tokenizers, pool over subwords to create word embedding.

        Args:
            lstxt: A list of sentences.
        Returns:
            embedding: (B, L, E)
                       B = len(lstxt)
                       L is computed like this:
                       The sentences are truncated to the last word whose complete sub
                       word tokenization "fits inside" the maximum number of sub word
                       tokens allowed by `sub_word_embedder` per sequence.  L is the
                       number of words in the sentence with the most word tokens after
                       the truncation described above.

                       For example,

                       lssent = ["love embeddings"]
                       lslsword_id = [ [1, 2] ]

                       sub_word_tokenization = [ "love", "embed", "#dings" ]
                       word_tokenization = [ "love", "embeddings" ]
                       sub_word_embedder.max_seq_len == 2 # True
                       L == 1 # True, since "embeddings" doesn't fit within 2 sub word
                              # tokens
        """
        lswords = self._word_vocab.batch_get_toks(lslsword_id)
        lssubwordids_per_word = [
            [self._sub_word_vocab.tokenize_and_get_tok_ids(word) for word in words]
            for words in lswords
        ]
        # "Flat" sub word tokenization for each sequence
        lssubwordids: T.List[T.List[int]] = []
        # The number of sub words in each word
        lssubword_counts: T.List[T.List[int]] = []

        max_subword_seq_len = float("inf")
        if self._sub_word_embedder.max_seq_len is not None:
            max_subword_seq_len = float(self._sub_word_embedder.max_seq_len)
        for subwordids_per_word in lssubwordids_per_word:
            # "Flatten" to one list
            subwordids: T.List[int] = []
            subword_counts: T.List[int] = []
            for subwordids_for_one_word in subwordids_per_word:
                # Check if subword tokenization exceeds the limit
                if len(subwordids) > max_subword_seq_len:
                    break
                subwordids.extend(subwordids_for_one_word)
                subword_counts.append(len(subwordids_for_one_word))
            lssubword_counts.append(subword_counts)
            lssubwordids.append(subwordids)

        subword_embs = self._sub_word_embedder(lssubwordids)

        pooled_word_embs = self.pool_sequences(subword_embs, lssubword_counts)
        return pooled_word_embs

    def pool_sequences(
        self, subword_seqs: torch.Tensor, lssubword_counts: T.List[T.List[int]]
    ) -> torch.Tensor:
        """Pool over sub word embeddings to yield word embeddings.

        Args:
            subword_seqs: (B, L, ...)
            lssubword_counts: The number of subwords within each "word".

        Returns:
            word_seqs: (B, L, ...)
                L here will be max([ sum(subword_counts) for subword_counts in
                lssubword_counts ])
        """
        # Check sub word sequences lengths fit within subword_seq.shape
        max_subword_seq_len = max(
            [sum(subword_counts) for subword_counts in lssubword_counts]
        )
        assert max_subword_seq_len <= subword_seqs.size("L")

        # Figure out the longest word seq length
        max_word_seq_len = max(map(len, lssubword_counts))

        # Get the padding vector
        padding_vec = self._sub_word_embedder([[self._sub_word_vocab.padding_tok_id]])
        padding_vec = padding_vec.squeeze()

        # Word embeddings per seq
        lsword_seq: T.List[torch.Tensor] = []

        for subword_seq, subword_counts in zip(subword_seqs, lssubword_counts):
            beg_and_end_indices = itertools.accumulate([0] + subword_counts)
            beg_iterator, end_iterator = itertools.tee(beg_and_end_indices, 2)

            next(end_iterator)  # Consume the 0 at the beginning
            word_seq_len = len(subword_counts)
            word_seq = torch.stack(
                [
                    subword_seq[beg:end].mean(dim=0).rename(None)
                    for beg, end in zip(beg_iterator, end_iterator)
                ]
                + [padding_vec] * (max_word_seq_len - word_seq_len)
            )

            # TODO: Remove
            assert len(word_seq) == len(subword_counts)
            lsword_seq.append(word_seq)
        word_seqs = torch.stack(lsword_seq).rename("B", "L", "E")  # type: ignore
        return word_seqs

    @property
    def embedding_dim(self) -> int:
        return self._sub_word_embedder.embedding_dim

    @property
    def max_seq_len(self) -> T.Optional[int]:
        """Look at superclass doc."""
        return None


class PositionalEmbedder(Embedder):
    def __init__(self, embedding_dim: int) -> None:
        """Yields positional encoding.

        Note the input is only used for shape/position information.
        """
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
        """Create the sinusodial embeddings.

        Returns:
            positional_embs: (1, L, E)

        This is called every time we get a request for a position that exceeds our
        cached version.
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

    @property
    def max_seq_len(self) -> T.Optional[int]:
        """Look at superclass doc."""
        return None


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
