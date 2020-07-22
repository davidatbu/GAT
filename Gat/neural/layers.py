"""nn.Module subclasses not "big enough" to be in models.py."""
from __future__ import annotations

import abc
import itertools
import logging
import math
import typing as T

import lazy_import
import numpy as np  # type: ignore
import torch
import typing_extensions as TT
from torch import nn
from torch import Tensor

if T.TYPE_CHECKING:
    from transformers import BertModel
    from transformers import AutoModel
    from transformers import AutoConfig
else:
    AutoModel = lazy_import.lazy_class("transformers.AutoModel ")
    AutoConfig = lazy_import.lazy_class("transformers.AutoConfig ")
    BertModel = lazy_import.lazy_class("transformers.BertModel ")

from Gat import configs
from Gat import utils
from Gat.data import vocabs
from Gat.data.tokenizers import Tokenizer


logger = logging.getLogger("__main__")


class PreparedBatch(T.NamedTuple):
    lslsnode_id: T.List[T.List[int]]
    batched_adj: torch.Tensor
    edge_types: torch.Tensor
    target: torch.Tensor


class GraphMultiHeadAttention(nn.Module):  # type: ignore
    def __init__(
        self, embed_dim: int, num_heads: int, edge_dropout_p: float,
    ):
        """Multihead attention that supports an arbitrary graph.

        Note that nn.MultiHeadAttention(the PyTorch native):
            1. doesn't support graph like inputs.  ie, it assumes
               every node/token is connected with every other token.
            2. Has no way to take edge features into attention calculation,
               or calculating node features.
        We do both things.
        """
        super().__init__()
        self.embed_dim = embed_dim
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

        self.softmax = nn.Softmax(dim=-1)
        # target: (B, L_left, L_right)
        self.dropout = nn.Dropout(p=edge_dropout_p)
        self._neg_infinity: torch.Tensor
        self._zero: torch.Tensor
        self.register_buffer("_neg_infinity", torch.tensor(-float("inf")))
        self.register_buffer("_zero", torch.tensor(0, dtype=torch.float))

    def forward(
        self,
        node_features: torch.Tensor,
        batched_adj: torch.Tensor,
        key_edge_features: T.Optional[torch.Tensor] = None,
        value_edge_features: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """.

        Args:
            batched_adj: (B, L, L)
                batched_adj[b, i, j] means there's a directed edge from the i-th node to
                the j-th node of the b-th graph in the batch.

                That means, node features of the j-th node will affect the calculation
                of the node features of the i-th node.
            node_features: (B, N, E)
            value_edge_features and key_edge_features: (B, L_left, L_right, E)
                edge_features[b, i, j, :] are the features of the directed edge from
                the i-th node to the j-th node of the b-th graph in the batch.

        Returns:
            result: (B, N, E)
        """
        if value_edge_features is not None:
            raise Exception("passing in value_edge_features is not yet implemented")

        Q = self.W_query(node_features)
        # (B, L, E)
        K = self.W_key(node_features)
        # (B, L, E)
        V = self.W_value(node_features)
        # (B, L, E)

        # Reshape using self.num_heads to compute probabilities headwize
        transed_Q = self._transpose_for_scores(Q)
        # (B, num_heads, L, head_size)
        transed_K = self._transpose_for_scores(K)
        # (B, num_heads, L, head_size)
        transed_V = self._transpose_for_scores(V)
        # (B, num_heads, L, head_size)

        # Compute node attention probability
        transed_K_for_matmul = transed_K.transpose(-2, -1)
        # (B, num_heads, head_size, L)
        att_scores = torch.matmul(transed_Q, transed_K_for_matmul)
        # (B, num_heads, L, L)

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
                transed_Q,
                key_edge_features,
            )
            # (B, num_heads, L, L)

            att_scores = att_scores + edge_att_scores

        att_scores /= math.sqrt(self.embed_dim)
        # Prepare  batched_adj to broadT.cast to head size
        B, L, L = batched_adj.size()
        batched_adj = batched_adj.unsqueeze(1).expand(B, self.num_heads, L, L)
        # (B, num_heads, L, L)

        # Inject the graph structure by setting non existent edges' scores to
        # negative infinity
        att_scores_after_graph_injected = torch.where(
            batched_adj, att_scores, self._neg_infinity
        )
        # (B, num_heads, L, L)
        att_probs = self.softmax(att_scores_after_graph_injected)
        # (B, num_heads, L, L)

        # If a node was not connected to any edge, we'll
        # have some nans in here, set the nans to zero
        att_probs = torch.where(att_probs != att_probs, self._zero, att_probs)
        # (B, num_heads, L, L)

        # Apply dropout
        att_probs = self.dropout(att_probs)
        # (B, num_heads, L, L)

        # Do convolution using attention
        new_node_features = torch.matmul(att_probs, transed_V)
        # (B, num_heads, L, head_size)

        if value_edge_features:
            raise NotImplementedError()

        # Reshape to concatenate the heads again
        new_node_features = new_node_features.transpose(1, 2)
        # (B, L, num_heads, head_size)

        B, L, _, _ = new_node_features.size()
        E = self.embed_dim
        new_node_features = new_node_features.reshape(B, L, E)
        # new_node_features: (B, N, E)

        # Pass them through W_o finally
        new_node_features = self.W_out(new_node_features)
        # new_node_features: (B, N, E)

        return new_node_features

    def __call__(
        self,
        node_features: torch.Tensor,
        batched_adj: torch.Tensor,
        key_edge_features: T.Optional[torch.Tensor] = None,
        value_edge_features: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Only here to help mypy with typechecking. Can be removed without harm."""
        return super().__call__(  # type: ignore
            node_features=node_features,
            batched_adj=batched_adj,
            key_edge_features=key_edge_features,
            value_edge_features=value_edge_features,
        )

    def _transpose_for_scores(self, W: torch.Tensor) -> torch.Tensor:
        """

        Args:
            W: (B, L, E)

        Returns:
            W: (B, self.num_heads, L, self.head_size)
        """

        B, L, E = W.size()
        W = W.view(B, L, self.num_heads, self.head_size)
        # (B, L, num_heads, head_size)

        transposed = W.transpose(1, 2)
        # (B, num_heads, L, head_size)
        return transposed


class Rezero(nn.Module):  # type: ignore
    def __init__(self, layer: nn.Module[torch.Tensor]):
        """Wrap a `layer` with a rezero conneciton."""
        super().__init__()
        self.rezero_weight = nn.Parameter(torch.tensor([0], dtype=torch.float))
        self.layer = layer

    def forward(self, h: torch.Tensor, **kwargs: T.Any) -> torch.Tensor:
        after_rezero = h + self.rezero_weight * self.layer(h, **kwargs)
        return after_rezero

    def __call__(self, h: torch.Tensor, **kwargs: T.Any) -> torch.Tensor:
        """Only here to help mypy with typechecking. Can be removed without harm."""
        return super().__call__(h=h, **kwargs)  # type: ignore


class ResidualAndNorm(nn.Module):  # type: ignore
    def __init__(self, dim: int, layer: nn.Module[torch.Tensor]):
        """Wrap a `layer` with a residual conneciton and layer normalization."""
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.layer = layer

    def forward(self, h: torch.Tensor, **kwargs: T.Any) -> torch.Tensor:
        after_residual = h + self.layer(h, **kwargs)
        after_layer_norm = self.layer_norm(after_residual)
        return after_layer_norm

    def __call__(self, h: torch.Tensor, **kwargs: T.Any) -> torch.Tensor:
        """Only here to help mypy with typechecking. Can be removed without harm."""
        return super().__call__(h=h, **kwargs)  # type: ignore


class FeedForward(nn.Module):  # type: ignore
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

    def __call__(self, h: torch.Tensor, **kwargs: T.Any) -> torch.Tensor:
        """Only here to help mypy with typechecking. Can be removed without harm."""
        return super().__call__(h=h, **kwargs)  # type: ignore


class GraphMultiHeadAttentionWrapped(nn.Module):  # type: ignore
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        edge_dropout_p: float,
        rezero_or_residual: TT.Literal["rezero", "residual"] = "rezero",
    ):
        """Wrapped with rezero or residual connection."""
        super().__init__()

        multihead_att = GraphMultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, edge_dropout_p=edge_dropout_p,
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
        batched_adj: torch.Tensor,
        key_edge_features: T.Optional[torch.FloatTensor] = None,
        value_edge_features: T.Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        return self.wrapper(
            node_features,
            batched_adj=batched_adj,
            key_edge_features=key_edge_features,
            value_edge_features=value_edge_features,
        )

    def __call__(
        self,
        node_features: torch.Tensor,
        batched_adj: torch.Tensor,
        key_edge_features: T.Optional[torch.Tensor] = None,
        value_edge_features: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Only here to help mypy with typechecking. Can be removed without harm."""
        return super().__call__(  # type: ignore
            node_features=node_features,
            batched_adj=batched_adj,
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

    def __call__(self, node_features: torch.Tensor) -> torch.Tensor:
        """Only here to help mypy with typechecking. Can be removed without harm."""
        return super().__call__(node_features=node_features)  # type: ignore


class Embedder(nn.Module, abc.ABC):  # type: ignore
    """An abstract class."""

    def __init__(self, vocab: T.Optional[vocabs.Vocab]) -> None:
        """An abstract embedder to specify input and output. # noqa: 
        """
        super().__init__()
        if vocab is not None:
            self._vocab = vocab

    @abc.abstractmethod
    def forward(self, lslstok_id: T.List[T.List[int]]) -> torch.Tensor:
        """.

        Args:
            tok_ids: (B, L)
        Returns:
            embs: (B, L, E)
        """
        pass

    def __call__(self, lslstok_id: T.List[T.List[int]]) -> torch.Tensor:
        """Only here to help mypy with typechecking. Can be removed without harm."""
        return super().__call__(lslstok_id)  # type: ignore[no-any-return]

    @abc.abstractproperty
    def embedding_dim(self) -> int:
        pass

    @abc.abstractproperty
    def max_seq_len(self) -> T.Optional[int]:
        """Maximum length of a sequence that can be outputted."""
        pass

    def prepare_for_embedder(self, lslstok_id: T.List[T.List[int]],) -> torch.Tensor:
        """Pad/truncate tokens, convert them to torch tensors and move them to device.

        The padding token is `self.padding_tok_id`. The length of the sequence after
        padding/truncating will be equal to the longest sequence in `lslstok_id` if
        `seq_len` is `None`.

        Args:
            lslstok_id:
            embedder:
            device:

        Returns:
            tok_ids: (B, L)
        """
        seq_len = max(map(len, lslstok_id))
        if self.max_seq_len is not None and self.max_seq_len > seq_len:
            seq_len = self.max_seq_len

        padding_tok_id = self._vocab.get_tok_id(self._vocab.padding_tok)
        padded_lslstok_id = [
            lstok_id[:seq_len] + [padding_tok_id] * max(0, seq_len - len(lstok_id))
            for lstok_id in lslstok_id
        ]
        tok_ids: torch.Tensor = torch.tensor(
            padded_lslstok_id, dtype=torch.long, device=next(self.parameters()).device,
        )
        # (B, L)
        return tok_ids

    def strip_after_embedder(self, embs: torch.Tensor) -> torch.Tensor:
        """Strip special tokens after passing through embedder.

        Currently, we use this only to remove the [cls] token from BERT. (We don't even
        remove the [sep] token with it).

        Args:
            embs: (B, L, E)

        Returns:
            embs: (B, L, E)
        """
        return embs


class BertEmbedder(Embedder):
    _model_name: T.Literal["bert-base-uncased"] = "bert-base-uncased"

    def __init__(self, vocab: vocabs.BertVocab) -> None:
        """Initialize bert model and so on."""
        self._vocab: vocabs.BertVocab
        super().__init__(vocab)
        self._embedding_dim = 768

        # Setup transformers BERT
        config = AutoConfig.from_pretrained(
            # TODO: Probalby don't need output_hidden_states for now
            pretrained_model_name_or_path=self._model_name,
            output_hidden_states=True,
        )
        self._model: BertModel = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self._model_name, config=config
        )
        self._cls_tok_id = self._vocab.get_tok_id(self._vocab.cls_tok)
        self._sep_tok_id = self._vocab.get_tok_id(self._vocab.sep_tok)
        self._padding_tok_id = self._vocab.get_tok_id(self._vocab.padding_tok)

    def forward(self, lslstok_id: T.List[T.List[int]]) -> torch.Tensor:
        """Get the BERT token embeddings.

        Assumes that tok_ids already is "well-prepared", ie, has the [cls] and [pad] and
        [sep] tokens in all the right places.

        Args:
            tok_ids: (B, L)
        Returns:
            embs: (B, L, E)
                B = len(tok_ids)
                L = self._model.config.max_position_embeddings
                And E is the num of dimensions the BERT vectors
        """
        tok_ids = self.prepare_for_embedder(lslstok_id)
        outputs = self._model(tok_ids)

        last_hidden_outputs = outputs[0]
        # (B, L, E)
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
    def max_seq_len(self) -> int:
        """Look at superclass doc."""
        return self._model.config.max_position_embeddings

    def prepare_for_embedder(self, lslstok_id: T.List[T.List[int]]) -> torch.Tensor:
        """Pad/truncate tokens, convert them to torch tensors and move them to device.

        This adds [cls], [sep], and obviously [pad] in the correct spots.
        The length of the sequence after padding/truncating will be equal to the longest
        sequence in `lslstok_id`, or embedder.max_seq_len, whichever is smaller.

        Args:
            lslstok_id:
            embedder:
            device:

        Returns:
            tok_ids: (B, L)
        """
        num_special_tokens = 2  # CLS and SEP
        non_special_tok_seq_len = max(map(len, lslstok_id))

        if (
            self.max_seq_len is not None
            and non_special_tok_seq_len > self.max_seq_len - num_special_tokens
        ):
            non_special_tok_seq_len = self.max_seq_len - num_special_tokens

        padded_lslstok_id = [
            [self._cls_tok_id]
            + lstok_id[:non_special_tok_seq_len]
            + [self._sep_tok_id]
            + [self._padding_tok_id] * max(0, non_special_tok_seq_len - len(lstok_id))
            for lstok_id in lslstok_id
        ]
        tok_ids: torch.Tensor = torch.tensor(
            padded_lslstok_id, dtype=torch.long, device=next(self.parameters()).device
        )
        # (B, L)

        return tok_ids

    def strip_after_embedder(self, embs: torch.Tensor) -> torch.Tensor:
        """Look at superclass doc.

        Args:
            embs: (B, L, E)

        Returns:
            embs: (B, L, E)
        """
        stripped = embs[:, 1:]
        # (B, L, E)
        return stripped


class BasicEmbedder(Embedder):
    @T.overload
    def __init__(self, vocab: vocabs.Vocab, *, embedding_dim: int,) -> None:
        ...

    @T.overload
    def __init__(self, vocab: vocabs.Vocab, *, pretrained_embs: torch.Tensor,) -> None:
        ...

    def __init__(
        self,
        vocab: vocabs.Vocab,
        pretrained_embs: T.Optional[torch.Tensor] = None,
        embedding_dim: T.Optional[int] = None,
    ) -> None:
        """Wrapper around `nn.Embedding` that conforms to `Embedder`."""
        super().__init__(vocab)

        embedder: nn.Embedding
        padding_tok_id = vocab.get_tok_id(vocab.padding_tok)
        if pretrained_embs is not None:
            assert embedding_dim is None
            embedder = nn.Embedding.from_pretrained(
                pretrained_embs, padding_idx=padding_tok_id
            )

        else:
            assert embedding_dim is not None
            assert vocab.vocab_size is not None
            embedder = nn.Embedding(
                num_embeddings=vocab.vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_tok_id,
            )

        self._embedder = embedder

    def forward(self, lslstok_id: T.List[T.List[int]]) -> torch.Tensor:
        """.

        Args:
            tok_ids: (*)

        Returns:
            embs: (*, E)
        """
        tok_ids = self.prepare_for_embedder(lslstok_id)
        res = self._embedder(tok_ids)
        # (*, E)
        return res

    @property
    def embedding_dim(self) -> int:
        return self._embedder.embedding_dim

    @property
    def max_seq_len(self) -> T.Optional[int]:
        """Look at superclass doc."""
        return None


class ReconcilingEmbedder(Embedder):
    def __init__(
        self,
        sub_word_vocab: vocabs.Vocab,
        word_vocab: vocabs.Vocab,
        sub_word_embedder: Embedder,
    ) -> None:
        """Pool over subword embeddings.

        Args:
            sub_word_vocab: We access `sub_word_vocab.tokenizer` and
                `sub_word_vocab.padding_tok_id`.
            word_vocab: We'll use `word_vcab.get_lslstok()`.
            sub_word_embedder: We use it to get subword embeddings, and access
                `sub_.wrod_embedder.max_seq_len`.
        """
        super().__init__(vocab=word_vocab)
        self._sub_word_vocab = sub_word_vocab
        self._sub_word_embedder = sub_word_embedder

        # NOTE: Since we're doing graph self attention(ie, only
        # nodes that were specified to be connected to node x
        # wil be considered in computing the feature of node x in a self attention
        # layer), we can set the padding_vec to be a requires_grad=False constant
        E = sub_word_embedder.embedding_dim
        self.register_buffer(
            "_padding_vec", torch.zeros([1, E], dtype=torch.float, requires_grad=False)
        )
        self._padding_vec: Tensor

    def forward(self, lslswordid: T.List[T.List[int]]) -> torch.Tensor:
        """Pool over subwords to create word embedding.

        Args:
            word_ids: (B, L)
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
        lslsword = self._vocab.get_lslstok(lslswordid)
        lssubwordids_per_word: T.List[T.List[T.List[int]]] = []

        for seq_num in range(len(lslswordid)):

            lsword = lslsword[seq_num]
            lswordid = lslswordid[seq_num]

            subwordids_per_word: T.List[T.List[int]] = []

            for word_num in range(len(lswordid)):
                word = lsword[word_num]
                wordid = lswordid[word_num]

                if wordid == self._vocab.get_tok_id(
                    self._vocab.padding_tok
                ):  # End of sequence
                    break

                subwordids_for_one_word = self._sub_word_vocab.tokenize_and_get_lstok_id(
                    word
                )
                subwordids_per_word.append(subwordids_for_one_word)
            lssubwordids_per_word.append(subwordids_per_word)

        # "Flat" sub word tokenization for each sequence
        lslssubwordid: T.List[T.List[int]] = []
        # The number of sub words in each word
        lslssubword_count: T.List[T.List[int]] = []

        max_subword_seq_len = float("inf")
        if self._sub_word_embedder.max_seq_len is not None:
            max_subword_seq_len = float(self._sub_word_embedder.max_seq_len)
        for subwordids_per_word in lssubwordids_per_word:
            # "Flatten" to one list
            lssubwordid: T.List[int] = []
            subword_counts: T.List[int] = []
            for subwordids_for_one_word in subwordids_per_word:
                # Check if subword tokenization exceeds the limit
                if len(lssubwordid) > max_subword_seq_len:
                    break
                lssubwordid.extend(subwordids_for_one_word)
                subword_counts.append(len(subwordids_for_one_word))
            lslssubword_count.append(subword_counts)
            lslssubwordid.append(lssubwordid)

        with_special_tok_subword_embs = self._sub_word_embedder(lslssubwordid)

        subword_embs = self._sub_word_embedder.strip_after_embedder(
            with_special_tok_subword_embs
        )

        pooled_word_embs = self.pool_sequences(subword_embs, lslssubword_count)
        return pooled_word_embs

    def pool_sequences(
        self, subword_embs: torch.Tensor, lslssubword_count: T.List[T.List[int]]
    ) -> torch.Tensor:
        """Pool over sub word embeddings to yield word embeddings.

        Args:
            subword_embs: (B, L, *)
            lslssubword_count: The number of subwords within each "word".

        Returns:
            word_embs: (B, L, *)
                L here will be max([ sum(subword_counts) for subword_counts in
                lslssubword_count ])
        """
        # Check sub word sequences lengths fit within subword_seq.shape
        max_subword_seq_len = max(
            [sum(subword_counts) for subword_counts in lslssubword_count]
        )
        assert max_subword_seq_len <= subword_embs.size(1)

        # Figure out the longest word seq length
        max_word_seq_len = max(map(len, lslssubword_count))

        B = subword_embs.size(0)
        L = max_word_seq_len
        E = subword_embs.size(2)
        embs = torch.zeros(
            [B, L, E], dtype=torch.float, device=next(self.parameters()).device
        )

        for batch_num, lssubword_count in enumerate(lslssubword_count):
            beg_and_end_indices = itertools.accumulate([0] + lssubword_count)
            beg_iterator, end_iterator = itertools.tee(beg_and_end_indices, 2)
            next(end_iterator)  # Consume the 0 at the beginning

            for word_idx, (beg, end) in enumerate(zip(beg_iterator, end_iterator)):
                embs[batch_num, word_idx] = subword_embs[batch_num, beg:end].mean(
                    dim=0, keepdim=True
                )

            # Fill up the rest with the subword padding vector
            word_seq_len = end  # after for loop finishes.

            embs[batch_num, word_seq_len:] = self._padding_vec

        return embs

    def prepare_for_embedder(self, lslstok_id: T.List[T.List[int]]) -> Tensor:
        assert False, "This should never be called."

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
        super().__init__(vocab=None)
        initial_max_length = 100
        self._embedding_dim = embedding_dim

        self.embs: torch.Tensor
        self.register_buffer("embs", self._create_embs(initial_max_length))

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, lslswordid: T.List[T.List[int]]) -> torch.Tensor:
        """Only needs the length information from lslstok_id.

        Args:
            token_ids: (B, L)
        Returns:
            positional_embs: (B, L, E)
                Where L is the size of the longest lstok_id in lslstok_id
        """
        cur_max = max(map(len, lslswordid))
        if cur_max > self.embs.size(1):
            logger.info(f"Increasing max position embedding to {cur_max}")
            self.register_buffer("embs", self._create_embs(cur_max))

        B = len(lslswordid)
        return self.embs[:, :cur_max].expand(B, -1, -1)

    def _create_embs(self, max_length: int) -> torch.Tensor:
        """Create the sinusodial embeddings.

        Returns:
            positional_embs: (1, L, E)

        This is called every time we get a request for a position that exceeds our
        cached version.
        """
        embs = torch.zeros(1, max_length, self.embedding_dim, dtype=torch.float)
        # (1, L, E)
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
        return embs

    @property
    def max_seq_len(self) -> T.Optional[int]:
        """Look at superclass doc."""
        return None


class GATLayered(nn.Module):  # type: ignore
    def __init__(
        self,
        config: configs.GATLayeredConfig,
        lsnode_feature_embedder: T.List[Embedder],
        key_edge_feature_embedder: T.Optional[nn.Embedding],
        value_edge_feature_embedder: T.Optional[nn.Embedding],
    ):
        """It's like the transformer.

        Layers of GraphMultiHeadAttention and FeedForward, each wrapped by a rezero
        connection (or a residual and a layer norm).

        Embedding dim is inferrred from the embedders passed.

        Args:
            num_heads: The number of heads for multi head attention.
            edge_dropout_p: Probablity with which to drop out edges.
            rezero_or_residual:
            intermediate_dim: In the feedforward layers.
            num_layers: Number of layers, where "FeedForward+SelfAttention" is one
                layer.
            feat_dropout_p:
            lsnode_feature_embedder: A list of embedders to look up the node ids in.
                Note that, for NLP tasks, this will most likely be a list of lenght two,
                contianing the word embedder, and the positional embedder.
            key_edge_feature_embedder: For incorporating edges into attention
                calculation.
                If set to None, won't incorporate edges into attention calculation.
            value_edge_feature_embedder: For incorporating edges into node convolution.
                Not yet implemented.
        """
        embedding_dim = lsnode_feature_embedder[0].embedding_dim

        assert embedding_dim % config.num_heads == 0
        for embedder in lsnode_feature_embedder:
            assert embedder.embedding_dim == embedding_dim

        for embdr in [value_edge_feature_embedder, key_edge_feature_embedder]:
            if embdr:
                assert embdr.embedding_dim == embedding_dim // config.num_heads
        super().__init__()

        self._lsnode_feature_embedder: T.Sequence[Embedder] = nn.ModuleList(  # type: ignore
            lsnode_feature_embedder
        )
        self._key_edge_feature_embedder = key_edge_feature_embedder
        self._value_edge_feature_embedder = value_edge_feature_embedder
        self._lsmultihead_att_wrapper: T.Iterable[GraphMultiHeadAttentionWrapped] = nn.ModuleList(  # type: ignore # noqa:
            [
                GraphMultiHeadAttentionWrapped(
                    embed_dim=embedding_dim,
                    num_heads=config.num_heads,
                    edge_dropout_p=config.edge_dropout_p,
                    rezero_or_residual=config.rezero_or_residual,
                )
                for _ in range(config.num_layers)
            ]
        )
        setattr(self._lsmultihead_att_wrapper, "debug_name", "lsmultihead_att_wrapper")
        self._lsfeed_forward_wrapper: T.Iterable[FeedForwardWrapped] = nn.ModuleList(  # type: ignore # noqa:
            [
                FeedForwardWrapped(
                    in_out_dim=embedding_dim,
                    intermediate_dim=config.intermediate_dim,
                    rezero_or_residual=config.rezero_or_residual,
                    feat_dropout_p=config.feat_dropout_p,
                )
                for _ in range(config.num_layers)
            ]
        )
        setattr(self._lsfeed_forward_wrapper, "debug_name", "lsfeedforward_wrapper")

        self._dropout = nn.Dropout(config.feat_dropout_p)

    def forward(self, prepared_batch: PreparedBatch) -> Tensor:
        node_features = self._lsnode_feature_embedder[0](prepared_batch.lslsnode_id)
        for embedder in self._lsnode_feature_embedder[1:]:
            node_features = node_features + embedder(prepared_batch.lslsnode_id)

        node_features = self._dropout(node_features)

        if self._key_edge_feature_embedder:
            assert prepared_batch.edge_types is not None
            key_edge_features = self._key_edge_feature_embedder(
                prepared_batch.edge_types
            )
            key_edge_features = self._dropout(key_edge_features)

        for multihead_att_wrapped, feed_forward_wrapped in zip(
            self._lsmultihead_att_wrapper, self._lsfeed_forward_wrapper
        ):
            node_features = multihead_att_wrapped(
                node_features=node_features,
                batched_adj=prepared_batch.batched_adj,
                key_edge_features=key_edge_features,
            )

            node_features = feed_forward_wrapped(node_features=node_features)

        return node_features

    def __call__(self, prepared_batch: PreparedBatch) -> Tensor:
        return super().__call__(  # type: ignore
            prepared_batch
        )


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
