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
from torch import Tensor
from transformers import AutoConfig
from transformers import AutoModel
from transformers import BertModel

from Gat import data


logger = logging.getLogger("__main__")


# Look at
# https://mypy.readthedocs.io/en/stable/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
# for why this is necessary.


class GraphMultiHeadAttention(nn.Module):  # type: ignore
    def __init__(
        self, embed_dim: int, num_heads: int, edge_dropout_p: float,
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
        node_features: torch.Tensor,
        adj: torch.Tensor,
        key_edge_features: T.Optional[torch.Tensor] = None,
        value_edge_features: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        return new_node_features

    def __call__(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        key_edge_features: T.Optional[torch.Tensor] = None,
        value_edge_features: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Only here to help mypy with typechecking. Can be removed without harm."""
        return super().__call__(  # type: ignore
            node_features=node_features,
            adj=adj,
            key_edge_features=key_edge_features,
            value_edge_features=value_edge_features,
        )

    def _transpose_for_scores(self, W: torch.FloatTensor) -> torch.FloatTensor:
        W = W.unflatten(  # type: ignore
            "E", [("num_heads", self.num_heads), ("head_size", self.head_size)]
        )

        # Returning  (B, num_heads, N, head_size)
        return W.transpose("N", "num_heads")  # type: ignore


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

    def __call__(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        key_edge_features: T.Optional[torch.Tensor] = None,
        value_edge_features: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Only here to help mypy with typechecking. Can be removed without harm."""
        return super().__call__(  # type: ignore
            node_features=node_features,
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

    def __init__(self) -> None:
        """An abstract embedder to specify input and output. # noqa: 
        """
        # TODO: Why do we need vocab here?
        super().__init__()

    @abc.abstractmethod
    def forward(self, tok_ids: torch.LongTensor) -> torch.Tensor:
        """.

        Args:
            tok_ids: (B, L)
        Returns:
            embs: (B, L, E)
        """
        pass

    def __call__(self, tok_ids: torch.LongTensor) -> torch.Tensor:
        """Only here to help mypy with typechecking. Can be removed without harm."""
        return super().__call__(tok_ids)  # type: ignore

    @abc.abstractproperty
    def embedding_dim(self) -> int:
        pass

    @abc.abstractproperty
    def max_seq_len(self) -> T.Optional[int]:
        """Maximum length of a sequence that can be outputted."""
        pass


class BertEmbedder(Embedder):
    _model_name: T.Literal["bert-base-uncased"] = "bert-base-uncased"

    def __init__(self, vocab: data.BertVocab) -> None:
        """Initialize bert model and so on."""
        super().__init__()
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

        self._vocab: data.BertVocab

    def forward(self, tok_ids: torch.LongTensor) -> torch.Tensor:
        """Get the BERT token embeddings.

        Assumes that tok_ids already is "well-prepared", ie, has the [CLS] and [PAD] and
        [SEP] tokens in all the right places.

        Args:
            tok_ids: (B, L)
        Returns:
            embs: (B, L, E)
                B = len(tok_ids)
                L = self._model.config.max_position_embeddings
                And E is the num of dimensions the BERT vectors
        """
        assert tok_ids.names == ("B", "L")  # type: ignore
        if tok_ids.size("L") > self.max_seq_len:
            raise Exception(f"BertEmbedder supports input with L<={self.max_seq_len}")

        outputs = self._model(tok_ids.rename(None))

        last_hidden_outputs = outputs[0]
        last_hidden_outputs = last_hidden_outputs.refine_names(
            "B", "L", "E"  # type: ignore
        )

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


class BasicEmbedder(Embedder):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int,
    ) -> None:
        """Wrapper around `nn.Embedding` that conforms to `Embedder`."""
        super().__init__()
        self._embedder = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self._embedding_dim = embedding_dim
        self._padding_idx = padding_idx

    def forward(self, tok_ids: torch.LongTensor) -> torch.Tensor:
        """.

        This doesn't check if tok_ids is less than 512 long along the L dimension,
        which is necessary for BERT.

        Args:
            tok_ids: (B, L)

        Returns:
            embs: (B, L, E)
        """
        assert tok_ids.names == ("B", "L")  # type: ignore
        res = self._embedder(tok_ids.rename(None)).rename("B", "L", "E")  # type: ignore
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

        # Get the padding vector once, because we'll need it again and again
        prepared_pad_tok_id = sub_word_vocab.prepare_for_embedder(
            [[sub_word_vocab.padding_tok_id]], sub_word_embedder
        )
        with_special_toks = self._sub_word_embedder(prepared_pad_tok_id)
        without_special_toks = self._sub_word_vocab.strip_after_embedder(
            with_special_toks
        )

        no_padding_toks = without_special_toks[:, :1]
        self._padding_vec = no_padding_toks.squeeze()

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
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
        lslswordid: T.List[T.List[int]] = word_ids.cpu().numpy().tolist()
        lslsword = self._word_vocab.batch_get_toks(lslswordid)
        lssubwordids_per_word: T.List[T.List[T.List[int]]] = []

        for seq_num in range(len(lslswordid)):

            lsword = lslsword[seq_num]
            lswordid = lslswordid[seq_num]

            subwordids_per_word: T.List[T.List[int]] = []

            for word_num in range(len(lswordid)):
                word = lsword[word_num]
                wordid = lswordid[word_num]

                if wordid == self._word_vocab.padding_tok_id:  # End of sequence
                    break

                subwordids_for_one_word = self._sub_word_vocab.tokenize_and_get_tok_ids(
                    word
                )
                subwordids_per_word.append(subwordids_for_one_word)
            lssubwordids_per_word.append(subwordids_per_word)

        # "Flat" sub word tokenization for each sequence
        lslssubwordid: T.List[T.List[int]] = []
        # The number of sub words in each word
        lssubword_counts: T.List[T.List[int]] = []

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
            lssubword_counts.append(subword_counts)
            lslssubwordid.append(lssubwordid)

        prepared_subwordids = self._sub_word_vocab.prepare_for_embedder(
            lslssubwordid,
            embedder=self._sub_word_embedder,
            device=next(self.parameters()).device,
        )

        with_special_tok_subword_embs = self._sub_word_embedder(prepared_subwordids)
        subword_embs = self._sub_word_vocab.strip_after_embedder(
            with_special_tok_subword_embs
        )

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
                + [self._padding_vec] * (max_word_seq_len - word_seq_len)
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

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Only needs the length information from lslstok_id.

        Args:
            token_ids: (B, L)
        Returns:
            positional_embs: (B, L, E)
                Where L is the size of the longest lstok_id in lslstok_id
        """
        cur_max = token_ids.size("L")
        if cur_max > self.embs.size("L"):
            logger.info(f"Increasing max position embedding to {cur_max}")
            self.register_buffer("embs", self._create_embs(cur_max))

        batch_size = token_ids.size("B")
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


class GATLayered(nn.Module):  # type: ignore
    def __init__(
        self,
        num_heads: int,
        edge_dropout_p: float,
        rezero_or_residual: T.Literal["rezero", "residual"],
        intermediate_dim: int,
        num_layers: int,
        feat_dropout_p: float,
        lsnode_feature_embedder: T.List[Embedder],
        key_edge_feature_embedder: T.Optional[Embedder],
        value_edge_feature_embedder: T.Optional[Embedder],
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
        for embedder in (
            lsnode_feature_embedder
            + ([key_edge_feature_embedder] if key_edge_feature_embedder else [])
            + ([value_edge_feature_embedder] if value_edge_feature_embedder else [])
        ):
            assert embedder.embedding_dim == embedding_dim
        super().__init__()

        self._lsnode_feature_embedder = lsnode_feature_embedder
        self._key_edge_feature_embedder = key_edge_feature_embedder
        self._value_edge_feature_embedder = value_edge_feature_embedder
        self._lsmultihead_att_wrapper: T.Iterable[GraphMultiHeadAttentionWrapped] = nn.ModuleList(  # type: ignore # noqa:
            [
                GraphMultiHeadAttentionWrapped(
                    embed_dim=embedding_dim,
                    num_heads=num_heads,
                    edge_dropout_p=edge_dropout_p,
                    rezero_or_residual=rezero_or_residual,
                )
                for _ in range(num_layers)
            ]
        )
        self._lsfeed_forward_wrapper: T.Iterable[FeedForwardWrapped] = nn.ModuleList(  # type: ignore # noqa:
            [
                FeedForwardWrapped(
                    in_out_dim=embedding_dim,
                    intermediate_dim=intermediate_dim,
                    rezero_or_residual=rezero_or_residual,
                    feat_dropout_p=feat_dropout_p,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, node_ids: torch.LongTensor, adj: Tensor, edge_types: torch.LongTensor
    ) -> Tensor:
        node_features = self._lsnode_feature_embedder[0](node_ids)
        for embedder in self._lsnode_feature_embedder[1:]:
            node_features = node_features + embedder(node_ids)

        if self._key_edge_feature_embedder:
            key_edge_features = self._key_edge_feature_embedder(edge_types)

        for multihead_att_wrapped, feed_forward_wrapped in zip(
            self._lsmultihead_att_wrapper, self._lsfeed_forward_wrapper
        ):
            node_features = multihead_att_wrapped(
                node_features=node_features,
                adj=adj,
                key_edge_features=key_edge_features,
            )

            node_features = feed_forward_wrapped(node_features=node_features)

        return node_features

    def __call__(
        self, word_ids: torch.LongTensor, adj: Tensor, edge_types: torch.LongTensor
    ) -> Tensor:
        return super.__call__(  # type: ignore
            word_ids=word_ids, adj=adj, edge_types=edge_types
        )


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
