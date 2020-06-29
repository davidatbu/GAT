from __future__ import annotations

import abc
import itertools
import typing as T

import spacy  # type: ignore
import torch
from bpemb import BPEmb  # type: ignore
from torch import nn
from transformers import AutoTokenizer
from transformers import BertTokenizer

from Gat.data import vocabs
from Gat.neural.layers import Embedder


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def _tokenize(self, txt: str) -> T.List[str]:
        pass

    @staticmethod
    def split_on_special_toks(txt: str, lsspecial_tok: T.List[str]) -> T.List[str]:
        """Used to avoid splitting in the middle of special tokens.

        >>> Tokenizer.split_on_special_toks(
            ...     txt="[cls]Who's a good doggy?[pad]",
            ...     lsspecial_tok=["[cls]", "[pad]"]
            ... )
        [ "[cls]", "Whos' a good doggy?", "[pad]" ]
        """
        if lsspecial_tok == []:
            return [txt]

        lspart: T.List[str] = []
        tok = lsspecial_tok[0]
        while True:
            try:
                idx = txt.index(tok)
                part_before, part_after = txt[:idx], txt[idx + len(tok) :]
                if part_before:
                    lspart.append(part_before)
                lspart.append(tok)
                txt = part_after

            except ValueError:
                break

        if txt:
            lspart.append(txt)

        # Recurse with the other special tokens
        if len(lsspecial_tok) > 1:
            new_lspart = []
            for part in lspart:
                new_lspart.extend(
                    Tokenizer.split_on_special_toks(part, lsspecial_tok[1:])
                )
        else:
            new_lspart = lspart
        return new_lspart

    def tokenize(
        self, txt: str, lsspecial_tok: T.Optional[T.List[str]] = None
    ) -> T.List[str]:
        """Tokenize, making sure never to "cut across" special tokens."""
        if lsspecial_tok is None:
            return self._tokenize(txt)
        res = []
        for part in self.split_on_special_toks(txt, lsspecial_tok):
            if part in lsspecial_tok:
                res.append(part)
            else:
                res.extend(self._tokenize(part))
        return res

    def batch_tokenize(
        self, lstxt: T.List[str], max_len: T.Optional[int] = None
    ) -> T.List[T.List[str]]:
        """Batch version."""
        return [self.tokenize(txt) for txt in lstxt]

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass


class WrappedSpacyTokenizer(Tokenizer):
    """Wrapper around `nlp=spacy.load()` and `nlp(txt)`."""

    def __init__(self, spacy_model_name: str = "en_core_web_sm") -> None:
        """Loads spacy model."""
        # We're doing only the base model for now
        self._spacy_model_name = spacy_model_name
        self._tokenizer = spacy.load(
            self._spacy_model_name, disable=["tagger", "parser", "ner"]
        )

    def _tokenize(self, txt: str) -> T.List[str]:
        spacy_toks = self._tokenizer(txt)
        return [spacy_tok.text for spacy_tok in spacy_toks]

    def __repr__(self) -> str:
        return f"WrappedSpacyTokenizer_{self._spacy_model_name}"


class WrappedBertTokenizer(Tokenizer):
    """Wrap around BERT's tokenizer, also provide access to the "unwrapped tokenizer.

    We need the unwrapped because we want to do prepare some input to run thorugh a
    `transformers.modeling_bert.BertModel`.
    """

    def __init__(self) -> None:
        """Initialize BERT tokenizer."""
        # We're doing only the base model for now
        self._bert_model_name: T.Literal["bert-base-uncased"] = "bert-base-uncased"
        self._unwrapped_tokenizer = AutoTokenizer.from_pretrained(
            self._bert_model_name,
            do_lower_case=False,  # We handle lower casing ourselves, for consistency
        )

    def _tokenize(self, txt: str) -> T.List[str]:
        return self._unwrapped_tokenizer.tokenize(txt)  # type: ignore

    @property
    def unwrapped_tokenizer(self) -> BertTokenizer:
        return self._unwrapped_tokenizer

    @property
    def bert_model_name(self) -> str:
        """Which BERT model we are using.

        Used to ensure that the right tokenizer was used to prepare inputs to pass
        through a `Gat.layers.BertEmbedder`.
        """
        return self._bert_model_name

    def __repr__(self) -> str:
        return f"WrappedBertTokenizer-{self._bert_model_name}"


class WrappedBPETokenizer(Tokenizer):
    def __init__(self, bpemb_en: BPEmb) -> None:
        """. """
        assert bpemb_en.lang == "en"
        self._pbemb_en = bpemb_en
        self._vocab_size = bpemb_en.vocab_size

    def _tokenize(self, txt: str) -> T.List[str]:
        return self._pbemb_en.encode(txt)  # type: ignore

    def __repr__(self) -> str:
        return f"{self.__class__}-vocab_size_{str(self._vocab_size)}"


if T.TYPE_CHECKING:
    nnModule = nn.Module[torch.Tensor]
else:
    nnModule = nn.Module


class TokenizingReconciler(nnModule):
    def __init__(
        self,
        sub_word_vocab: vocabs.Vocab,
        word_tokenizer: Tokenizer,
        sub_word_embedder: Embedder,
    ) -> None:
        """Pool over subword embeddings.

        Args:
            sub_word_vocab: We access `sub_word_vocab.tokenizer` and
                `sub_word_vocab.padding_tok_id`.
            word_tokenizer:
            sub_word_embedder: We use it to get subword embeddings, and access
                `sub_.wrod_embedder.max_seq_len`.
        """
        super().__init__()
        self._sub_word_vocab = sub_word_vocab
        self._word_tokenizer = word_tokenizer
        self._sub_word_embedder = sub_word_embedder

    def forward(self, lstxt: T.List[str]) -> torch.Tensor:
        """Tokenize using the two tokenizers, pool over subwords to create word embedding.

        Args:
            lstxt: A list of sentences.
        Returns:
            embedding: (B, L, E)
                       B = len(lstxt)
                       L is computed like this:
                       The sentences are tokenized by self.word_vocab.tokenizer, and
                       truncated to the last word whose complete sub word tokenization
                       "fits inside" the maximum number of sub word tokens allowed by
                       `sub_word_embedder` per sequence.
                       L is the number of words in the sentence with the most word
                       tokens after the truncation described above.

                       For example,

                       sent = "love embeddings"
                       sub_word_tokenization = [ "love", "embed", "#dings" ]
                       word_tokenization = [ "love", "embeddings" ]
                       sub_word_embedder.max_seq_len == 2 # True
                       L == 1 # True, since "embeddings" doesn't fit within 2 sub word
                              # tokens
        """
        lswords = self._word_tokenizer.batch_tokenize(lstxt)
        lssubwordids_per_word = [
            [self._sub_word_vocab.tokenize_and_get_lstok_id(word) for word in words]
            for words in lswords
        ]
        # "Flat" sub word tokenization for each sequence
        lslssubwordid: T.List[T.List[int]] = []
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
            lslssubwordid.append(subwordids)

        subword_ids = torch.tensor(
            lslssubwordid, dtype=torch.long, device=next(self.parameters()).device
        )
        subword_embs = self._sub_word_embedder(subword_ids)

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
        padding_vec = self._sub_word_embedder(
            torch.tensor(
                [[self._sub_word_vocab.get_tok_id(self._sub_word_vocab.padding_tok)]]
            )
        )
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
