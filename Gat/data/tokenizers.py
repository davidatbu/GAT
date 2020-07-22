from __future__ import annotations

import abc
import itertools
import typing as T
from pathlib import Path

import lazy_import
import spacy  # type: ignore
import torch
from torch import nn

if T.TYPE_CHECKING:
    from transformers import AutoTokenizer
    from transformers import BertTokenizer
    import youtokentome as yttm
else:
    AutoTokenizer = lazy_import.lazy_class("transformers.AutoTokenizer")
    BertTokenizer = lazy_import.lazy_class("transformers.BertTokenizer")
    yttm = lazy_import.lazy_module("youtokentome")


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
        self._unwrapped_tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(  # type: ignore[assignment]
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
    def __init__(self, bpe: yttm.BPE, bpe_settings_str: str) -> None:
        """. """
        self._bpe = bpe
        self._bpe_settings_str = bpe_settings_str

    def _tokenize(self, txt: str) -> T.List[str]:
        return self._bpe.encode(
            [txt],
            output_type=yttm.OutputType.SUBWORD,
            bos=False,
            eos=False,
            reverse=False,
            dropout_prob=0.0,
        )[0]

    def __repr__(self) -> str:
        return f"{self.__class__}-model_file_{str(self._bpe_settings_str)}"
