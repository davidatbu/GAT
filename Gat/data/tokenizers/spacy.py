import typing as T

import spacy  # type: ignore

from ..tokenizers import base


class WrappedSpacyTokenizer(base.Tokenizer):
    def __init__(self) -> None:
        # We're doing only the base model for now
        self._spacy_model_name = "en_core_web_sm"
        self._tokenizer = spacy.load(
            self._spacy_model_name, disable=["tagger", "parser", "ner"]
        )

    def tokenize(self, txt: str) -> T.List[str]:
        spacy_toks = self._tokenizer(txt)
        return [spacy_tok.text for spacy_tok in spacy_toks]

    def __repr__(self) -> str:
        return f"WrappedSpacyTokenizer_{self._spacy_model_name}"
