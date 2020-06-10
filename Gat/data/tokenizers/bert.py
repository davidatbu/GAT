import typing as T

from transformers import tokenization_bert

from ..tokenizers import base


class WrappedBertTokenizer(base.Tokenizer):
    def __init__(self) -> None:
        # We're doing only the base model for now
        self._bert_model_name = "bert-base-uncased"
        self._tokenizer = tokenization_bert.BertTokenizer.from_pretrained(
            self._bert_model_name,
            do_lower_case=False,  # We handle lower casing ourselves, for consistency
        )

    def tokenize(self, txt: str) -> T.List[str]:
        return self._tokenizer.tokenize(txt)  # type: ignore

    def __repr__(self) -> str:
        return f"WrappedBertTokenizer{self._bert_model_name}"
