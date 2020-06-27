"""Wrapped BERT tokenizer."""
import typing as T

from transformers import AutoTokenizer
from transformers import BertTokenizer

from Gat.data.tokenizers.base import Tokenizer


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

    def tokenize(self, txt: str) -> T.List[str]:
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


__all__ = ["WrappedBertTokenizer"]
