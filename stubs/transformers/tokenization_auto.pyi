import typing as T

from .tokenization_bert import BertTokenizer

class AutoTokenizer:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: T.Literal["bert-base-uncased"],
        *inputs: T.Any,
        **kwargs: T.Any,
    ) -> BertTokenizer: ...
