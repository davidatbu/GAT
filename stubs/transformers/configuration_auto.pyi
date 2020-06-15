import typing as T

from .configuration_bert import BertConfig

class AutoConfig:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: T.Literal["bert-base-uncased"],
        **kwargs: T.Any,
    ) -> BertConfig: ...

__all__ = ["AutoConfig", "BertConfig"]
