from __future__ import annotations

import typing as T

from .configuration_auto import BertConfig
from .modeling_bert import BertModel

__all__ = ["AutoModel", "BertModel"]

class AutoModel:
    @classmethod
    def from_config(cls, config: BertConfig) -> BertModel: ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: T.Literal["bert-base-uncased"],
        *model_args: T.Any,
        **kwargs: T.Any
    ) -> BertModel: ...
