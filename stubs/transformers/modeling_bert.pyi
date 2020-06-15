import typing as T

import torch

from .configuration_auto import BertConfig
from .modeling_utils import PreTrainedModel

class BertPreTrainedModel(PreTrainedModel):
    config_class: BertConfig

class BertModel(BertPreTrainedModel):
    def forward(
        self,
        input_ids: T.Optional[torch.Tensor],
        token_type_ids: T.Optional[torch.Tensor],
        position_ids: T.Optional[torch.Tensor],
    ) -> torch.Tensor: ...
