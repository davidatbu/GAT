import typing as T

from torch import nn

if T.TYPE_CHECKING:
    nnModule = nn.Module[T.Any]
else:
    nnModule = nn.Module

class PreTrainedModel(nnModule):
    config_class: T.Any
    pretrained_model_archive_map: T.Dict[str, str]
    load_tf_weights: bool
    base_model_prefix: str

__all__ = ["PreTrainedModel"]
