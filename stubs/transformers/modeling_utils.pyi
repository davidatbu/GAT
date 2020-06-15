import typing as T

class PreTrainedModel:
    config_class: T.Any
    pretrained_model_archive_map: T.Dict[str, str]
    load_tf_weights: bool
    base_model_prefix: str

__all__ = ["PreTrainedModel"]
