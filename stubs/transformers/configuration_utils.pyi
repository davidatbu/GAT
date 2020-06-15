import typing as T

class PretrainedConfig:
    pretrained_config_archive_map: T.Dict[str, str]
    model_type: str = ""
