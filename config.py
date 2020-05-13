from typing import Any
from typing import Dict
from typing import Union


class Config:
    def __init__(self, **kwargs: Dict[str, Union[int, float, bool]]):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, attr_name: str, attr: Any) -> None:
        if not hasattr(self, attr_name):
            raise Exception("Not allowed to set arbitrary attrs in this config.")
        super().__setattr__(attr_name, attr)


class GATConfig(Config):

    # Autocomplete's sake
    vocab_size: int
    in_features: int
    nmid_layers: int
    nhid: int
    nheads: int
    final_conat: bool = True
    batch_norm: bool = True
    edge_dropout_p: float = 0.1
    feat_dropout_p: float = 0.1
    alpha: float = 0.2
