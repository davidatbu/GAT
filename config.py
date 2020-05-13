from typing import Dict
from typing import Union


class GATConfig(Dict[str, Union[int, bool, float]]):
    def __init__(
        self,
        vocab_size: int,
        in_features: int,
        nmid_layers: int,
        nhid: int,
        nheads: int,
        final_conat: bool = True,
        batch_norm: bool = True,
        edge_dropout_p: float = 0.1,
        feat_dropout_p: float = 0.1,
        alpha: float = 0.2,
    ):

        super().__init__()
        self["vocab_size"] = vocab_size
        self["in_features"] = in_features
        self["nmid_layers"] = nmid_layers
        self["nhid"] = nhid
        self["nheads"] = nheads
        self["final_conat"] = final_conat
        self["batch_norm"] = batch_norm
        self["edge_dropout_p"] = edge_dropout_p
        self["feat_dropout_p"] = feat_dropout_p
        self["alpha"] = alpha

    def __setitem__(self, key: str, value: Union[int, bool, float]) -> None:
        if key not in self:
            raise Exception("Not allowed to set arbitrary keys in this config.")
        super().__setitem__(key, value)
