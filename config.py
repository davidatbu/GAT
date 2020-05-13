from typing import Any


class Config:
    # Dirty, dirty trick of having it as a class variable,
    # then an instance variable later.
    _past_init_config = False

    def __setattr__(self, attr_name: str, attr: Any) -> None:
        if self._past_init_config and not hasattr(self, attr_name):
            raise Exception(f"{attr_name} is not an attribute of this config.")
        super().__setattr__(attr_name, attr)


class GATConfig(Config):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        nmid_layers: int,
        nhid: int,
        nheads: int,
        final_conat: bool = True,
        batch_norm: bool = True,
        edge_dropout_p: float = 0.0,
        feat_dropout_p: float = 0.1,
        alpha: float = 0.2,
        do_residual: bool = True,
        do_layer_norm: bool = True,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.nmid_layers = nmid_layers
        self.nhid = nhid
        self.nheads = nheads
        self.final_conat = final_conat
        self.batch_norm = batch_norm
        self.edge_dropout_p = edge_dropout_p
        self.feat_dropout_p = feat_dropout_p
        self.alpha = alpha
        self.do_residual = do_residual
        self.do_layer_norm = do_layer_norm

        self._past_init_config = True
