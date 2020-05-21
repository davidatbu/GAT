from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union


class Config:
    # Dirty, dirty trick of having it as a class variable,
    # then an instance variable later.
    _past_init_config = False

    attr_names: List[str] = []

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any], pop_used: bool = True
    ) -> Tuple[Union["TrainConfig", "GATForSeqClsfConfig"], Dict[str, Any]]:

        d = d.copy()
        our_kwargs: Dict[str, Any] = {}
        for attr_name in list(
            d.keys()
        ):  # Hopefully list() copies, cuz otherwise we'll be modifying dictwhile loping thorugh
            if attr_name in cls.attr_names:
                our_kwargs[attr_name] = d.pop(attr_name)

        return cls(**our_kwargs), d  # type: ignore


class TrainConfig(Config):

    attr_names: List[str] = [
        "lr",
        "epochs",
        "train_batch_size",
        "eval_batch_size",
        "do_eval_every_epoch",
        "collate_fn",
        "use_cuda",
    ]

    def __init__(
        self,
        lr: float,
        epochs: int,
        train_batch_size: int,
        eval_batch_size: int,
        collate_fn: Callable[[Any], Any],
        use_cuda: bool = True,
        do_eval_every_epoch: bool = True,
    ):
        self.lr = lr
        self.use_cuda = use_cuda
        self.epochs = epochs
        self.do_eval_every_epoch = do_eval_every_epoch
        self.eval_batch_size = eval_batch_size
        self.train_batch_size = train_batch_size
        self.collate_fn = collate_fn


class GATConfig(Config):

    attr_names: List[str] = [
        "vocab_size",
        "embedding_dim",
        "cls_id",
        "nmid_layers",
        "nhid",
        "nheads",
        "final_conat",
        "batch_norm",
        "edge_dropout_p",
        "feat_dropout_p",
        "alpha",
        "do_residual",
        "undirected",
        "do_layer_norm",
        "nedge_type",
    ]

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        cls_id: int,
        nmid_layers: int,
        nhid: int,
        nheads: int,
        nedge_type: int,
        final_conat: bool = True,
        batch_norm: bool = True,
        edge_dropout_p: float = 0.0,
        feat_dropout_p: float = 0.3,
        alpha: float = 0.2,
        do_residual: bool = True,
        do_layer_norm: bool = True,
        undirected: bool = True,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.cls_id = cls_id
        self.nmid_layers = nmid_layers
        self.nhid = nhid
        self.nheads = nheads
        self.nedge_type = nedge_type
        self.final_conat = final_conat
        self.batch_norm = batch_norm
        self.edge_dropout_p = edge_dropout_p
        self.feat_dropout_p = feat_dropout_p
        self.alpha = alpha
        self.do_residual = do_residual
        self.do_layer_norm = do_layer_norm
        self.undirected = undirected

        self._past_init_config = True


class GATForSeqClsfConfig(GATConfig):

    attr_names: List[str] = GATConfig.attr_names + ["nclass"]

    def __init__(self, nclass: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.nclass = nclass
