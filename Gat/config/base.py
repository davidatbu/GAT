from typing import Any
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

from typing_extensions import Literal


class Config:

    _attr_names: List[str] = []

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Tuple["Config", Dict[str, Any]]:

        d = d.copy()
        our_kwargs: Dict[str, Any] = {}
        for _attr_names in list(
            d.keys()
        ):  # Hopefully list() copies, cuz otherwise we'll be modifying dictwhile loping thorugh
            if _attr_names in cls._attr_names:
                our_kwargs[_attr_names] = d.pop(_attr_names)

        return cls(**our_kwargs), d  # type: ignore

    def __setattr__(
        self, n: str, v: Optional[Union["Config", int, float, str]]
    ) -> None:
        if n not in self._attr_names:
            raise Exception(
                f"{n} is not a valid confiugration for {self.__class__.__name__}"
            )

        super().__setattr__(n, v)
        if not self.validate():
            raise Exception(f"{str(self)} is invalid.")

    def __str__(self) -> str:
        return str(self.as_dict())

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                d[k] = v.as_dict()
            else:
                d[k] = v
        return d

    def validate(self) -> bool:
        return True


class TrainConfig(Config):

    _attr_names: List[str] = [
        "lr",
        "epochs",
        "train_batch_size",
        "eval_batch_size",
        "do_eval_every_epoch",
        "use_cuda",
        "dataset_dir",
        "sent2graph_name",
    ]

    def __init__(
        self,
        lr: float,
        epochs: int,
        dataset_dir: str,
        sent2graph_name: Literal["dep", "srl"],
        train_batch_size: int,
        eval_batch_size: int,
        use_cuda: bool = True,
        do_eval_every_epoch: bool = True,
    ):
        self.lr = lr
        self.use_cuda = use_cuda
        self.epochs = epochs
        self.sent2graph_name = sent2graph_name
        self.do_eval_every_epoch = do_eval_every_epoch
        self.eval_batch_size = eval_batch_size
        self.train_batch_size = train_batch_size
        self.dataset_dir = dataset_dir


class GATConfig(Config):

    _attr_names: List[str] = [
        "vocab_size",
        "embedding_dim",
        "intermediate_dim",
        "cls_id",
        "nmid_layers",
        "nhid",
        "nheads",
        "final_conat",
        "batch_norm",
        "edge_dropout_p",
        "feat_dropout_p",
        "alpha",
        "undirected",
        "do_layer_norm",
        "nedge_type",
        "do_rezero",
    ]

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        intermediate_dim: int,
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
        do_layer_norm: bool = True,
        undirected: bool = True,
        do_rezero: bool = True,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim
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
        self.do_layer_norm = do_layer_norm
        self.undirected = undirected
        self.do_rezero = do_rezero

    def validate(self) -> bool:
        return True


class GATForSeqClsfConfig(GATConfig):

    _attr_names: List[str] = GATConfig._attr_names + ["nclass"]

    def __init__(self, nclass: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.nclass = nclass


_T = TypeVar("_T")


class EverythingConfig(Config, Generic[_T]):
    _attr_names: List[str] = Config._attr_names + ["trainer", "model"]

    def __init__(self, trainer: TrainConfig, model: _T):
        self.trainer = trainer
        self.model = model
