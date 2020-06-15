import typing as T

import typing_extensions as TT


class Config:

    _attr_names: T.List[str] = []

    @classmethod
    def from_dict(cls, d: T.Dict[str, T.Any]) -> T.Tuple["Config", T.Dict[str, T.Any]]:

        d = d.copy()
        our_kwargs: T.Dict[str, T.Any] = {}
        for _attr_names in list(
            d.keys()
        ):  # Hopefully list() copies, cuz otherwise we'll be modifying dictwhile loping thorugh
            if _attr_names in cls._attr_names:
                our_kwargs[_attr_names] = d.pop(_attr_names)

        return cls(**our_kwargs), d  # type: ignore

    def __setattr__(
        self, n: str, v: T.Optional[T.Union["Config", int, float, str]]
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

    def as_dict(self) -> T.Dict[str, T.Any]:
        d: T.Dict[str, T.Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                d[k] = v.as_dict()
            else:
                d[k] = v
        return d

    def validate(self) -> bool:
        return True


class TrainerConfig(Config):

    _attr_names: T.List[str] = [
        "lr",
        "epochs",
        "train_batch_size",
        "eval_batch_size",
        "do_eval_every_epoch",
        "do_eval_on_truncated_train_set",
        "use_cuda",
        "dataset_dir",
        "sent2graph_name",
    ] + Config._attr_names

    def __init__(
        self,
        lr: float,
        epochs: int,
        dataset_dir: str,
        sent2graph_name: TT.Literal["dep", "srl"],
        train_batch_size: int,
        eval_batch_size: int,
        use_cuda: bool = True,
        do_eval_every_epoch: bool = True,
        do_eval_on_truncated_train_set: bool = True,
    ):
        """Look as superclass doc."""
        self.lr = lr
        self.use_cuda = use_cuda
        self.epochs = epochs
        self.sent2graph_name = sent2graph_name
        self.do_eval_every_epoch = do_eval_every_epoch
        self.eval_batch_size = eval_batch_size
        self.train_batch_size = train_batch_size
        self.dataset_dir = dataset_dir
        self.do_eval_on_truncated_train_set = do_eval_on_truncated_train_set


class GATConfig(Config):

    _attr_names: T.List[str] = [
        "embedder_type",
        "vocab_size",
        "embed_dim",
        "embedder_config",
        "intermediate_dim",
        "cls_id",
        "nmid_layers",
        "nhid",
        "num_heads",
        "final_conat",
        "batch_norm",
        "edge_dropout_p",
        "feat_dropout_p",
        "alpha",
        "undirected",
        "nedge_type",
        "include_edge_features",
        "rezero_or_residual",
    ] + Config._attr_names

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        embedder_type: TT.Literal["bert", "simple"],
        intermediate_dim: int,
        cls_id: int,
        nmid_layers: int,
        nhid: int,
        num_heads: int,
        nedge_type: int,
        final_conat: bool = True,
        batch_norm: bool = True,
        edge_dropout_p: float = 0.0,
        feat_dropout_p: float = 0.3,
        alpha: float = 0.2,
        undirected: bool = True,
        include_edge_features: bool = True,
        rezero_or_residual: TT.Literal["rezero", "residual"] = "rezero",
    ):
        """Look at superclass doc."""
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.embedder_type = embedder_type

        self.intermediate_dim = intermediate_dim
        self.cls_id = cls_id
        self.nmid_layers = nmid_layers
        self.nhid = nhid
        self.num_heads = num_heads
        self.nedge_type = nedge_type
        self.final_conat = final_conat
        self.batch_norm = batch_norm
        self.edge_dropout_p = edge_dropout_p
        self.feat_dropout_p = feat_dropout_p
        self.alpha = alpha
        self.undirected = undirected
        self.include_edge_features = include_edge_features
        self.rezero_or_residual = rezero_or_residual

    def validate(self) -> bool:
        return True


class GATForSeqClsfConfig(GATConfig):

    _attr_names: T.List[str] = GATConfig._attr_names + ["nclass"]

    def __init__(self, nclass: int, **kwargs: T.Any):
        super().__init__(**kwargs)
        self.nclass = nclass


_T = T.TypeVar("_T")


class EverythingConfig(Config, T.Generic[_T]):
    _attr_names: T.List[str] = Config._attr_names + ["trainer", "model"]

    def __init__(self, trainer: TrainerConfig, model: _T):
        self.trainer = trainer
        self.model = model
