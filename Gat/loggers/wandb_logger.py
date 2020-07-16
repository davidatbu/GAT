# type: ignore
"""
Weights and Biases
------------------
"""
import os
import typing as T
from argparse import Namespace
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch.nn as nn
import wandb
from pytorch_lightning.loggers.base import LightningLoggerBase  # type: ignore
from pytorch_lightning.utilities import rank_zero_only  # type: ignore
from wandb.wandb_run import Run  # type: ignore


class WandbLogger(LightningLoggerBase):
    """
    Log using `Weights and Biases <https://www.wandb.com/>`_. Install it with pip:

    .. code-block:: bash

        pip install wandb

    Args:
        name: Display name for the run.
        save_dir: Path where data is saved.
        offline: Run offline (data can be streamed later to wandb servers).
        id: Sets the version, mainly used to resume a previous run.
        anonymous: Enables or explicitly disables anonymous logging.
        version: Sets the version, mainly used to resume a previous run.
        project: The name of the project to which this run will belong.
        tags: Tags associated with this run.
        log_model: Save checkpoints in wandb dir to upload on W&B servers.
        experiment: WandB experiment object
        entity: The team posting this run (default: your username or your default team)
        group: A unique string shared by all runs in a given group

    Example:
        >>> from pytorch_lightning.loggers import WandbLogger
        >>> from pytorch_lightning import Trainer
        >>> wandb_logger = WandbLogger()
        >>> trainer = Trainer(logger=wandb_logger)

    See Also:
        - `Tutorial <https://app.wandb.ai/cayush/pytorchlightning/reports/
          Use-Pytorch-Lightning-with-Weights-%26-Biases--Vmlldzo2NjQ1Mw>`__
          on how to use W&B with Pytorch Lightning.

    """

    def __init__(
        self,
        name: Optional[str] = None,
        config: T.Optional[T.Dict[str.T.Any]] = None,
        save_dir: Optional[str] = None,
        offline: bool = False,
        id: Optional[str] = None,
        anonymous: bool = False,
        version: Optional[str] = None,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        log_model: bool = False,
        experiment: Run = None,
        entity=None,
        group: Optional[str] = None,
        sync_tensorboard: bool = False,
    ):
        super().__init__()
        self._name = name
        self._config = config
        self._save_dir = save_dir
        self._anonymous = "allow" if anonymous else None
        self._id = version or id
        self._tags = tags
        self._project = project
        self._experiment = experiment
        self._offline = offline
        self._entity = entity
        self._log_model = log_model
        self._group = group
        self._sync_tensorboard = sync_tensorboard

    def __getstate__(self) -> T.Dict[str, T.Any]:
        state = self.__dict__.copy()
        # args needed to reload correct experiment
        state["_id"] = self._experiment.id if self._experiment is not None else None

        # cannot be pickled
        state["_experiment"] = None
        return state

    @property
    def experiment(self) -> Run:
        r"""

        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_wandb_function()

        """
        assert wandb is not None
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"
            self._experiment = wandb.init(
                name=self._name,
                dir=self._save_dir,
                project=self._project,
                anonymous=self._anonymous,
                reinit=True,
                config=self._config,
                id=self._id,
                resume="allow",
                tags=self._tags,
                entity=self._entity,
                group=self._group,
                sync_tensorboard=self._sync_tensorboard,
            )
            # save checkpoints in wandb dir to upload on W&B servers
            if self._log_model:
                self.save_dir = self._experiment.dir
        return self._experiment

    def watch(
        self, model: nn.Module, log: str = "gradients", log_freq: int = 100
    ) -> None:
        self.experiment.watch(model, log=log, log_freq=log_freq)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        self.experiment.log(
            {"global_step": step, **metrics} if step is not None else metrics
        )

    @property
    def name(self) -> str:
        # don't create an experiment if we don't have one
        name = self._experiment.project_name() if self._experiment else None
        return name

    @property
    def version(self) -> str:
        # don't create an experiment if we don't have one
        return self._experiment.id if self._experiment else None
