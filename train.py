import logging
import random
import typing as T
from pathlib import Path
from pprint import pformat

import numpy as np
import sklearn.metrics as skmetrics
import torch
import torch.nn as nn
import torch.optim as optim
import wandb  # type: ignore
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm  # type: ignore

from Gat import data
from Gat.config import base as config
from Gat.neural import models
from Gat.utils import base as utils


logger = logging.getLogger("__main__")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)  # type: ignore

_T = T.TypeVar("_T")


def _prefix_keys(d: T.Dict[str, _T], prefix: str) -> T.Dict[str, _T]:
    return {f"{prefix}_{k}": v for k, v in d.items()}


_ExampleType = T.TypeVar("_ExampleType")


class Trainer(T.Generic[_ExampleType]):
    def __init__(self, train_config: config.TrainerConfig) -> None:
        self.config = train_config
        self._prepare_data()

    def _prepare_data(self) -> None:
        self._val_name = "dev"
        datasets, txt_srcs, vocab = data.load_splits(
            Path(self.config.dataset_dir),
            sent2graph_name=self.config.sent2graph_name,
            lstxt_col=["sentence"],
            splits=["train", self._val_name],
        )

        self._datasets = datasets
        self._txt_srcs = txt_srcs
        self._vocab = vocab

    @property
    def train_dataset(self) -> Dataset[_ExampleType]:
        return self._datasets["train"]

    @property
    def val_dataset(self) -> _Dataset:
        return self._datasets[self._val_name]

    @property
    def vocab(self) -> data.Vocab:
        return self._vocab

    def train(self, model: models.GATForSeqClsf,) -> None:
        # Model and optimizer
        if self.config.use_cuda:
            model.cuda()

        train_dataset, val_dataset = (
            self._datasets["train"],
            self._datasets[self._val_name],
        )

        # wandb.watch(model)  # Watch the gradients
        # Get the computational graph
        with SummaryWriter(log_dir=wandb.run.dir) as tb_writer:
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.train_batch_size,
                collate_fn=data.SentenceGraphDataset.collate_fn,
            )
            X, y = next(iter(loader))
            prepared_X = model.prepare_batch(X)
            tb_writer.add_graph(model, (prepared_X,))

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            collate_fn=data.SentenceGraphDataset.collate_fn,
        )
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)

        examples_seen = 0
        batches_seen = 0

        train_dataset_slice = data.SliceDataset(train_dataset, n=len(val_dataset))
        val_metrics, _, _ = self.evaluate(model, val_dataset)
        val_metrics = _prefix_keys(val_metrics, "val")
        logger.info(f"Before training | val: {val_metrics}")
        running_loss = torch.tensor(
            0, dtype=torch.float, device=next(model.parameters()).device
        )
        for epoch in range(1, self.config.epochs + 1):
            pbar_desc = f"epoch: {epoch}"
            pbar = tqdm(train_loader, desc=pbar_desc)
            for X, y in pbar:
                prepared_X = model.prepare_batch(X)
                running_loss += train_one_step(model, optimizer, prepared_X, y)
                examples_seen += self.config.train_batch_size
                batches_seen += 1
                pbar.set_description(
                    f"{pbar_desc} | train loss running: {(running_loss / batches_seen).item()}"
                )

            if self.config.do_eval_every_epoch:
                val_metrics, val_logits, val_true = self.evaluate(model, val_dataset)
                train_metrics, _, _ = self.evaluate(model, train_dataset_slice)

                all_metrics = dict(
                    **_prefix_keys(val_metrics, "val"),
                    **_prefix_keys(train_metrics, "train"),
                )
                wandb.log(all_metrics, step=examples_seen)
                logger.info(pformat(all_metrics))

        if not self.config.do_eval_every_epoch:
            val_metrics, val_logits, val_true = self.evaluate(model, val_dataset)
            train_metrics, _, _ = self.evaluate(model, train_dataset_slice)

            all_metrics = dict(
                **_prefix_keys(val_metrics, "val"),
                **_prefix_keys(train_metrics, "train"),
            )

        # Computed in either the for loop or after the for loop
        wandb.log(all_metrics)
        logger.info(pformat(all_metrics))

        self.analyze_predict(
            logits=val_logits,
            true_=val_true,
            ds=val_dataset,
            txt_src=self._txt_srcs[self._val_name],
        )

    def evaluate(
        self, model: models.GATForSeqClsf, dataset: Dataset  # type: ignore
    ) -> T.Tuple[T.Dict[str, float], np.ndarray, np.ndarray]:

        val_loader = DataLoader(
            dataset,
            batch_size=self.config.eval_batch_size,
            collate_fn=data.SentenceGraphDataset.collate_fn,
        )

        model.eval()
        with torch.no_grad():
            loss = torch.tensor(
                0, dtype=torch.float, device=next(model.parameters()).device
            )
            all_logits: T.Optional[np.ndarray] = None
            all_true: T.Optional[np.ndarray] = None
            for X, y in val_loader:
                prepared_X = model.prepare_batch(X)
                logits, one_step_loss = model(prepared_X=prepared_X, y=y)
                loss += one_step_loss

                if all_logits is None:
                    all_logits = logits.detach().cpu().numpy()
                    all_true = np.array(y)
                else:
                    all_logits = np.concatenate(
                        [all_logits, logits.detach().cpu().numpy()], axis=0
                    )
                    all_true = np.concatenate([all_true, np.array(y)], axis=0)

        all_preds = np.argmax(all_logits, axis=1)
        acc: float = skmetrics.accuracy_score(all_true, all_preds)
        loss /= len(val_loader)
        metrics = {
            "loss": float(loss.item()),
            "acc": acc,
        }

        assert all_logits is not None
        assert all_true is not None
        return (metrics, all_logits, all_true)

    def analyze_predict(
        self,
        logits: np.ndarray,
        true_: np.ndarray,
        txt_src: data.TextSource,
        ds: data.SentenceGraphDataset,
    ) -> None:
        preds: np.ndarray = logits.argmax(axis=1)
        matches: np.ndarray = np.equal(preds, true_)
        table_rows: T.List[T.Tuple[utils.Cell, ...]] = [
            (
                utils.TextCell(txt_src[i].lssent[0]),
                utils.SvgCell(ds.sentgraph_to_svg(ds[i].lssentgraph[0])),
                utils.NumCell(preds[i]),
                utils.NumCell(ds[i].lbl_id),
            )
            for i in range(preds.shape[0])
        ]
        row_colors = [None if i else "red" for i in matches]
        table_html = utils.html_table(
            rows=table_rows,
            headers=tuple(
                utils.TextCell(i)
                for i in ["Original", "Tokenized Parse", "Predicted", "Gold"]
            ),
            row_colors=row_colors,
        )

        cm = confusion_matrix(true_, preds, labels=range(len(self.vocab.id2lbl)))
        cm_plot = utils.plotly_cm(cm, labels=self.vocab.id2lbl)
        wandb.log(
            {
                "val_preds": wandb.Html(table_html, inject=False),
                "confusion_matrix": cm_plot,
            }
        )


def train_one_step(
    model: nn.Module, optimizer: optim.Optimizer, prepared_X: T.Any, y: T.Any  # type: ignore
) -> Tensor:
    model.train()  # Turn on the train mode
    optimizer.zero_grad()
    logits, loss = model(prepared_X=prepared_X, y=y)
    loss.backward()
    # Clipping here maybe?
    optimizer.step()
    return loss  # type: ignore


def main() -> None:
    trainer_config = config.TrainerConfig(
        lr=1e-3,
        train_batch_size=128,
        eval_batch_size=128,
        epochs=10,
        # dataset_dir="actual_data/glue_data/SST-2",
        sent2graph_name="dep",
        # dataset_dir="actual_data/paraphrase/paws_small",
        dataset_dir="actual_data/SST-2_small",
    )

    trainer = Trainer(trainer_config)

    model_config = config.GATForSeqClsfConfig(
        vocab_size=300,
        cls_id=trainer.vocab.cls_tok_id,
        num_heads=6,
        embedding_dim=300,
        intermediate_dim=300,
        feat_dropout_p=0.3,
        nclass=len(trainer.vocab._id2lbl),
        nmid_layers=10,
        nedge_type=len(trainer.val_dataset.sent2graph.id2edge_type),
    )

    all_config = config.EverythingConfig(trainer=trainer_config, model=model_config)
    logger.info("About to try: " + pformat(all_config))

    model = models.GATForSeqClsf(all_config.model, emb_init=None)
    wandb.init(
        project="gat",
        config=all_config.as_dict(),
        dir="./wandb_runs",
        sync_tensorboard=True,
    )

    trainer.train(model)


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    main()
