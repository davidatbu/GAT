from __future__ import annotations

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
import wandb
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.trainer import seed_everything
from pytorch_lightning.trainer import Trainer
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore

from Gat import data
from Gat import utils
from Gat.config import base as config
from Gat.neural import models


logger = logging.getLogger("__main__")

seed_everything(0)


class OneBatch(T.NamedTuple):
    word_ids: torch.LongTensor
    batched_adj: torch.BoolTensor
    edge_types: torch.LongTensor
    target: torch.LongTensor


class LitGatForSequenceClassification(LightningModule):
    def __init__(
        self, all_config: config.EverythingConfig,
    ):
        super().__init__()
        self._all_config = all_config
        self.save_hyperparameters(all_config.as_flat_dict())

    def setup(self, stage: str) -> None:
        self._setup_data()
        self._setup_model()

    def _setup_data(self) -> None:
        preprop_config = self._all_config.preprop

        datasets, txt_srcs, word_vocab = data.load_splits(
            unk_thres=preprop_config.unk_thres,
            sent2graph_name=preprop_config.sent2graph_name,
            dataset_dir=Path(preprop_config.dataset_dir),
            lstxt_col=["sentence"],
            splits=["train", "val"],
        )

        for key, dataset in datasets.items():
            dataset = data.ConnectToClsDataset(dataset)
            if self._all_config.preprop.undirected:
                dataset = data.UndirectedDataset(dataset)
            datasets[key] = dataset

        self._datasets = datasets
        self._txt_srcs = txt_srcs
        self._word_vocab: data.BasicVocab = word_vocab

    def _setup_model(self) -> None:
        model_config = self._all_config.model

        # Set dataset dependent configuration
        model_config.dataset_dep = config.GATForSequenceClassificationDatasetDepConfig(
            num_classes=len(self._datasets["train"].vocab.labels.all_lbls),
            num_edge_types=len(self._datasets["train"].id2edge_type),
        )
        if model_config.node_embedding_type == "pooled_bert":
            sub_word_vocab: T.Optional[data.BertVocab] = data.BertVocab()
        else:
            sub_word_vocab = None

        self._gat_model = models.GATForSequenceClassification(
            model_config, word_vocab=self._word_vocab, sub_word_vocab=sub_word_vocab,
        )
        self._crs_entrpy = nn.CrossEntropyLoss()
        self._trainer_config = self._all_config.trainer

    def _collate_fn(self, lsgraph_example: T.List[utils.GraphExample]) -> OneBatch:
        """Turn `GraphExample` into a series of `torch.Tensor`s  """
        lslsgraph: T.List[T.List[utils.Graph]]
        lslbl_id: T.List[int]
        lslsgraph, lslbl_id = map(list, zip(*lsgraph_example))  # type: ignore

        # Since we're doing single sentence classification, we don't need additional
        # nesting
        lsgraph = [lsgraph[0] for lsgraph in lslsgraph]

        lslsedge: T.List[T.List[T.Tuple[int, int]]]
        lslsedge_type: T.List[T.List[int]]
        lslsimp_node: T.List[T.List[int]]
        lsnodeid2wordid: T.List[T.List[int]]

        lslsedge, lslsedge_type, lslsimp_node, lsnodeid2wordid = map(  # type: ignore
            list, zip(*lsgraph)
        )
        word_ids: torch.LongTensor = self._word_vocab.prepare_for_embedder(
            lsnodeid2wordid, self._gat_model.word_embedder
        )
        # Build the adjacnecy matrices
        batched_adj: torch.BoolTensor = torch.zeros(  # type: ignore
            word_ids.size("B"),
            word_ids.size("L"),
            word_ids.size("L"),
            names=("B", "L_left", "L_right"),
            dtype=torch.bool,
        )

        # Build the edge types
        edge_types: torch.LongTensor = torch.zeros(  # type: ignore
            word_ids.size("B"),
            word_ids.size("L"),
            word_ids.size("L"),
            # names=("B", "L_left", "L_right"),
            dtype=torch.long,
        )
        for batch_num, (lsedge, lsedge_type) in enumerate(zip(lslsedge, lslsedge_type)):
            indexing_arrs: T.Tuple[T.List[int], T.List[int]] = tuple(zip(*lsedge))  # type: ignore
            batched_adj.rename(None)[batch_num][indexing_arrs[0], indexing_arrs[1]] = 1
            edge_types.rename(None)[batch_num][
                indexing_arrs[0], indexing_arrs[1]
            ] = torch.tensor(lsedge_type, dtype=torch.long)

        target: torch.LongTensor = torch.tensor(  # type: ignore
            lslbl_id, names=("B",)
        )

        return OneBatch(
            word_ids=word_ids,
            batched_adj=batched_adj,
            edge_types=edge_types,
            target=target,
        )

    def train_dataloader(self) -> DataLoader[utils.GraphExample]:
        res = DataLoader(
            dataset=self._datasets["train"],
            collate_fn=self._collate_fn,
            batch_size=self._trainer_config.train_batch_size,
        )

        return res

    def val_dataloader(self) -> T.List[DataLoader[OneBatch]]:
        res = DataLoader(
            self._datasets["val"],
            collate_fn=self._collate_fn,
            batch_size=self._trainer_config.eval_batch_size,
        )

        self._val_dataset_names = ["val"]
        # iF we were to evaluate multiple datasets( for example, if we wanted
        # to get the loss on the training dataset itself,), we'd indicate that here
        return [res]

    def configure_optimizers(self) -> optim.optimizer.Optimizer:
        params = list(self.parameters())
        print(f"passing params of length: {len(params)}")
        return optim.Adam(params, lr=self._trainer_config.lr)

    def forward(  # type: ignore
        self,
        word_ids: torch.LongTensor,
        batched_adj: torch.BoolTensor,
        edge_types: torch.LongTensor,
    ) -> torch.Tensor:
        logits = self._gat_model(word_ids, batched_adj, edge_types)
        return logits

    def __call__(
        self,
        word_ids: torch.LongTensor,
        batched_adj: torch.BoolTensor,
        edge_types: torch.LongTensor,
    ) -> torch.Tensor:
        return super().__call__(word_ids, batched_adj, edge_types)  # type: ignore

    def training_step(  # type: ignore
        self, batch: OneBatch, batch_idx: int
    ) -> T.Dict[str, T.Union[Tensor, T.Dict[str, Tensor]]]:
        logits = self(batch.word_ids, batch.batched_adj, batch.edge_types)
        loss = self._crs_entrpy(logits.rename(None), batch.target.rename(None))

        return {
            "loss": loss,
        }

    def validation_step(  # type: ignore
        self, batch: OneBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> T.Dict[str, Tensor]:
        logits = self(batch.word_ids, batch.batched_adj, batch.edge_types)
        return {"logits": logits, "target": batch.target}

    def on_train_start(self) -> None:
        one_example = next(iter(self.train_dataloader()))
        if hasattr(self.logger.experiment, "add_graph"):
            self.logger.experiment.add_graph(self._gat_model, one_example)

    def validation_epoch_end(
        self,
        outputs: T.Union[
            T.List[T.Dict[str, Tensor]], T.List[T.List[T.Dict[str, Tensor]]]
        ],
    ) -> T.Dict[str, T.Dict[str, Tensor]]:
        res: T.Dict[str, Tensor] = {}

        lslsoutput: T.List[T.List[T.Dict[str, Tensor]]]
        if isinstance(outputs[0], dict):
            lslsoutput = [outputs]  # type: ignore
        else:
            lslsoutput = outputs  # type: ignore
        for i, lsoutput in enumerate(lslsoutput):
            val_dataset_name = self._val_dataset_names[i]
            all_logits = torch.cat([output["logits"] for output in lsoutput])
            all_target = torch.cat([output["target"] for output in lsoutput])

            all_preds = all_logits.rename(None).argmax(dim=1)
            acc = accuracy(all_preds, all_target)
            res.update({f"{val_dataset_name}_acc": acc})

        return {"progress_bar": res, "log": res}

    """
    def on_train_end(self) -> None:
        return
        self.analyze_predict(
            logits=val_logits,
            true_=val_true,
            ds=val_dataset,
            txt_src=self._txt_srcs[self._val_name],
        )

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
        self.logger.log(
            {
                "val_preds": wandb.Html(table_html, inject=False),
                "confusion_matrix": cm_plot,
            }
        )
        """


def main() -> None:
    all_config = config.EverythingConfig(
        trainer=config.TrainerConfig(
            lr=1e-3, train_batch_size=128, eval_batch_size=128, epochs=10,
        ),
        preprop=config.PreprocessingConfig(
            undirected=True,
            dataset_dir="actual_data/SST-2_tiny",
            # dataset_dir="actual_data/glue_data/SST-2",
            # dataset_dir="actual_data/paraphrase/paws_small",
            sent2graph_name="dep",
        ),
        model=config.GATForSequenceClassificationConfig(
            embedding_dim=768,
            gat_layered=config.GATLayeredConfig(
                num_heads=6, intermediate_dim=768, feat_dropout_p=0.3, num_layers=10,
            ),
            node_embedding_type="pooled_bert",
            use_edge_features=True,
            dataset_dep=None,
        ),
    )

    model = LitGatForSequenceClassification(all_config)

    logger = WandbLogger(project="gat")
    trainer = Trainer(logger=logger)

    trainer.fit(model)


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    main()
