from __future__ import annotations

import logging
import typing as T
from pathlib import Path

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from pytorch_lightning.callbacks.early_stopping import EarlyStopping  # type: ignore
from pytorch_lightning.core.lightning import LightningModule  # type: ignore
from pytorch_lightning.loggers.base import LightningLoggerBase  # type: ignore
from pytorch_lightning.metrics.functional import accuracy  # type: ignore
from pytorch_lightning.trainer import seed_everything  # type: ignore
from pytorch_lightning.trainer import Trainer
from torch import Tensor
from torch.utils.data import DataLoader  # type: ignore

import Gat.data.utils
from Gat import configs
from Gat import data
from Gat import utils
from Gat.loggers.wandb_logger import WandbLogger  # type: ignore [attr-defined]
from Gat.neural import layers
from Gat.neural import models


# from pytorch_lightning.loggers import TensorBoardLogger


logger = logging.getLogger("__main__")

seed_everything(0)


class LitGatForSequenceClassification(LightningModule):
    def __init__(
        self, all_config: configs.EverythingConfig,
    ):
        super().__init__()
        self._all_config = all_config

        self.save_hyperparameters(self._all_config.as_flat_dict())

    def setup(self, stage: str) -> None:
        self._setup_data()
        self._setup_model()

    def _setup_data(self) -> None:
        preprop_config = self._all_config.preprop

        dataset_per_split, txt_src_per_split, word_vocab = data.utils.load_splits(
            unk_thres=preprop_config.unk_thres,
            sent2graph_name=preprop_config.sent2graph_name,
            dataset_dir=Path(preprop_config.dataset_dir),
            lstxt_col=["sentence"],
            splits=["train", "val"],
        )

        for key, dataset in dataset_per_split.items():
            dataset = data.datasets.ConnectToClsDataset(dataset)
            if self._all_config.preprop.undirected:
                dataset = data.datasets.UndirectedDataset(dataset)
            dataset_per_split[key] = dataset

        self._dataset_per_split = dataset_per_split
        self._txt_src_per_split = txt_src_per_split
        self._word_vocab: data.vocabs.BasicVocab = word_vocab

        # Set dataset dependent configuration
        self._all_config.model.dataset_dep = configs.GATForSequenceClassificationDatasetDepConfig(
            num_classes=len(self._dataset_per_split["train"].numerizer.labels.all_lbls),
            num_edge_types=len(self._dataset_per_split["train"].id2edge_type),
        )

    def _setup_model(self) -> None:
        model_config = self._all_config.model

        if model_config.node_embedding_type == "pooled_bert":
            sub_word_vocab: T.Optional[data.vocabs.Vocab] = data.vocabs.BertVocab()
        elif model_config.node_embedding_type == "bpe":
            assert model_config.bpe_vocab_size is not None
            sub_word_vocab = data.vocabs.BPEVocab(
                txt_src=self._txt_src_per_split["train"],
                bpe_vocab_size=model_config.bpe_vocab_size,
                load_pretrained_embs=False,
                lower_case=self._all_config.preprop.lower_case,
                cache_dir=Path(self._all_config.preprop.dataset_dir),
            )
        elif model_config.node_embedding_type == "basic":
            sub_word_vocab = None
        else:
            raise Exception(
                "model_config._validate() should have raised an excetpion, actually."
            )

        self._gat_model = models.GATForSequenceClassification(
            model_config, word_vocab=self._word_vocab, sub_word_vocab=sub_word_vocab,
        )
        self._crs_entrpy = nn.CrossEntropyLoss()
        self._trainer_config = self._all_config.trainer

    def _prepare_batch(
        self, lsgraph_example: T.List[utils.GraphExample]
    ) -> layers.PreparedBatch:
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

        lslsedge, lslsedge_type, lslsimp_node, lsnodeid2wordid = [
            list(tup_graph_attr) for tup_graph_attr in zip(*lsgraph)  # type: ignore
        ]

        B = len(lsgraph_example)
        # TODO: Clean up access of private attributes
        L = self._gat_model._word_embedder.max_seq_len
        if L is None:  # No maximum sequence by embedder
            L = max(map(len, lsnodeid2wordid))

        # Build the adjacnecy matrices
        batched_adj = torch.zeros([B, L, L], dtype=torch.bool)
        batched_adj.requires_grad_(False)

        # Build the edge types
        edge_types: torch.Tensor = torch.zeros(
            [B, L, L], dtype=torch.long,
        )
        edge_types.requires_grad_(False)
        edge_types.detach_()
        for batch_num, (lsedge, lsedge_type) in enumerate(zip(lslsedge, lslsedge_type)):
            indexing_arrs: T.Tuple[T.List[int], T.List[int]] = tuple(zip(*lsedge))  # type: ignore
            batched_adj[batch_num][indexing_arrs[0], indexing_arrs[1]] = 1
            edge_types[batch_num][indexing_arrs[0], indexing_arrs[1]] = torch.tensor(
                lsedge_type, dtype=torch.long
            )

        target = torch.tensor(lslbl_id, dtype=torch.long)
        # (B,)

        return layers.PreparedBatch(
            lslsnode_id=lsnodeid2wordid,
            batched_adj=batched_adj,
            edge_types=edge_types,
            target=target,
        )

    def train_dataloader(self) -> DataLoader[utils.GraphExample]:
        res = DataLoader(
            dataset=self._dataset_per_split["train"],
            collate_fn=self._prepare_batch,
            batch_size=self._trainer_config.train_batch_size,
            num_workers=8,
        )

        return res

    def val_dataloader(self) -> T.List[DataLoader[utils.GraphExample]]:
        val_dataloader = DataLoader(
            self._dataset_per_split["val"],
            collate_fn=self._prepare_batch,
            batch_size=self._trainer_config.eval_batch_size,
            num_workers=8,
        )

        cut_train_dataset = data.datasets.CutDataset(
            self._dataset_per_split["train"],
            total_len=len(self._dataset_per_split["val"]),
        )
        cut_train_dataloader = DataLoader(
            cut_train_dataset,
            collate_fn=self._prepare_batch,
            batch_size=self._trainer_config.eval_batch_size,
            num_workers=8,
        )

        self._val_dataset_names = ["val", "cut_train"]

        return [val_dataloader, cut_train_dataloader]

    def configure_optimizers(self) -> optim.optimizer.Optimizer:
        params = list(self.parameters())
        print(f"passing params of length: {len(params)}")
        return optim.Adam(params, lr=self._trainer_config.lr)

    def forward(  # type: ignore
        self, lsgraph_example: T.List[utils.GraphExample]
    ) -> torch.Tensor:
        logits = self._gat_model(lsgraph_example)
        return logits

    def __call__(self, prepared_batch: layers.PreparedBatch) -> torch.Tensor:
        return super().__call__(prepared_batch)  # type: ignore[no-any-return]

    def training_step(  # type: ignore
        self, prepared_batch: layers.PreparedBatch, batch_idx: int
    ) -> T.Dict[str, T.Union[Tensor, T.Dict[str, Tensor]]]:
        # TODO: What we pass to self() shouldn't contains .target
        logits = self(prepared_batch)
        loss = self._crs_entrpy(logits, prepared_batch.target)

        return {
            "loss": loss,
        }

    def validation_step(  # type: ignore
        self,
        prepared_batch: layers.PreparedBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> T.Dict[str, Tensor]:
        logits = self(prepared_batch)
        return {"logits": logits.detach(), "target": prepared_batch.target}

    def on_train_start(self) -> None:
        return
        one_batch: T.List[utils.GraphExample] = next(iter(self.train_dataloader()))
        # NOTE: The tb logger must be the first
        self.logger[0].experiment.add_graph(
            self._gat_model, (one_batch),
        )

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
            # (B, C)
            all_target = torch.cat([output["target"] for output in lsoutput])
            # (B,)

            all_preds = all_logits.argmax(dim=1)
            # (B,)
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
            txt_src=self._txt_src_per_split[self._val_name],
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
    all_config = configs.EverythingConfig(
        trainer=configs.TrainerConfig(
            lr=1e-3, train_batch_size=512, eval_batch_size=512, epochs=40,
        ),
        preprop=configs.PreprocessingConfig(
            undirected=True,
            # dataset_dir="actual_data/SST-2_tiny",
            dataset_dir="actual_data/SST-2_small",
            # dataset_dir="actual_data/SST-2",
            # dataset_dir="actual_data/glue_data/SST-2",
            # dataset_dir="actual_data/paraphrase/paws_small",
            sent2graph_name="dep",
            unk_thres=None,
        ),
        model=configs.GATForSequenceClassificationConfig(
            embedding_dim=300,
            gat_layered=configs.GATLayeredConfig(
                num_heads=5, intermediate_dim=300, feat_dropout_p=0.3, num_layers=12,
            ),
            node_embedding_type="bpe",
            bpe_vocab_size=25000,
            use_edge_features=True,
            dataset_dep=None,
            use_pretrained_embs=True,
        ),
    )

    early_stop_callback: T.Optional[EarlyStopping] = None
    if all_config.trainer.early_stop_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor="val_acc",
            min_delta=0.00,
            patience=all_config.trainer.early_stop_patience,
            verbose=False,
            mode="max",
        )

    model = LitGatForSequenceClassification(all_config)

    loggers: T.List[LightningLoggerBase] = []
    wandb_logger = WandbLogger(project="gat", sync_tensorboard=True)
    # tb_logger = TensorBoardLogger(save_dir=wandb_logger.experiment.dir)
    # TB logger must be first
    # loggers.append(tb_logger)
    loggers.append(wandb_logger)
    trainer = Trainer(
        logger=loggers,
        max_epochs=all_config.trainer.epochs,
        gpus=1,
        early_stop_callback=early_stop_callback,
    )

    trainer.fit(model)


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    main()
