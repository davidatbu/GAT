import datetime
import logging
import random
from pathlib import Path
from pprint import pformat
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import sklearn.metrics as skmetrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

import wandb  # type: ignore
from config import Config
from config import EverythingConfig
from config import GATForSeqClsfConfig
from config import TrainConfig
from data import load_splits
from data import SliceDataset
from data import TextSource
from data import VocabAndEmb
from models import GATForSeqClsf


logger = logging.getLogger("__main__")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)  # type: ignore


def main() -> None:
    dataset_dir = Path("data/glue_data/SST-2")
    val_name = "dev"
    datasets, txt_srcs, vocab_and_emb = load_splits(
        dataset_dir, lstxt_col=["sentence"], splits=["train", val_name]
    )
    vocab_size = vocab_and_emb.embs.size(0)

    all_config = EverythingConfig(
        trainer=TrainConfig(
            lr=1e-3,
            train_batch_size=128,
            eval_batch_size=128,
            epochs=20,
            dataset_dir=str(dataset_dir),
        ),
        model=GATForSeqClsfConfig(
            vocab_size=vocab_size,
            cls_id=vocab_and_emb._cls_id,
            nhid=50,
            nheads=6,
            embedding_dim=300,
            feat_dropout_p=0.3,
            nclass=len(vocab_and_emb._id2lbl),
            nmid_layers=6,
            nedge_type=len(datasets["train"].sent2graph.id2edge_type),
        ),
    )

    logger.info("About to try: " + pformat(all_config))
    wandb.init(project="gat", config=all_config.as_dict())

    model = GATForSeqClsf(all_config.model, emb_init=vocab_and_emb.embs)
    if all_config.trainer.use_cuda:
        model.cuda()

    train(
        model,
        data_loader_kwargs={},
        train_config=all_config.trainer,
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
    )


def train(
    model: GATForSeqClsf,
    train_dataset: Dataset,  # type: ignore
    val_dataset: Dataset,  # type: ignore
    data_loader_kwargs: Dict[str, Any],
    train_config: TrainConfig,
    analyze_predict_kwargs: Optional[Dict[str, Any]] = None,
    tqdm_position: int = 0,
) -> Dict[str, float]:
    # Model and optimizer
    optimizer = optim.Adam(model.parameters(), lr=train_config.lr)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.train_batch_size,
        collate_fn=train_config.collate_fn,
        **data_loader_kwargs,
    )

    examples_seen = 0
    batches_seen = 0
    if train_tb_writer is not None:
        X, _ = next(iter(train_loader))
        prepared_X = model.prepare_batch(X)
        train_tb_writer.add_graph(model, (prepared_X,))

    train_dataset_slice = SliceDataset(train_dataset, n=len(val_dataset))
    eval_metrics, _, _ = evaluate(model, val_dataset, train_config)
    train_metrics, _, _ = evaluate(model, train_dataset_slice, train_config)
    logger.info(
        f"Before training | eval: {eval_metrics} | partial train: {train_metrics}"
    )
    running_loss = torch.tensor(
        0, dtype=torch.float, device=next(model.parameters()).device
    )
    for epoch in range(1, train_config.epochs + 1):
        pbar_desc = f"epoch: {epoch}"
        pbar = tqdm(train_loader, desc=pbar_desc, position=2 * tqdm_position)
        for X, y in pbar:
            prepared_X = model.prepare_batch(X)
            running_loss += train_one_step(model, optimizer, prepared_X, y)
            examples_seen += train_config.train_batch_size
            batches_seen += 1
            pbar.set_description(
                f"{pbar_desc} | train loss running: {(running_loss / batches_seen).item()}"
            )

        if train_config.do_eval_every_epoch:
            eval_metrics, eval_logits, eval_true = evaluate(
                model, val_dataset, train_config
            )
            train_metrics, _, _ = evaluate(model, train_dataset_slice, train_config)
            logger.info(
                f"{pbar_desc} | eval: {eval_metrics} | partial train: {train_metrics}"
            )

            if val_tb_writer is not None:
                for metric, value in eval_metrics.items():
                    val_tb_writer.add_scalar(metric, value, global_step=examples_seen)

            if train_tb_writer is not None:
                for metric, value in train_metrics.items():
                    train_tb_writer.add_scalar(metric, value, global_step=examples_seen)

            if analyze_predict_kwargs is not None:
                analyze_predict(
                    logits=eval_logits,
                    true_=eval_true,
                    tb_writer=val_tb_writer,
                    **analyze_predict_kwargs,
                )

    if not train_config.do_eval_every_epoch:
        eval_metrics, _, _ = evaluate(model, val_dataset, train_config)
        train_metrics, _, _ = evaluate(model, train_dataset, train_config)
        logger.info("eval results: " + pformat(eval_metrics))

        if val_tb_writer is not None:
            for metric, value in eval_metrics.items():
                val_tb_writer.add_scalar(metric, value, global_step=examples_seen)

        if train_tb_writer is not None:
            for metric, value in eval_metrics.items():
                train_tb_writer.add_scalar(metric, value, global_step=examples_seen)

    return eval_metrics


def evaluate(
    model: GATForSeqClsf,
    val_dataset: Dataset,  # type: ignore
    train_config: TrainConfig,
    tqdm_position: int = 0,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.eval_batch_size,
        collate_fn=train_config.collate_fn,
    )

    model.eval()
    with torch.no_grad():
        loss = torch.tensor(
            0, dtype=torch.float, device=next(model.parameters()).device
        )
        all_logits: Optional[np.ndarray] = None
        all_true: Optional[np.ndarray] = None
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


def analyze_predict(logits: np.ndarray, true_: np.ndarray, tb_writer: SummaryWriter, dataset: Dataset, vocab_and_emb: VocabAndEmb, txt_src: TextSource) -> None:  # type: ignore
    matches: np.ndarray = logits == true_
    correct_indices = matches.nonzero()
    incorrect_indices = (~matches).nonzero()

    tb_writer.add_text()


def train_one_step(
    model: nn.Module, optimizer: optim.Optimizer, prepared_X: Any, y: Any  # type: ignore
) -> Tensor:
    model.train()  # Turn on the train mode
    optimizer.zero_grad()
    logits, loss = model(prepared_X=prepared_X, y=y)
    loss.backward()
    # Clipping here maybe?
    optimizer.step()
    return loss  # type: ignore


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    main()
