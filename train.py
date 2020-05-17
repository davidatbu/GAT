import logging
import random
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from pprint import pformat
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import sklearn.metrics as skmetrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

from config import GATForSeqClsfConfig
from config import TrainConfig
from data import load_splits
from data import SentenceGraphDataset
from models import GATForSeqClsf

# from multiprocessing import Pool

logger = logging.getLogger("__main__")


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--exp_name", "-n", type=str, default="experiment")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # type: ignore

    return args


def tune_hparam(
    parameterization: Dict[str, Any],
    model_kwargs: Dict[str, Any],
    train_func_kwargs: Dict[str, Any],
    tqdm_position: int,
) -> Dict[str, Tuple[float, Optional[float]]]:
    logger.info(f"About to try: " + pformat(parameterization))

    # Unpack configs
    model_config, remaining_config = GATForSeqClsfConfig.from_dict(parameterization)
    remaining_config.update({"collate_fn": SentenceGraphDataset.collate_fn})
    train_config, remaining_config = TrainConfig.from_dict(remaining_config)
    assert len(remaining_config) == 0

    assert isinstance(model_config, GATForSeqClsfConfig)
    model = GATForSeqClsf(model_config, **model_kwargs)

    assert isinstance(train_config, TrainConfig)
    return train(
        model,
        data_loader_kwargs={},
        train_config=train_config,
        tqdm_position=tqdm_position,
        **train_func_kwargs,
    )


def main() -> None:

    dataset_dir = Path(
        "/project/llamagrp/davidat/projects/graphs/pyGAT/data/gv_2018_1160_examples/"
    )
    datasets_per_split, vocab_and_emb = load_splits(dataset_dir)
    train_dataset = datasets_per_split["train"]
    val_dataset = datasets_per_split["val"]
    vocab_size = vocab_and_emb.embs.size(0)

    args = parse_args()
    # default `log_dir` is "runs" - we'll be more specific here
    tb_writer = SummaryWriter(str(dataset_dir / "tb_run/" / args.ex_name))

    train_func_kwargs = {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "tb_writer": tb_writer,
    }
    model_kwargs = {"emb_init": vocab_and_emb.embs}

    lsparameterization: List[Dict[str, Any]] = [
        {"name": "lr", "type": "choice", "values": [1e-2, 1e-3, 1e-4]},
        {"name": "train_batch_size", "type": "choice", "values": [128, 64, 32, 4]},
        {"name": "eval_batch_size", "type": "fixed", "value": 64},
        {"name": "epochs", "type": "choice", "values": [15, 10, 9, 8, 7, 3]},
        {"name": "vocab_size", "type": "fixed", "value": vocab_size},
        {"name": "cls_id", "type": "fixed", "value": vocab_and_emb._cls_id},
        {"name": "nhid", "type": "fixed", "value": 50},
        {"name": "nheads", "type": "fixed", "value": 6},
        {"name": "embedding_dim", "type": "fixed", "value": 300},
        {"name": "nclass", "type": "fixed", "value": len(vocab_and_emb._id2lbl)},
        {"name": "nmid_layers", "type": "choice", "values": [3, 4, 5, 6, 7, 10, 12]},
    ]

    for i, parameterization in enumerate(lsparameterization):
        tune_hparam(
            parameterization,
            model_kwargs=model_kwargs,
            tqdm_position=i,
            train_func_kwargs=train_func_kwargs,
        )


def train(
    model: GATForSeqClsf,
    train_dataset: SentenceGraphDataset,
    val_dataset: SentenceGraphDataset,
    data_loader_kwargs: Dict[str, Any],
    train_config: TrainConfig,
    tb_writer: SummaryWriter,
    tqdm_position: int = 0,
) -> Dict[str, Tuple[float, Optional[float]]]:
    # Model and optimizer
    optimizer = optim.Adam(model.parameters(), lr=train_config.lr)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.train_batch_size,
        collate_fn=train_config.collate_fn,
        **data_loader_kwargs,
    )

    tb_writer.add_graph(model)

    running_loss = torch.tensor(9, dtype=torch.float)
    examples_seen = 0
    batches_seen = 0
    for epoch in tqdm(
        range(train_config.epochs), position=2 * tqdm_position, desc="training epochs"
    ):
        for X, y in tqdm(
            train_loader, desc="training batches seen", position=2 * tqdm_position + 1
        ):
            one_step_loss = train_one_step(model, optimizer, X, y)
            examples_seen += train_config.train_batch_size
            running_loss += one_step_loss
            batches_seen += 1

        if train_config.do_eval_every_epoch:
            eval_metrics, _, _ = evaluate(model, val_dataset, train_config)
            running_loss += one_step_loss
            running_mean_train_loss = (running_loss / batches_seen).item()
            eval_metrics.update({"avg_train_loss": (running_mean_train_loss, None)})
            logger.info(f"eval results: " + pformat(eval_metrics))

            for metric, value in eval_metrics.items():
                pass

    if not train_config.do_eval_every_epoch:
        eval_metrics, _, _ = evaluate(model, val_dataset, train_config)
        running_mean_train_loss = (running_loss / examples_seen).item()
        eval_metrics.update({"avg_train_loss": (running_mean_train_loss, None)})
        logger.info(f"eval results: " + pformat(eval_metrics))

    return eval_metrics


# TODO: Use bigger batch size for validation
def evaluate(
    model: GATForSeqClsf,
    val_dataset: SentenceGraphDataset,
    train_config: TrainConfig,
    tqdm_position: int = 0,
) -> Tuple[Dict[str, Tuple[float, Optional[float]]], np.ndarray, np.ndarray]:

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.eval_batch_size,
        collate_fn=train_config.collate_fn,
    )

    model.eval()
    with torch.no_grad():
        loss = torch.tensor(0, dtype=torch.float)
        all_logits: Optional[np.ndarray] = None
        all_y: Optional[np.ndarray] = None
        for X, y in tqdm(
            val_loader, desc="validating batches seen", position=2 * tqdm_position
        ):
            print(X, y)
            logits, one_step_loss = model(X=X, y=y)
            loss += one_step_loss

            if all_logits is None:
                all_logits = logits.detach().numpy()
                all_y = np.array(y)
            else:
                all_logits = np.concatenate(
                    [all_logits, logits.detach().numpy()], axis=0
                )
                all_y = np.concatenate([all_y, np.array(y)], axis=0)

    all_preds = np.argmax(all_logits, axis=1)
    acc: float = skmetrics.accuracy_score(all_y, all_preds)
    loss /= len(val_loader)
    eval_metrics: Dict[str, Tuple[float, Optional[float]]] = {
        "val_loss": (float(loss.item()), None),
        "val_acc": (acc, None),
    }

    assert all_logits is not None
    assert all_y is not None
    return (eval_metrics, all_logits, all_y)


def train_one_step(
    model: nn.Module, optimizer: optim.Optimizer, X: Any, y: Any  # type: ignore
) -> Tensor:
    model.train()  # Turn on the train mode
    optimizer.zero_grad()
    logits, loss = model(X=X, y=y)
    loss.backward()
    # Clipping here maybe?
    optimizer.step()
    return loss  # type: ignore


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    main()
