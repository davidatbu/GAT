import logging
import random
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import sklearn.metrics as skmetrics
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from ray import tune
from torch import Tensor
from torch.utils.data import DataLoader

from config import GATForSeqClsfConfig
from config import TrainConfig
from data import load_splits
from data import SentenceGraphDataset
from models import GATForSeqClsf

logger = logging.getLogger("__main__")


def parse_args() -> Namespace:
    parser = ArgumentParser()

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # type: ignore

    return args


def main() -> None:
    datasets_per_split, vocab_and_emb = load_splits(
        Path(
            "/projectnb/llamagrp/davidat/projects/graphs/data/ready/gv_2018_1160_examples/raw/"
        )
    )
    train_dataset = datasets_per_split["train"]
    val_dataset = datasets_per_split["val"]

    vocab_size = vocab_and_emb.embs.size(0)

    search_space = {
        "lr": tune.loguniform(1e-2, 1e-5),
        "train_batch_size": tune.choice([1]),
        "eval_batch_size": tune.choice([1]),
        "epochs": tune.grid_search(list(range(2, 10))),
        "collate_fn": tune.grid_search([SentenceGraphDataset.collate_fn]),
        "vocab_size": tune.grid_search([vocab_size]),
        "cls_id": tune.grid_search([vocab_and_emb._cls_id]),
        "nhid": tune.grid_search([50]),
        "nheads": tune.grid_search([6]),
        "embedding_dim": tune.grid_search([300]),
        "nmid_layers": tune.grid_search([0]),
    }

    def tune_hparam(tune_config: Dict[str, Any]) -> None:

        # Unpack configs
        model_config, remaining_config = GATForSeqClsfConfig.from_dict(tune_config)
        train_config, remaining_config = TrainConfig.from_dict(tune_config)
        assert len(remaining_config) == 0

        assert isinstance(model_config, GATForSeqClsfConfig)
        model = GATForSeqClsf(model_config, emb_init=vocab_and_emb.embs)

        assert isinstance(train_config, TrainConfig)
        train(model, train_dataset, val_dataset, train_config)

    analysis = tune.run(tune_hparam, config=search_space)

    logdir = str(analysis.get_best_logdir(metric="val_acc"))
    print(f"BEST LOG DIR IS {logdir}")


def train(
    model: GATForSeqClsf,
    train_dataset: SentenceGraphDataset,
    val_dataset: SentenceGraphDataset,
    train_config: TrainConfig,
) -> None:
    # Model and optimizer
    optimizer = optim.Adam(model.parameters(), lr=train_config.lr)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.train_batch_size,
        collate_fn=train_config.collate_fn,
    )

    running_loss = torch.tensor(9, dtype=torch.float)
    steps = 0
    for epoch in range(train_config.epochs):
        for X, y in tqdm(train_loader, desc="training "):
            _, one_step_loss = train_one_step(model, optimizer, X, y)
            steps += 1
        running_loss += one_step_loss
        running_mean_train_loss = (running_loss / steps).item()

        eval_result = evaluate(model, val_dataset, train_config)
        tune.track.log(running_mean_train_loss=running_mean_train_loss, **eval_result)


# TODO: Use bigger batch size for validation
def evaluate(
    model: GATForSeqClsf, val_dataset: SentenceGraphDataset, train_config: TrainConfig,
) -> Dict[str, float]:

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
        for X, y in tqdm(val_loader, desc="validating "):
            logits, one_step_loss = model(X=X, y=y)
            loss += one_step_loss

            if all_logits is None:
                all_logits = logits.numpy()
                all_y = np.array(y)
            else:
                all_logits = np.concatenate([all_logits, logits.numpy()], axis=0)
                all_y = np.concatenate([all_y, np.array(y)], axis=0)

    all_preds = np.argmax(all_logits, axis=1)
    acc = skmetrics.accuracy_score(all_y, all_preds)
    loss /= len(val_loader)
    return {
        "val_loss": loss.item(),
        "val_acc": acc,
    }


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
    logger.setLevel(logging.DEBUG)
    main()
