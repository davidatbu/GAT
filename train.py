import logging
import random
from argparse import ArgumentParser
from argparse import Namespace
from multiprocessing import Pool
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
from ax.service.ax_client import AxClient  # type: ignore
from ax.service.utils.instantiation import TParameterRepresentation  # type: ignore
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

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


dataset_dir = Path(
    "/projectnb/llamagrp/davidat/projects/graphs/data/ready/gv_2018_1160_examples/raw/"
)

datasets_per_split, vocab_and_emb = load_splits(dataset_dir)
train_dataset = datasets_per_split["train"]
val_dataset = datasets_per_split["val"]

vocab_size = vocab_and_emb.embs.size(0)


def tune_hparam(
    parameterization: Dict[str, Any]
) -> Dict[str, Tuple[float, Optional[float]]]:
    logger.info(f"About to try: " + pformat(parameterization))

    # Unpack configs
    model_config, remaining_config = GATForSeqClsfConfig.from_dict(parameterization)
    remaining_config.update({"collate_fn": SentenceGraphDataset.collate_fn})
    train_config, remaining_config = TrainConfig.from_dict(remaining_config)
    assert len(remaining_config) == 0

    assert isinstance(model_config, GATForSeqClsfConfig)
    model = GATForSeqClsf(model_config, emb_init=vocab_and_emb.embs)

    assert isinstance(train_config, TrainConfig)
    return train(
        model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        data_loader_kwargs={},
        train_config=train_config,
    )


def main() -> None:

    parameters: List[TParameterRepresentation] = [
        {"name": "lr", "type": "fixed", "value": 1e-3},
        {"name": "train_batch_size", "type": "choice", "values": [32, 16, 4]},
        {"name": "eval_batch_size", "type": "fixed", "value": 64},
        {"name": "epochs", "type": "range", "bounds": [3, 5]},
        {"name": "vocab_size", "type": "fixed", "value": vocab_size},
        {"name": "cls_id", "type": "fixed", "value": vocab_and_emb._cls_id},
        {"name": "nhid", "type": "fixed", "value": 50},
        {"name": "nheads", "type": "fixed", "value": 6},
        {"name": "embedding_dim", "type": "fixed", "value": 300},
        {"name": "nclass", "type": "fixed", "value": len(vocab_and_emb._id2lbl)},
        {"name": "nmid_layers", "type": "choice", "values": [5, 6, 3, 2]},
    ]
    ax_client = AxClient(enforce_sequential_optimization=True)
    ax_client.create_experiment(
        name="gat_expermient",
        parameters=parameters,
        objective_name="val_acc",
        minimize=False,
    )
    trials: List[Tuple[Dict[str, Any], int]] = [
        ax_client.get_next_trial() for _ in range(24)
    ]

    lsparameterization, lstrial_index = zip(*trials)
    assert set(map(type, lsparameterization)) == {dict}
    with Pool(20) as p:
        lseval_result = p.map(tune_hparam, lsparameterization)
        # Local evaluation here can be replaced with deployment to external system.
    for trial_index, eval_result in zip(lstrial_index, lseval_result):
        ax_client.complete_trial(trial_index=trial_index, raw_data=eval_result)

    ax_client.save_to_json_file(str(dataset_dir / "ax_client_snapshot.json"))


def train(
    model: GATForSeqClsf,
    train_dataset: SentenceGraphDataset,
    val_dataset: SentenceGraphDataset,
    data_loader_kwargs: Dict[str, Any],
    train_config: TrainConfig,
) -> Dict[str, Tuple[float, Optional[float]]]:
    # Model and optimizer
    optimizer = optim.Adam(model.parameters(), lr=train_config.lr)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.train_batch_size,
        collate_fn=train_config.collate_fn,
        **data_loader_kwargs,
    )

    running_loss = torch.tensor(9, dtype=torch.float)
    steps = 0
    for epoch in tqdm(range(train_config.epochs), desc="training epochs"):
        for X, y in tqdm(train_loader, desc="training steps"):
            one_step_loss = train_one_step(model, optimizer, X, y)
            steps += 1
        running_loss += one_step_loss

    eval_result = evaluate(model, val_dataset, train_config)
    running_mean_train_loss = (running_loss / steps).item()
    eval_result.update({"avg_train_loss": (running_mean_train_loss, None)})
    logger.info(f"eval results: " + pformat(eval_result))
    return eval_result


# TODO: Use bigger batch size for validation
def evaluate(
    model: GATForSeqClsf, val_dataset: SentenceGraphDataset, train_config: TrainConfig,
) -> Dict[str, Tuple[float, Optional[float]]]:

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
        for X, y in tqdm(val_loader, desc="validating steps"):
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
    acc = skmetrics.accuracy_score(all_y, all_preds)
    loss /= len(val_loader)
    return {
        "val_loss": (loss.item(), None),
        "val_acc": (acc, None),
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
    logger.setLevel(logging.INFO)
    main()
