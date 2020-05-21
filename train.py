import datetime
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
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

from config import GATForSeqClsfConfig
from config import TrainConfig
from data import load_splits
from data import SentenceGraphDataset
from data import SliceDataset
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
    train_dataset: Dataset,  # type: ignore
    val_dataset: Dataset,  # type: ignore
    base_tb_dir: Path,
    tqdm_position: int = 0,
) -> Dict[str, float]:
    logger.info("About to try: " + pformat(parameterization))

    # Unpack configs
    model_config, remaining_config = GATForSeqClsfConfig.from_dict(parameterization)
    remaining_config.update({"collate_fn": SentenceGraphDataset.collate_fn})
    train_config, remaining_config = TrainConfig.from_dict(remaining_config)
    assert len(remaining_config) == 0

    model = GATForSeqClsf(model_config, **model_kwargs)  # type: ignore
    if train_config.use_cuda:  # type: ignore
        model.cuda()
    fmted_time = datetime.datetime.now().strftime("%y%m%d.%H%M%S")
    base_tb_dir.mkdir(exist_ok=True)
    tb_dir = base_tb_dir / f"{fmted_time}"
    tb_dir.mkdir(exist_ok=True)
    train_tb_writer = SummaryWriter(str(tb_dir / "train/"))
    val_tb_writer = SummaryWriter(str(tb_dir / "val/"))

    results = train(
        model,
        data_loader_kwargs={},
        train_config=train_config,  # type: ignore
        tqdm_position=tqdm_position,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_tb_writer=train_tb_writer,
        val_tb_writer=val_tb_writer,
    )
    train_tb_writer.close()
    val_tb_writer.close()
    return results


def main() -> None:

    dataset_dir = Path("data/glue_data/SST-2")
    datasets_per_split, vocab_and_emb = load_splits(
        dataset_dir, lstxt_col=["sentence"], splits=["train", "dev"]
    )
    train_dataset = datasets_per_split["train"]
    val_dataset = datasets_per_split["dev"]
    vocab_size = vocab_and_emb.embs.size(0)

    model_kwargs = {"emb_init": vocab_and_emb.embs}

    lsparameterization: List[Dict[str, Any]] = [
        {
            "lr": 3e-4,
            "train_batch_size": 256,
            "eval_batch_size": 256,
            "epochs": 20,
            "vocab_size": vocab_size,
            "cls_id": vocab_and_emb._cls_id,
            "nhid": 50,
            "nheads": 6,
            "embedding_dim": 300,
            "nclass": len(vocab_and_emb._id2lbl),
            "nmid_layers": 6,
            "nedge_type": 99999,
        }
    ]

    for parameterization in lsparameterization:
        tune_hparam(
            parameterization,
            model_kwargs=model_kwargs,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            base_tb_dir=dataset_dir / "tb_logs",
        )


def train(
    model: GATForSeqClsf,
    train_dataset: Dataset,  # type: ignore
    val_dataset: Dataset,  # type: ignore
    data_loader_kwargs: Dict[str, Any],
    train_config: TrainConfig,
    train_tb_writer: Optional[SummaryWriter] = None,
    val_tb_writer: Optional[SummaryWriter] = None,
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
    for epoch in range(1, train_config.epochs + 1):
        pbar_desc = f"epoch: {epoch}"
        pbar = tqdm(train_loader, desc=pbar_desc, position=2 * tqdm_position)
        for X, y in pbar:
            prepared_X = model.prepare_batch(X)
            train_one_step(model, optimizer, prepared_X, y)
            examples_seen += train_config.train_batch_size
            batches_seen += 1

        if train_config.do_eval_every_epoch:
            eval_metrics, _, _ = evaluate(model, val_dataset, train_config)
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
        all_y: Optional[np.ndarray] = None
        for X, y in val_loader:
            prepared_X = model.prepare_batch(X)
            logits, one_step_loss = model(prepared_X=prepared_X, y=y)
            loss += one_step_loss

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
                all_y = np.array(y)
            else:
                all_logits = np.concatenate(
                    [all_logits, logits.detach().cpu().numpy()], axis=0
                )
                all_y = np.concatenate([all_y, np.array(y)], axis=0)

    all_preds = np.argmax(all_logits, axis=1)
    acc: float = skmetrics.accuracy_score(all_y, all_preds)
    loss /= len(val_loader)
    metrics = {
        "loss": float(loss.item()),
        "acc": acc,
    }

    assert all_logits is not None
    assert all_y is not None
    return (metrics, all_logits, all_y)


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
