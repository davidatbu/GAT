"""Tests for layers.

PyTest is useful. But it's not intuitive at all. Please check out how PyTest works
first.
"""
from __future__ import annotations

import tempfile
import typing as T
from pathlib import Path

import pytest
import torch
from torch import nn
from tqdm import tqdm  # type: ignore

from Gat.config import base as config
from Gat.neural import layers
from tests.conftest import VocabSetup


@pytest.fixture(scope="session")
def temp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="GAT_test"))


@pytest.fixture(scope="session")
def use_cuda() -> bool:
    # This was stolen from private torch code
    if not hasattr(torch._C, "_cuda_isDriverSufficient"):
        return False
    return torch.cuda.is_available()


class _GatSetup(T.NamedTuple):
    all_config: config.EverythingConfig[config.GATConfig]
    seq_length: int
    node_features: torch.Tensor
    adj: torch.Tensor
    node_labels: torch.Tensor


@pytest.fixture
def gat_setup(use_cuda: bool) -> _GatSetup:
    if use_cuda:
        device = "cuda"
    else:
        device = "cpu"

    gat_config = config.GATConfig(
        embed_dim=768,
        vocab_size=99,
        intermediate_dim=99,
        cls_id=99,
        nmid_layers=99,
        num_heads=99,
        nhid=99,
        nedge_type=99,
        embedder_type="bert",
    )
    trainer_config = config.TrainerConfig(
        lr=1e-3,
        epochs=99,
        dataset_dir="99",
        sent2graph_name="99",  # type: ignore
        train_batch_size=3,
        eval_batch_size=99,
    )
    all_config = config.EverythingConfig(model=gat_config, trainer=trainer_config)
    seq_length = 13
    node_features: torch.Tensor = torch.randn(  # type: ignore
        all_config.trainer.train_batch_size,
        seq_length,
        all_config.model.embed_dim,
        names=("B", "N", "E"),
        device=device,
    )

    adj: torch.Tensor = torch.randn(  # type: ignore
        all_config.trainer.train_batch_size,
        seq_length,
        seq_length,
        names=("B", "N_left", "N_right"),
        device=device,
    ) > 1
    adj.rename(None)[
        :, range(seq_length), range(seq_length)
    ] = True  # Make sure all the self loops are there

    node_labels: torch.Tensor = torch.zeros(  # type: ignore
        all_config.trainer.train_batch_size,
        seq_length,
        dtype=torch.long,
        names=("B", "N"),
        device=device,
    )

    node_labels.rename(None)[:, range(13)] = torch.tensor(range(13), device=device)

    return _GatSetup(
        all_config=all_config,
        seq_length=seq_length,
        node_features=node_features,
        adj=adj,
        node_labels=node_labels,
    )


def test_best_num_heads(gat_setup: _GatSetup, use_cuda: bool) -> None:

    if not use_cuda:
        print("test skipped. no cuda.")
        return

    crs_entrpy = nn.CrossEntropyLoss()
    n_steps = 5000

    if use_cuda:
        device = "cuda"
    else:
        device = "cpu"

    steps_to_converge_for_num_heads: T.Dict[int, int] = {}
    for num_heads in (2, 4, 12, 64, 384):
        head_size = gat_setup.all_config.model.embed_dim // num_heads
        # Edge features are dependent on head size
        key_edge_features: torch.FloatTensor = torch.randn(  # type: ignore
            gat_setup.all_config.trainer.train_batch_size,
            gat_setup.seq_length,
            gat_setup.seq_length,
            head_size,
            names=("B", "N_left", "N_right", "head_size"),
            requires_grad=True,
            device=device,
        )

        multihead_att = layers.GraphMultiHeadAttention(
            embed_dim=gat_setup.all_config.model.embed_dim,
            num_heads=num_heads,
            include_edge_features=True,
            edge_dropout_p=0.3,
        )
        if use_cuda:
            multihead_att.cuda()
        adam = torch.optim.Adam(
            # Include key_edge_features in features to be optimized
            [key_edge_features]
            + T.cast(T.List[torch.FloatTensor], list(multihead_att.parameters())),
            lr=1e-3,
        )
        multihead_att.train()
        for step in tqdm(range(1, n_steps + 1), desc=f"num_heads={num_heads}"):
            after_self_att = multihead_att(
                adj=gat_setup.adj,
                node_features=gat_setup.node_features,
                key_edge_features=key_edge_features,
            )
            loss = crs_entrpy(
                after_self_att.flatten(["B", "N"], "BN").rename(None),
                gat_setup.node_labels.flatten(["B", "N"], "BN").rename(None),
            )
            preds = after_self_att.rename(None).argmax(dim=-1)
            if torch.all(torch.eq(preds, gat_setup.node_labels)):
                steps_to_converge_for_num_heads[num_heads] = step
                break
            loss.backward()
            adam.step()
        if not torch.all(torch.eq(preds, gat_setup.node_labels)):
            print(f"Did not converge for num_heads={num_heads}")
    print(
        "converged for heads: "
        + " | ".join(
            f"num_heads: {num_heads}, step: {steps}"
            for num_heads, steps in steps_to_converge_for_num_heads.items()
        )
    )


def test_basic_embedder(vocab_setup: VocabSetup) -> None:
    basic_embedder = layers.BasicEmbedder(
        vocab=vocab_setup.basic_vocab,
        num_embeddings=3,
        embedding_dim=768,
        padding_idx=vocab_setup.basic_vocab.padding_tok_id,
    )
    embs = basic_embedder.forward(vocab_setup.lslstok_id)
    assert embs[0, 1].sum().item() == 0.0
    embs_size = tuple(embs.size())
    assert embs_size == (2, 2, basic_embedder.embedding_dim)


def test_pos_embedder(vocab_setup: VocabSetup) -> None:
    pos_embedder = layers.PositionalEmbedder(embedding_dim=768)
    embs = pos_embedder.forward(vocab_setup.lslstok_id)
    embs_size = tuple(embs.size())
    assert embs_size == (2, 2, pos_embedder.embedding_dim)


def test_bert(vocab_setup: VocabSetup, bert_embedder: layers.BertEmbedder) -> None:

    lssent = ["pacification", "something"]
    lslstok_id = vocab_setup.bert_vocab.batch_tokenize_and_get_tok_ids(lssent)

    our_embs = bert_embedder.forward(lslstok_id)

    max_seq_len = max(map(len, lslstok_id))
    assert our_embs.size("L") == max_seq_len
