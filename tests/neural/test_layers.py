"""Tests for layers.

PyTest is useful. But it's not intuitive at all. Please check out how PyTest works
first.
"""
from __future__ import annotations

import tempfile
import typing as T
import unittest
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm  # type: ignore

from Gat.neural import layers
from tests.common import DeviceMixin
from tests.common import EverythingConfigMixin


class TestGraphMultiHeadAttention(
    DeviceMixin, EverythingConfigMixin, unittest.TestCase
):
    def setUp(self) -> None:
        super().setUp()
        self._seq_length = 6
        self._node_features = torch.randn(
            [
                self._all_config.trainer.train_batch_size,
                self._seq_length,
                self._all_config.model.embedding_dim,
            ],
            device=self._device,
        )
        # (B, L, E)
        self._batched_adj: torch.BoolTensor = (
            torch.randn(
                [
                    self._all_config.trainer.train_batch_size,
                    self._seq_length,
                    self._seq_length,
                ],
                device=self._device,
            )
            > 1
        )
        # (B, L, L)

        self._batched_adj[
            :, range(self._seq_length), range(self._seq_length)
        ] = True  # Make sure all the self loops are there
        self._node_labels = torch.zeros(
            [self._all_config.trainer.train_batch_size, self._seq_length],
            dtype=torch.long,
        )
        # (B, L)

        self._node_labels[:, range(self._seq_length)] = torch.tensor(
            range(self._seq_length)
        )
        # (B, L)

    def test_best_num_heads(self) -> None:

        if not self._device == torch.device("cuda"):
            lsnum_heads = [4, 12]
        else:
            lsnum_heads = [2, 4, 12, 64, 384]

        crs_entrpy = nn.CrossEntropyLoss()
        n_steps = 5000

        steps_to_converge_for_num_heads: T.Dict[int, int] = {}
        for num_heads in lsnum_heads:
            head_size = self._all_config.model.embedding_dim // num_heads
            # Edge features are dependent on head size
            key_edge_features: torch.FloatTensor = torch.randn(  # type: ignore
                [
                    self._all_config.trainer.train_batch_size,
                    self._seq_length,
                    self._seq_length,
                    head_size,
                ],
                requires_grad=True,
                device=self._device,
                dtype=torch.float,
            )
            # (B, L, L, head_size)

            multihead_att = layers.GraphMultiHeadAttention(
                embed_dim=self._all_config.model.embedding_dim,
                num_heads=num_heads,
                edge_dropout_p=0.3,
            )
            multihead_att.to(self._device)
            adam = torch.optim.Adam(
                # Include key_edge_features in features to be optimized
                [key_edge_features]
                + T.cast(T.List[torch.FloatTensor], list(multihead_att.parameters())),
                lr=1e-3,
            )
            multihead_att.train()
            for step in tqdm(range(1, n_steps + 1), desc=f"num_heads={num_heads}"):
                after_self_att = multihead_att(
                    batched_adj=self._batched_adj,
                    node_features=self._node_features,
                    key_edge_features=key_edge_features,
                )
                # (B, L, E)
                B, L, E = after_self_att.size()
                loss = crs_entrpy(
                    after_self_att.view(B * L, E), self._node_labels.view(B * L)
                )
                # (,)
                preds = after_self_att.argmax(dim=-1)
                # (B, L)
                if torch.all(torch.eq(preds, self._node_labels)):
                    steps_to_converge_for_num_heads[num_heads] = step
                    break
                loss.backward()
                adam.step()
            if not torch.all(torch.eq(preds, self._node_labels)):
                print(f"Did not converge for num_heads={num_heads}")
        print(
            "converged for heads: "
            + " | ".join(
                f"num_heads: {num_heads}, step: {steps}"
                for num_heads, steps in steps_to_converge_for_num_heads.items()
            )
        )
