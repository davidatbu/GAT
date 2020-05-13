from unittest import main
from unittest import TestCase

import torch
import torch.nn as nn
from torch import testing

from block import block_diag
from layers import AttHead
from layers import GATLayer


class BlockDiag(TestCase):
    def setUp(self) -> None:
        self.a = torch.tensor([[1, 0], [1, 1]])
        self.b = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 1]])

        self.a_b = torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1],
            ]
        )

    def tearDown(self) -> None:
        pass

    def testIt(self) -> None:
        res = block_diag(self.a, self.b)
        testing.assert_allclose(res, self.a_b)  # type: ignore


class TestAttHead(TestCase):
    def setUp(self) -> None:
        self.X = torch.tensor(
            [[1 * j for i in range(1, 11)] for j in range(1, 4)], dtype=torch.float
        )
        self.adj = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 0, 1]])
        self.y = torch.tensor([0, 1, 0,])
        self.head = AttHead(
            in_features=self.X.size(1),
            out_features=int(self.y.max().item()) + 1,
            edge_dropout_p=0.0,
            alpha=0.2,
            activ=False,
        )
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def testConverges(self) -> None:
        head = self.head

        adam = torch.optim.Adam(head.parameters(), lr=1e-4)
        head.train()
        n_steps = 800
        crs_entrpy = nn.CrossEntropyLoss()
        for step in range(n_steps):
            logits = head(self.X, self.adj)
            loss = crs_entrpy(logits, self.y)
            loss.backward()
            print(loss)
            adam.step()
        print(logits.argmax(dim=1))

        pass


class TestGatLayer(TestCase):
    def setUp(self) -> None:
        self.X = torch.tensor(
            [[1 * j for i in range(1, 11)] for j in range(1, 4)], dtype=torch.float
        )
        self.adj = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 0, 1]])
        self.y = torch.tensor([0, 1, 0])
        self.layer = GATLayer(
            nhid=2,
            nheads=5,
            in_features=self.X.size(1),
            out_features=int(self.y.max().item()) + 1,
            edge_dropout_p=0.0,
            feats_dropout_p=0,
            alpha=0.2,
            concat=False,
        )

    def tearDown(self) -> None:
        pass

    def testConverges(self) -> None:
        layer = self.layer

        adam = torch.optim.Adam(layer.parameters(), lr=1e-3)
        layer.train()
        n_steps = 800
        crs_entrpy = nn.CrossEntropyLoss()
        for step in range(n_steps):
            logits = layer(self.X, self.adj)
            loss = crs_entrpy(logits, self.y)
            loss.backward()
            print(loss)
            adam.step()
        print(logits.argmax(dim=1))


if __name__ == "__main__":
    main()
