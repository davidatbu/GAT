import logging
from utils import load_splits
from pathlib import Path

import torch
import torch.nn as nn
from nose import main
from nose import tools
from torch import Tensor
from torch import testing
from torch.utils.data import DataLoader

from block import block_diag
from config import GATConfig
from config import GATForSeqClsfConfig
from data import FromIterableTextSource
from data import SentenceGraphDataset
from data import VocabAndEmb
from glove_embeddings import GloveWordToVec
from layers import AttHead
from layers import GATLayer
from layers import GATLayerWrapper
from models import GATForSeqClsf
from sent2graph import SRLSentenceToGraph

logger = logging.getLogger("__main__")


class BlockDiagTest:
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

    def testIt(self) -> None:
        res = block_diag(self.a, self.b)
        testing.assert_allclose(res, self.a_b)  # type: ignore


class BaseGat:
    def setUp(self) -> None:
        self.X = torch.tensor(
            [[(i * j) % 7 for i in range(1, 11)] for j in range(1, 4)],
            dtype=torch.float,
        )
        self.adj = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 0, 1]])
        self.y = torch.tensor([0, 1, 9])

        self.config = GATConfig(
            embedding_dim=self.X.size(1),
            cls_id=1,
            vocab_size=self.X.size(0),
            nmid_layers=5,
            nhid=5,
            nheads=2,
        )


class TestAttHead(BaseGat):
    def setUp(self) -> None:
        super().setUp()
        self.head = AttHead(self.config)

    def test_converges(self) -> None:
        head = self.head

        adam = torch.optim.Adam(head.parameters(), lr=1e-3)
        head.train()
        n_steps = 1000
        crs_entrpy = nn.CrossEntropyLoss()
        for step in range(n_steps):
            logits = head(self.X, self.adj)
            loss = crs_entrpy(logits, self.y)
            loss.backward()
            print(loss)
            adam.step()
        print(logits.argmax(dim=1))

        pass


class TestGATLayer(BaseGat):
    def test_concat(self) -> None:
        layer = GATLayer(self.config, concat=False)

        logits: Tensor = layer(self.X, self.adj)
        tools.eq_(logits.size(1), self.config.nhid)

    def test_non_concat(self) -> None:
        layer = GATLayer(self.config)
        logits: Tensor = layer(self.X, self.adj)
        tools.eq_(logits.size(1), self.config.embedding_dim)

    def test_converges(self) -> None:
        layer = GATLayer(self.config, concat=True)

        adam = torch.optim.Adam(layer.parameters(), lr=1e-3)
        layer.train()
        n_steps = 800
        crs_entrpy = nn.CrossEntropyLoss()
        print(self.X)
        print(self.y)
        for step in range(n_steps):
            logits = layer(self.X, self.adj)
            loss = crs_entrpy(logits, self.y)
            loss.backward()
            print(loss)
            adam.step()
        print(logits.argmax(dim=1))


class TestGATLayerWrapper(BaseGat):
    def test_converges(self) -> None:
        layer = GATLayerWrapper(self.config)

        adam = torch.optim.Adam(layer.parameters(), lr=1e-3)
        layer.train()
        n_steps = 800
        crs_entrpy = nn.CrossEntropyLoss()
        print(self.X)
        print(self.y)
        for step in range(n_steps):
            logits = layer(self.X, self.adj)
            loss = crs_entrpy(logits, self.y)
            loss.backward()
            print(loss)
            adam.step()
        print(logits.argmax(dim=1))


class TestGATForSeqClsf:
    def setUp(self) -> None:
        self.txt_src = FromIterableTextSource(
            [("I love you.", "positive"), ("I hate you.", "negative"),]
        )

        self.vocab_and_emb = VocabAndEmb(
            txt_src=self.txt_src,
            cache_dir=Path("/scratch"),
            embedder=GloveWordToVec(),
            unk_thres=1,
        )

        self.dataset = SentenceGraphDataset(
            txt_src=self.txt_src,
            cache_dir=Path("/scratch"),
            sent2graph=SRLSentenceToGraph(),
            vocab_and_emb=self.vocab_and_emb,
        )

        self.loader = DataLoader(
            self.dataset, collate_fn=SentenceGraphDataset.collate_fn, batch_size=2
        )

        self.config = GATForSeqClsfConfig(
            nclass=len(self.vocab_and_emb._id2lbl),
            vocab_size=self.vocab_and_emb.embs.size(0),
            embedding_dim=self.vocab_and_emb.embs.size(1),
            cls_id=self.vocab_and_emb._cls_id,
            nmid_layers=3,
            nhid=50,
            nheads=6,
        )

        self.gat_seq_clsf = GATForSeqClsf(self.config, self.vocab_and_emb.embs)

    def test_batching(self) -> None:
        X, y = next(iter(self.loader))

        # Rember vocab_and_emb._real_tokens_start
        expected_global_lsnode = [3, 4, 5, 6, 3, 7, 5, 6]
        expected_lsedge_index = [
            (0, 1),
            (1, 2),
            (1, 0),
            (2, 1),
            (4, 5),
            (5, 6),
            (5, 4),
            (6, 5),
        ]
        expected_lslshead_node = [[1, 3], [5, 7]]

        batch = self.gat_seq_clsf.prepare_batch(X)
        global_lsnode, lsedge_index, lslshead_node = batch
        tools.eq_(expected_lsedge_index, lsedge_index)
        tools.eq_(expected_global_lsnode, global_lsnode)
        tools.eq_(expected_lslshead_node, lslshead_node)

    def test_converges(self) -> None:
        gat_seq_clsf = self.gat_seq_clsf
        adam = torch.optim.Adam(gat_seq_clsf.parameters(), lr=1e-3)
        gat_seq_clsf.train()
        n_steps = 20
        X, y = next(iter(self.loader))

        print(X)
        print(y)
        for step in range(n_steps):
            logits, loss = gat_seq_clsf(X, y)
            loss.backward()
            print(loss)
            adam.step()
        preds = logits.argmax(dim=1).detach().numpy().tolist()
        print([self.vocab_and_emb._id2lbl[i] for i in preds])

    def tearDown(self) -> None:
        pass

    def test(self) -> None:
        pass

class TestOverfit:
    def setUp(self) -> None:
        datasets_per_split, vocab_and_emb = load_splits(
        Path(
            "/projectnb/llamagrp/davidat/projects/graphs/data/ready/gv_2018_1160_examples/raw/"
        )
    )
        self.vocab_and_emb = vocab_and_emb
        self.train_dataset = datasets_per_split["train"]
        self.val_dataset = datasets_per_split["val"]

    
        self.train_loader = DataLoader(
            self.train_dataset, collate_fn=SentenceGraphDataset.collate_fn, batch_size=2
        )

        self.config = GATForSeqClsfConfig(
            nclass=len(self.vocab_and_emb._id2lbl),
            vocab_size=self.vocab_and_emb.embs.size(0),
            embedding_dim=self.vocab_and_emb.embs.size(1),
            cls_id=self.vocab_and_emb._cls_id,
            nmid_layers=3,
            nhid=50,
            nheads=6,
        )

        self.gat_seq_clsf = GATForSeqClsf(self.config, self.vocab_and_emb.embs)

    def test_overfit(self) -> None:
        gat_seq_clsf = self.gat_seq_clsf
        adam = torch.optim.Adam(gat_seq_clsf.parameters(), lr=1e-3)
        gat_seq_clsf.train()
        n_steps = 20
        X, y = next(iter(self.loader))

        print(X)
        print(y)
        for step in range(n_steps):
            logits, loss = gat_seq_clsf(X, y)
            loss.backward()
            print(loss)
            adam.step()
        preds = logits.argmax(dim=1).detach().numpy().tolist()
        print([self.vocab_and_emb._id2lbl[i] for i in preds])

    def tearDown(self) -> None:
        pass

    def test(self) -> None:
        pass



if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
