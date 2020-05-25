import logging
from pathlib import Path
from pprint import pformat
from timeit import default_timer as timer
from typing import Any
from typing import List
from typing import Tuple
from typing import Union

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
from config import TrainConfig
from data import FromIterableTextSource
from data import load_splits
from data import SentenceGraphDataset
from data import VocabAndEmb
from glove_embeddings import GloveWordToVec
from layers import DotProductAttHead
from layers import GATLayer
from layers import GATLayerWrapper
from models import GATForSeqClsf
from sent2graph import SRLSentenceToGraph
from utils import flatten
from utils import grouper
from utils import html_table
from utils import reshape_like
from utils import SentExample

# from train import train

logger = logging.getLogger("__main__")
logger.setLevel(logging.INFO)


class Timer:
    def __init__(self, msg: str, fmt: str = "%0.3g") -> None:
        self.msg = msg
        self.fmt = fmt

    def __enter__(self) -> "Timer":
        self.start = timer()
        return self

    def __exit__(self, *args: Any) -> None:
        t = timer() - self.start
        pformat(("%s : " + self.fmt + " seconds") % (self.msg, t))
        self.time = t


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
        self.y = torch.tensor([0, 1, 4])

        self.config = GATConfig(
            embedding_dim=self.X.size(1),
            cls_id=1,
            vocab_size=self.X.size(0),
            nmid_layers=5,
            nhid=5,
            nheads=2,
            nedge_type=9999999,
        )


class TestAttHead(BaseGat):
    def setUp(self) -> None:
        super().setUp()
        self.head = DotProductAttHead(self.config)

    def test_converges(self) -> None:
        head = self.head

        adam = torch.optim.Adam(head.parameters(), lr=5e-4)
        head.train()
        n_steps = 1000
        crs_entrpy = nn.CrossEntropyLoss()
        for step in range(n_steps):
            logits = head(self.X, self.adj)
            loss = crs_entrpy(logits, self.y)
            loss.backward()
            print(pformat(loss))
            adam.step()
        print(pformat(logits.argmax(dim=1)))


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
        print(pformat(self.X))
        print(pformat(self.y))
        for step in range(n_steps):
            logits = layer(self.X, self.adj)
            loss = crs_entrpy(logits, self.y)
            loss.backward()
            print(pformat(loss))
            adam.step()
        print(pformat(logits.argmax(dim=1)))


class TestGATLayerWrapper(BaseGat):
    def test_converges(self) -> None:
        layer = GATLayerWrapper(self.config)

        adam = torch.optim.Adam(layer.parameters(), lr=1e-3)
        layer.train()
        n_steps = 800
        crs_entrpy = nn.CrossEntropyLoss()
        print(pformat(self.X))
        print(pformat(self.y))
        for step in range(n_steps):
            logits = layer(self.X, self.adj)
            loss = crs_entrpy(logits, self.y)
            loss.backward()
            print(pformat(loss))
            adam.step()
        print(pformat(logits.argmax(dim=1)))


class TestGATForSeqClsf:
    def setUp(self) -> None:
        self.txt_src = FromIterableTextSource(
            [
                SentExample(["I love you."], "positive"),
                SentExample(["I hate you."], "negative"),
            ]
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
            nhid=75,
            nedge_type=4,
            nheads=4,
        )

        self.gat_seq_clsf = GATForSeqClsf(self.config, self.vocab_and_emb.embs)


class TestGATForSeqClsfBasic(TestGATForSeqClsf):
    def test_batching(self) -> None:
        X, y = next(iter(self.loader))

        # Rember vocab_and_emb._real_tokens_start
        expected_nodeid2wordid = [3, 4, 5, 6, 3, 7, 5, 6]
        expected_lsedge = [
            (0, 1),
            (1, 2),
            (4, 5),
            (5, 6),
            (1, 0),
            (2, 1),
            (5, 4),
            (6, 5),
        ]
        expected_lslsimp_node = [[1, 3], [5, 7]]

        peeled_X = [ex[0] for ex in X]
        batch = self.gat_seq_clsf.prepare_batch(peeled_X)
        lsedge, lsedge_type, lslsimp_node, nodeid2wordid = batch
        tools.eq_(expected_lsedge, lsedge)
        tools.eq_(expected_nodeid2wordid, nodeid2wordid)
        tools.eq_(expected_lslsimp_node, lslsimp_node)

    def test_converges(self) -> None:
        gat_seq_clsf = self.gat_seq_clsf
        adam = torch.optim.Adam(gat_seq_clsf.parameters(), lr=1e-3)
        gat_seq_clsf.train()
        n_steps = 20
        X, y = next(iter(self.loader))

        print(pformat(X))
        print(pformat(y))
        for step in range(n_steps):
            logits, loss = gat_seq_clsf(X, y)
            loss.backward()
            print(pformat(loss))
            adam.step()
        preds = logits.argmax(dim=1).detach().numpy().tolist()
        print(pformat([self.vocab_and_emb._id2lbl[i] for i in preds]))

    def tearDown(self) -> None:
        pass

    def test(self) -> None:
        pass


class TestOverfit:
    def setUp(self) -> None:
        datasets_per_split, vocab_and_emb = load_splits(
            Path("data/SST-2_tiny"),
            splits=["train"],
            lstxt_col=["sentence"],
            unk_thres=1,
        )
        self.vocab_and_emb = vocab_and_emb
        self.train_dataset = datasets_per_split["train"]

        self.train_config = TrainConfig(
            lr=1e-3,
            epochs=10,
            train_batch_size=2,
            eval_batch_size=10,
            collate_fn=SentenceGraphDataset.collate_fn,
            do_eval_every_epoch=True,
        )

        model_config = GATForSeqClsfConfig(
            nclass=len(self.vocab_and_emb._id2lbl),
            nedge_type=9999,
            vocab_size=self.vocab_and_emb.embs.size(0),
            embedding_dim=self.vocab_and_emb.embs.size(1),
            cls_id=self.vocab_and_emb._cls_id,
            nmid_layers=0,
            nhid=50,
            nheads=6,
            feat_dropout_p=0.9,
        )

        self.gat_seq_clsf = GATForSeqClsf(model_config, self.vocab_and_emb.embs)

    def test_overfit(self) -> None:
        gat_seq_clsf = self.gat_seq_clsf
        train_config = self.train_config

        (
            before_train_eval_metrics,
            before_train_all_logits,
            before_train_all_y,
        ) = evaluate(gat_seq_clsf, self.train_dataset, train_config)
        logger.info(
            f"BEFORE TRAINING: eval metrics: {pformat(before_train_eval_metrics)}"
        )
        logger.info(f"BEFORE TRAINING: eval logits: {pformat(before_train_all_logits)}")
        logger.info(f"BEFORE TRAINING: eval y: {pformat(before_train_all_y)}")

        gat_seq_clsf.train()
        train(
            gat_seq_clsf,
            train_dataset=self.train_dataset,
            val_dataset=self.train_dataset,
            data_loader_kwargs={},
            train_config=train_config,
        )

    def tearDown(self) -> None:
        pass


class TestGATSanity(TestGATForSeqClsf):
    def test_inital_loss(self) -> None:
        X, y = next(iter(self.loader))
        logits, loss = self.gat_seq_clsf.forward(X, y)
        print(pformat(f"len(X))={len(X)}"))
        print(pformat(loss))


class TestShapers:
    def setUp(self) -> None:
        self.nested = [
            [[[4, 5], [1, 2], [0, 3]], [0, 1, 2], [3], [123, 234, 345, 456, 567, 678]]
        ]
        self.flat = [4, 5, 1, 2, 0, 3, 0, 1, 2, 3, 123, 234, 345, 456, 567, 678]

        self.nested2 = [
            [
                [[40, 45], [41, 42], [0, 3]],
                [0, 51, 52],
                [51323],
                [123, 234, 345, 456, 567, 678],
            ]
        ]

    def test_flatten(self) -> None:
        assert list(flatten(self.nested)) == self.flat

    def test_reshape(self) -> None:
        reshaped, consumed = reshape_like(self.flat, self.nested2)
        ls_reshaped = list(reshaped)
        assert ls_reshaped == self.nested
        assert consumed == len(self.flat)

    def test_grouper(self) -> None:
        grouped = list(grouper(self.flat, n=4))
        assert len(set(map(len, grouped))) <= 2
        print(pformat(grouped))
        flattened = list(flatten(grouped))
        assert flattened == self.flat


class TestSentenceGraphDataset:
    def setUp(self) -> None:
        self.txt_src = FromIterableTextSource(
            [
                SentExample(["I love you."], "positive"),
                SentExample(["I hate you."], "negative"),
                SentExample(["I adore you."], "negative"),
            ]
        )

        self.vocab_and_emb = VocabAndEmb(
            txt_src=self.txt_src,
            cache_dir=Path("/scratch"),
            embedder=None,
            unk_thres=1,
        )

        self.sent2graph = SRLSentenceToGraph()
        self.dataset = SentenceGraphDataset(
            txt_src=self.txt_src,
            cache_dir=Path("/scratch"),
            sent2graph=self.sent2graph,
            vocab_and_emb=self.vocab_and_emb,
            processing_batch_size=2,
        )

        self.loader = DataLoader(
            self.dataset, collate_fn=SentenceGraphDataset.collate_fn, batch_size=2
        )

    def test_vocab_and_emb(self) -> None:
        assert self.vocab_and_emb._id2word == [
            "[PAD]",
            "[CLS]",
            "[UNK]",
            "i",
            "love",
            "you",
            ".",
            "hate",
            "adore",
        ]

    def test_dataset(self) -> None:
        print(pformat([i for i in self.dataset]))

    def test_loader(self) -> None:
        print(pformat([i for i in self.loader]))


class Test:
    def setUp(self) -> None:
        datasets_per_split, vocab_and_emb = load_splits(
            Path("data/glue_data/SST-2"),
            splits=["dev", "train"],
            lstxt_col=["sentence"],
        )
        self.vocab_and_emb = vocab_and_emb
        self.train_dataset = datasets_per_split["train"]

        self.train_config = TrainConfig(
            lr=1e-3,
            epochs=10,
            train_batch_size=2,
            eval_batch_size=10,
            collate_fn=SentenceGraphDataset.collate_fn,
            do_eval_every_epoch=True,
        )

        model_config = GATForSeqClsfConfig(
            nclass=len(self.vocab_and_emb._id2lbl),
            nedge_type=9999,
            vocab_size=self.vocab_and_emb.embs.size(0),
            embedding_dim=self.vocab_and_emb.embs.size(1),
            cls_id=self.vocab_and_emb._cls_id,
            nmid_layers=0,
            nhid=50,
            nheads=6,
            feat_dropout_p=0.9,
        )

        self.gat_seq_clsf = GATForSeqClsf(model_config, self.vocab_and_emb.embs)

    def test_overfit(self) -> None:
        gat_seq_clsf = self.gat_seq_clsf
        train_config = self.train_config

        (
            before_train_eval_metrics,
            before_train_all_logits,
            before_train_all_y,
        ) = evaluate(gat_seq_clsf, self.train_dataset, train_config)
        logger.info(
            f"BEFORE TRAINING: eval metrics: {pformat(before_train_eval_metrics)}"
        )
        logger.info(f"BEFORE TRAINING: eval logits: {pformat(before_train_all_logits)}")
        logger.info(f"BEFORE TRAINING: eval y: {pformat(before_train_all_y)}")

        gat_seq_clsf.train()
        train(
            gat_seq_clsf,
            train_dataset=self.train_dataset,
            val_dataset=self.train_dataset,
            data_loader_kwargs={},
            train_config=train_config,
        )

    def tearDown(self) -> None:
        pass


class TestHtmlTable:
    def setUp(self) -> None:
        self.headers = ("Sentence", "Predicted", "Gold")
        self.data: List[Tuple[Union[str, float, int], ...]] = [
            ("I love you", 1, 1),
            ("I hate evil", 0, 0),
            (
                "The best way to ruin your life is to try to save it with your own wisdom",
                1,
                0,
            ),
        ]
        self.colors = [None, None, "red"]

    def test_it(self) -> None:
        print(html_table(self.data, self.headers, self.colors))


if __name__ == "__main__":
    main()
