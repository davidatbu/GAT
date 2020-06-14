import typing as T

import torch
from nose import tools  # type: ignore
from torch import nn
from tqdm import tqdm  # type: ignore

from ..config import base as config
from Gat.data import tokenizers
from Gat.neural import layers


class TestGraphMultiHeadAttention:
    def setUp(self) -> None:
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
        self.all_config = config.EverythingConfig(
            model=gat_config, trainer=trainer_config
        )
        self.seq_length = 13
        self.node_features: torch.FloatTensor = torch.randn(  # type: ignore
            self.all_config.trainer.train_batch_size,
            self.seq_length,
            self.all_config.model.embed_dim,
            names=("B", "N", "E"),
        ).cuda()

        self.adj: torch.BoolTensor = torch.randn(  # type: ignore
            self.all_config.trainer.train_batch_size,
            self.seq_length,
            self.seq_length,
            names=("B", "N_left", "N_right"),
        ).cuda() > 1
        self.adj.rename(None)[
            :, range(self.seq_length), range(self.seq_length)
        ] = True  # Make sure all the self loops are there

        self.node_labels = T.cast(
            torch.LongTensor,
            torch.zeros(  # type: ignore
                self.all_config.trainer.train_batch_size,
                self.seq_length,
                dtype=torch.long,
                names=("B", "N"),
            ),
        ).cuda()

        self.node_labels.rename(None)[:, range(13)] = torch.tensor(range(13)).cuda()

    def test_best_num_heads(self) -> None:

        crs_entrpy = nn.CrossEntropyLoss()
        n_steps = 5000

        steps_to_converge_for_num_heads: T.Dict[int, int] = {}
        for num_heads in (2, 4, 12, 64, 384):
            head_size = self.all_config.model.embed_dim // num_heads
            # Edge features are dependent on head size
            key_edge_features: torch.FloatTensor = torch.randn(  # type: ignore
                self.all_config.trainer.train_batch_size,
                self.seq_length,
                self.seq_length,
                head_size,
                names=("B", "N_left", "N_right", "head_size"),
                requires_grad=True,
                device="cuda",
            )

            multihead_att = layers.GraphMultiHeadAttention(
                embed_dim=self.all_config.model.embed_dim,
                num_heads=num_heads,
                include_edge_features=True,
                edge_dropout_p=0.3,
            )
            multihead_att.cuda()
            adam = torch.optim.Adam(
                # Include key_edge_features in features to be optimized
                [key_edge_features] + list(multihead_att.parameters()),  # type:ignore
                lr=1e-3,
            )
            multihead_att.train()
            for step in tqdm(range(1, n_steps + 1), desc=f"num_heads={num_heads}"):
                after_self_att = multihead_att(
                    adj=self.adj,
                    node_features=self.node_features,
                    key_edge_features=key_edge_features,
                )
                loss = crs_entrpy(
                    after_self_att.flatten(["B", "N"], "BN").rename(None),
                    self.node_labels.flatten(["B", "N"], "BN").rename(None),
                )
                preds = after_self_att.rename(None).argmax(dim=-1)
                if torch.all(torch.eq(preds, self.node_labels)):
                    steps_to_converge_for_num_heads[num_heads] = step
                    break
                loss.backward()
                adam.step()
            if not torch.all(torch.eq(preds, self.node_labels)):
                print(f"Did not converge for num_heads={num_heads}")
        print(
            "converged for heads: "
            + " | ".join(
                f"num_heads: {num_heads}, step: {steps}"
                for num_heads, steps in steps_to_converge_for_num_heads.items()
            )
        )


class TestEmbedder:
    def setUp(self) -> None:
        self._spacy_tokenizer = tokenizers.spacy.WrappedSpacyTokenizer()
        self._bert_tokenizer = tokenizers.bert.WrappedBertTokenizer()
        self._padding_idx = 0

        self._lslstok_id = [
            [1, 0],
            [2],
        ]

    def test_basic(self) -> None:
        basic_embedder = layers.BasicEmbedder(
            tokenizer=self._spacy_tokenizer,
            num_embeddings=3,
            embedding_dim=768,
            padding_idx=self._padding_idx,
        )
        embs = basic_embedder.forward(self._lslstok_id)
        tools.ok_(embs[0, 1].sum().item() == 0.0, msg="Padding vector not all zeros")
        embs_size = tuple(embs.size())
        tools.eq_(
            embs_size,
            (2, 2, basic_embedder.embedding_dim),
            msg=f"embs_size is {embs_size} "
            f"instead of {(2,2, basic_embedder.embedding_dim)}",
        )

    def test_pos(self) -> None:
        pos_embedder = layers.PositionalEmbedder(
            tokenizer=self._spacy_tokenizer, embedding_dim=768,
        )
        embs = pos_embedder.forward(self._lslstok_id)
        embs_size = tuple(embs.size())
        tools.eq_(
            embs_size,
            (2, 2, pos_embedder.embedding_dim),
            msg=f"embs_size is {embs_size} "
            f"instead of {(2,2, pos_embedder.embedding_dim)}",
        )

    def test_bert(self) -> None:
        lslstok = self._bert_tokenizer.batch_tokenize(
            ["i love embeddings."], "don't you?"]
        )
        embedder = layers.BertEmbedder(
            tokenizer=self._bert_tokenizer, last_how_many_layers=4
        )
        embs = embedder.forward(self._lslstok_id)
        embs_size = tuple(embs.size())
        tools.eq_(
            embs_size,
            (2, 2, embedder.embedding_dim),
            msg=f"embs_size is {embs_size} "
            f"instead of {(2,2, pos_embedder.embedding_dim)}",
        )
