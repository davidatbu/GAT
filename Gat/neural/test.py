import typing as T

import torch
from torch import nn
from tqdm import tqdm

from ..config import base as config
from .layers import GraphMultiHeadAttention


class TestGraphMultiHeadAttention:
    def setUp(self) -> None:
        gat_config = config.GATConfig(
            embed_dim=768,
            vocab_size=99,
            intermediate_dim=99,
            cls_id=99,
            nmid_layers=99,
            nheads=99,
            nhid=99,
            nedge_type=99,
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
            torch.LongTensor, torch.zeros(self.all_config.trainer.train_batch_size, self.seq_length, dtype=torch.long, names=("B", "N"))  # type: ignore
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

            multihead_att = GraphMultiHeadAttention(
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
