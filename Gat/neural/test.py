import typing as T

import torch
from torch import nn
from tqdm import tqdm

from .layers import GraphMultiHeadSelfAttention


class TestGraphMultiHeadSelfAttention:
    def setUp(self) -> None:
        self.N = 13
        self.E = 24
        self.B = 3

        self.node_features: torch.FloatTensor = torch.randn(  # type: ignore
            self.B, self.N, self.E, names=("B", "N", "E")
        )

        self.adj: torch.BoolTensor = torch.randn(  # type: ignore
            self.B, self.N, self.N, names=("B", "N_left", "N_right")
        ) > 1
        self.adj.rename(None)[
            :, range(self.N), range(self.N)
        ] = True  # Make sure all the self loops are there

        self.node_labels = T.cast(
            torch.LongTensor, torch.zeros(self.B, self.N, dtype=torch.long, names=("B", "N"))  # type: ignore
        )

        # Name all the nodes of index less than self.N//2 with embedding_dim-1 ( so that the node
        # features can serve as logit inputs to cross entorpy)
        self.node_labels[:, 0 : self.N // 2] = self.E - 1

    def test_best_num_heads(self) -> None:

        crs_entrpy = nn.CrossEntropyLoss()
        n_steps = 5000

        steps_to_converge_for_num_heads: T.Dict[int, int] = {}
        for num_heads in (2, 4, 6, 12):
            head_size = self.E // num_heads
            # Edge features are dependent on head size
            key_edge_features: torch.FloatTensor = torch.randn(  # type: ignore
                self.B,
                self.N,
                self.N,
                head_size,
                names=("B", "N_left", "N_right", "head_size"),
                requires_grad=True,
            )

            multihead_att = GraphMultiHeadSelfAttention(
                embed_dim=self.E, num_heads=num_heads, include_edge_features=True
            )
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
                f"num_heads: {num_heads}, step: {step}"
                for num_heads, steps in steps_to_converge_for_num_heads.items()
            )
        )
