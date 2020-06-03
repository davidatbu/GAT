import typing as T

import torch
from torch import nn
from tqdm import tqdm

from .layers import GraphMultiHeadSelfAttention

torch.manual_seed(1)  # type: ignore


class TestGraphMultiHeadSelfAttention:
    def setUp(self) -> None:
        N = 13
        E = 10
        B = 3
        self.E = E

        self.node_features: torch.FloatTensor = torch.randn(  # type: ignore
            B, N, E, names=("B", "N", "E")
        )

        self.adj: torch.BoolTensor = torch.randn(  # type: ignore
            B, N, N, names=("B", "N_left", "N_right")
        ) > 1
        self.adj.rename(None)[
            :, range(N), range(N)
        ] = True  # Make sure all the self loops are there

        self.node_labels = T.cast(
            torch.LongTensor, torch.zeros(B, N, dtype=torch.long, names=("B", "N"))  # type: ignore
        )

        # Name all the nodes of index less than N//2 with embedding_dim-1 ( so that the node
        # features can serve as logit inputs to cross entorpy)
        self.node_labels[:, 0 : N // 2] = E - 1

    def test_best_num_heads(self) -> None:

        crs_entrpy = nn.CrossEntropyLoss()
        n_steps = 5000

        steps_to_converge_for_num_heads: T.Dict[int, int] = {}
        for num_heads in (1, 2, 5, 10):
            multihead_att = GraphMultiHeadSelfAttention(
                embed_dim=self.E, num_heads=num_heads, num_edge_types=1
            )
            adam = torch.optim.Adam(multihead_att.parameters(), lr=1e-3)
            multihead_att.train()
            for step in tqdm(range(1, n_steps + 1), desc=f"num_heads={num_heads}"):
                after_self_att = multihead_att(self.node_features, self.adj)
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
                "num_steps_to_converge: "
                + " | ".join(
                    f"num_heads: {num_heads}, step: {step}"
                    for num_heads, steps in steps_to_converge_for_num_heads.items()
                )
            )
