from itertools import chain
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from srl_graph import SRLDataset
from srl_graph import SrlResp # needed for unpickling to work

dataset_dir = Path(
    "/projectnb/llamagrp/davidat/projects/graphs/data/ready/gv_2018_1160_examples"
)
dataset = SRLDataset(
    root=dataset_dir,
    undirected_edges=True,
    has_val=True,
    has_test=False,
    use_cache=True,
    sent_col="news_title",
    label_col="Q3 Theme1",
)


class Net(torch.nn.Module):  # type: ignore
    def __init__(self, heads: int, per_head_h: int, layers: int) -> None:
        super(Net, self).__init__()
        self.first_conv = GATConv(
            int(dataset.num_features), per_head_h, heads=heads, dropout=0
        )
        # On the Pubmed dataset, use heads=8 in conv2.
        self.middle_convs = ModuleList(
            [
                GATConv(per_head_h * heads, per_head_h, heads=heads, dropout=0)
                for i in range(layers - 2)
            ]
        )
        self.final_conv = GATConv(
            per_head_h * heads,
            int(dataset.num_classes),
            heads=1,
            concat=True,
            dropout=0,
        )

    def forward(self, data: Data) -> torch.Tensor:  # type: ignore
        x = data.x

        for conv in chain([self.first_conv], self.middle_convs, [self.final_conv]):
            x = F.dropout(x, p=0.6, training=self.training)  # type: ignore
            x = F.elu(conv(x, data.edge_index))

        return F.log_softmax(x, dim=1)


def train(
    model: torch.nn.Module,  # type: ignore
    optimizer: torch.optim.Optimizer,  # type: ignore
    data: Data,
) -> None:
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test(model: torch.nn.Module, data: Data) -> List[float]:  # type: ignore
    model.eval()
    logits, accs = model(data), []
    for _, mask in data("train_mask", "val_mask"):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = dataset[0]
    data.train_mask = data.train_mask.bool()
    data.val_mask = data.val_mask.bool()
    model, data = Net(per_head_h=8, heads=8, layers=6).to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for epoch in range(1, 3000):
        train(model, optimizer, data)
        log = "Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}"
        print(log.format(epoch, *test(model, data)))


if __name__ == "__main__":
    main()
