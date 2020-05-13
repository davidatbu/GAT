import torch
import torch.nn as nn
from torch import Tensor


class AttHead(nn.Module):  # type: ignore
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_dropout_p: float,
        alpha: float,
        activ: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.activ = activ

        self.W = nn.Linear(in_features, out_features, bias=False)

        self.a = nn.Parameter(torch.zeros(2 * out_features, 1, dtype=torch.float))  # type: ignore
        nn.init.xavier_uniform_(
            self.a, gain=nn.init.calculate_gain("leaky_relu", alpha)  # type: ignore
        )
        # self.a2 = torch.empty(out_features, 1, dtype=torch.float)
        # nn.init.xavier_uniform_(
        # self.a, gain=nn.init.calculate_gain("leaky_relu", alpha)  # type: ignore
        # )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=1)
        self.edge_dropout = nn.Dropout(p=edge_dropout_p)
        self.elu = nn.ELU()

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:  # type: ignore

        h = self.W(input)
        N = h.size(0)

        a_input = torch.cat(
            [h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1
        ).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # f_1 = h @ self.a1
        # f_2 = h @ self.a2
        # e = self.leakyrelu(f_1 + f_2.transpose(0, 1))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = self.softmax(attention)
        attention = self.edge_dropout(attention)
        h_prime = torch.matmul(attention, h)

        if self.activ:
            return self.elu(h_prime)
        else:
            return h_prime

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GATLayer(nn.Module):  # type: ignore
    def __init__(
        self,
        in_features: int,
        out_features: int,
        nhid: int,
        nheads: int,
        edge_dropout_p: float,
        feats_dropout_p: float,
        alpha: float,  # For LeakyRELU
        concat: bool = False,
    ):
        if not nhid * nheads == in_features:
            raise Exception("nhid * nheads != out_features")
        super(GATLayer, self).__init__()
        self.concat = concat

        self.attentions = nn.ModuleList(
            [
                AttHead(
                    in_features=in_features,
                    out_features=nhid,
                    edge_dropout_p=edge_dropout_p,
                    alpha=alpha,
                    activ=not (concat),
                )
                for _ in range(nheads)
            ]
        )
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        self.dropout = nn.Dropout(p=feats_dropout_p)

    def forward(self, h: Tensor, adj: Tensor) -> torch.Tensor:  # type: ignore
        h = self.dropout(h)
        lsatt_res = [att(h, adj) for att in self.attentions]
        if self.concat:
            h = torch.cat(lsatt_res, dim=1)
        else:
            h = torch.stack(lsatt_res, dim=0).mean(dim=0)
        return h
