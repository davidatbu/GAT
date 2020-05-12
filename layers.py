import torch
import torch.nn as nn


class GATHead(nn.Module[torch.Tensor]):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_p: float,
        alpha: float,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = torch.empty(in_features, out_features).type(torch.float)
        nn.init.xavier_uniform_(
            self.W, gain=nn.init.calculate_gain("relu")  # type: ignore
        )

        self.a = torch.empty(2 * out_features, 1, dtype=torch.float)
        nn.init.xavier_uniform_(
            self.a, gain=nn.init.calculate_gain("leaky_relu", alpha)  # type: ignore
        )
        # self.a1 = torch.empty(out_features, 1).type(torch.float)
        # nn.init.xavier_uniform(
        # self.a1, gain=np.sqrt(2.0),
        # )
        # self.a2 = torch.empty(out_features, 1).type(torch.float)
        # nn.init.xavier_uniform(self.a2, gain=np.sqrt(2.0))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.elu = nn.ELU()

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:  # type: ignore
        h = input @ self.W
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
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, h)

        if self.concat:
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
