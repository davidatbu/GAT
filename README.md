# Using multi-head attention with graph neural networks

This repo has code experimenting with using graph neural networks with multi head attention to do text tasks (like classification). 

There are many helpers with reading data, tokenizing, ..etc, but the main Pytorch module in this repo has the following `.forward()` signature.

```python
class GraphMultiHeadAttention(nn.Module): 
    def __init__(
        self, embed_dim: int, num_heads: int, edge_dropout_p: float,
    ): ...
    def forward(
        self,
        node_features: torch.Tensor,
        batched_adj: torch.Tensor,
        key_edge_features: T.Optional[torch.Tensor] = None,
        value_edge_features: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """.
        Args:
            batched_adj: (B, L, L)
                batched_adj[b, i, j] means there's a directed edge from the i-th node to
                the j-th node of the b-th graph in the batch.
                That means, node features of the j-th node will affect the calculation
                of the node features of the i-th node.
            node_features: (B, N, E)
            value_edge_features and key_edge_features: (B, L_left, L_right, E)
                edge_features[b, i, j, :] are the features of the directed edge from
                the i-th node to the j-th node of the b-th graph in the batch.
        Returns:
            result: (B, N, E)
        """
```
