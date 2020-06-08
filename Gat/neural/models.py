from functools import lru_cache
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..config.base import GATConfig
from ..config.base import GATForSeqClsfConfig
from ..neural import layers
from ..utils.base import Edge
from ..utils.base import EdgeType
from ..utils.base import Node
from ..utils.base import SentGraph


class GATLayered(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig, emb_init: Optional[Tensor]):
        super().__init__()

        self.token_embedder = layers.Embedder()  # type: ignore # TODO
        self.key_edge_embedder = layers.Embedder()  # type: ignore # TODO
        self.positional_embedder = layers.PositionalEmbedder(config.embed_dim)
        self.lsmultihead_att_wrapper = nn.ModuleList(
            [
                layers.GraphMultiHeadAttentionWrapped(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    include_edge_features=config.include_edge_features,
                    edge_dropout_p=config.edge_dropout_p,
                    rezero_or_residual=config.rezero_or_residual,
                )
                for _ in range(config.nmid_layers)
            ]
        )
        self.lsfeed_forward_wrapper = nn.ModuleList(
            [
                layers.FeedForwardWrapped(
                    in_out_dim=config.embed_dim,
                    intermediate_dim=config.intermediate_dim,
                    rezero_or_residual=config.rezero_or_residual,
                    feat_dropout_p=config.feat_dropout_p,
                )
                for _ in range(config.nmid_layers)
            ]
        )

    def forward(
        self, word_ids: Tensor, position_ids: Tensor, adj: Tensor, edge_types: Tensor
    ) -> Tensor:
        node_features = self.token_embedder(word_ids)
        node_features = node_features + self.positional_embedder(word_ids)
        key_edge_features = self.key_edge_embedder(edge_types)

        for multihead_att_wrapper, feed_forward_wrapper in zip(
            self.lsmultihead_att_wrapper, self.lsfeed_forward_wrapper
        ):
            node_features = multihead_att_wrapper(
                node_features=node_features,
                adj=adj,
                key_edge_features=key_edge_features,
            )

            node_features = feed_forward_wrapper(node_features=node_features)

        return node_features


class GATModel(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig, emb_init: Optional[Tensor]):
        super().__init__()
        self.cls_id = config.cls_id
        self.head_to_cls_edge_type = config.nedge_type
        self.undirected = config.undirected
        self.gat_layered = GATLayered(config, emb_init)

    def _coalesce_graph(
        self, batch: List[SentGraph]
    ) -> Tuple[List[Edge], List[EdgeType], List[List[Node]], List[int], List[int]]:
        """
        Increment the relative node numbers in the adjacency list, and the list of key nodes

        Also, add edges in the other direction if we're doing unidirected.
        """

        lsedge: List[Edge] = []
        lsedge_type: List[EdgeType] = []
        lslsimp_node: List[List[Node]] = []
        nodeid2wordid: List[int] = []
        lsposition_id: List[int] = []

        counter = 0

        for one_lsedge, one_lsedge_type, one_lsimp_node, one_nodeid2wordid in batch:
            assert one_nodeid2wordid is not None
            # Extend nodeid2wordid
            nodeid2wordid.extend(one_nodeid2wordid)

            # Extend edge index, but increment the numbers
            lsedge.extend(
                [(edge[0] + counter, edge[1] + counter) for edge in one_lsedge]
            )

            # Extend edge type
            lsedge_type.extend(one_lsedge_type)

            # Extend lslshead node as well, but increment the numbers
            lslsimp_node.append([node + counter for node in one_lsimp_node])

            # Add position ids
            lsposition_id.extend(range(len(one_nodeid2wordid)))

            # Increment node counter
            counter += len(one_nodeid2wordid)

        if self.undirected:
            setedge: Set[Edge] = set()  # Use to keep a unique list of nodes
            new_lsedge: List[Edge] = []
            new_lsedge_type: List[EdgeType] = []

            for (n1, n2), edge_type in zip(lsedge, lsedge_type):
                if n1 > n2:
                    n2, n1 = n1, n2
                if (n1, n2) not in setedge:
                    new_lsedge.extend([(n1, n2), (n2, n1)])
                    new_lsedge_type.extend([edge_type, edge_type])
                    setedge.add((n1, n2))

            lsedge = new_lsedge
            lsedge_type = new_lsedge_type
        return (lsedge, lsedge_type, lslsimp_node, nodeid2wordid, lsposition_id)


class GATForUnorderedSeqPairClsf(GATModel):
    raise NotImplementedError()


class GATForSeqClsf(GATModel):
    def __init__(self, config: GATForSeqClsfConfig, emb_init: Optional[Tensor]) -> None:
        super().__init__(config, emb_init=emb_init)
        embed_dim = config.embed_dim
        nclass = config.nclass
        feat_dropout_p = config.feat_dropout_p

        self.linear = nn.Linear(embed_dim, nclass)
        self.dropout = nn.Dropout(p=feat_dropout_p)
        self.crs_entrpy = nn.CrossEntropyLoss()

    @staticmethod
    def peeled_batch_yielder(batch: List[List[SentGraph]]) -> Iterator[SentGraph]:
        for ex in batch:
            yield ex[0]

    @lru_cache(maxsize=int(1e6))
    def connect_to_cls_node(self, sentgraph: SentGraph) -> SentGraph:
        # Connect all the "head nodes" to a new [CLS] node
        lsedge, lsedge_type, lsimp_node, nodeid2wordid = sentgraph
        assert nodeid2wordid is not None
        assert self.cls_id not in nodeid2wordid
        new_nodeid2wordid = nodeid2wordid + [self.cls_id]

        new_cls_node = len(new_nodeid2wordid) - 1
        lshead_to_cls_edge = [(node, new_cls_node) for node in lsimp_node]
        lshead_to_cls_edge_type = [self.head_to_cls_edge_type for _ in lsimp_node]
        new_lsedge = lsedge + lshead_to_cls_edge
        new_lsedge_type = lsedge_type + lshead_to_cls_edge_type

        new_lsimp_node = [new_cls_node]

        return SentGraph(
            lsedge=new_lsedge,
            lsedge_type=new_lsedge_type,
            lsimp_node=new_lsimp_node,
            nodeid2wordid=new_nodeid2wordid,
        )

    def prepare_batch(
        self, batch: List[List[SentGraph]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
         For each graph in batch
            connect each key node to a new, "CLS" node
            make that "CLS" node the only node in list of key nodes
        do super()._coalesce_graph()
        """

        # Ensure that we are processing only one sentgraph per example
        assert set(map(len, batch)) == {1}

        new_batch: List[SentGraph] = [
            self.connect_to_cls_node(sentgraph)
            for sentgraph in self.peeled_batch_yielder(batch)
        ]

        final_pre_torch_preped_X = self._coalesce_graph(new_batch)
        (
            lsedge,
            lsedge_type,
            lslsimp_node,
            nodeid2wordid,
            lsposition_id,
        ) = final_pre_torch_preped_X
        # "unpack" lscls_node ,since per batch, we're only looking at output of CLS token
        assert set(map(len, lslsimp_node)) == {1}

        # Device
        device = next(self.parameters()).device

        # word ids
        word_ids = torch.tensor(nodeid2wordid, dtype=torch.long, device=device)
        position_ids = torch.tensor(lsposition_id, dtype=torch.long, device=device)

        # ADjacency
        N = len(nodeid2wordid)
        adj: torch.Tensor = torch.zeros(N, N, dtype=torch.float, device=device)
        adj[list(zip(*lsedge))] = 1

        # Edge types
        edge_type: Tensor = torch.tensor(lsedge_type, dtype=torch.long, device=device)
        # Cls node
        cls_node = torch.tensor([lsimp_node[0] for lsimp_node in lslsimp_node])

        return (word_ids, position_ids, adj, edge_type, cls_node)

    def forward(
        self,
        prepared_X: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        y: Optional[List[int]] = None,
    ) -> Tuple[Tensor, ...]:
        """

        Returns
        -------
        """
        word_ids, position_ids, adj, edge_type, cls_node = prepared_X

        h = self.gat_layered(
            word_ids=word_ids, position_ids=position_ids, adj=adj, edge_type=edge_type
        )

        cls_id_h = h[cls_node]

        cls_id_h = self.dropout(cls_id_h)
        logits = self.linear(cls_id_h)

        if y is not None:
            new_y = torch.tensor(y, device=next(self.parameters()).device)
            loss = self.crs_entrpy(logits, new_y)
            return (logits, loss)
        return (logits,)
