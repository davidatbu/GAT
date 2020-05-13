from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from config import GATConfig
from config import GATForSeqClsfConfig
from layers import EmbeddingWrapper
from layers import GATLayerWrapper
from utils import Edge
from utils import Node
from utils import to_undirected


class GATLayered(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig, emb_init: Optional[Tensor]):
        super().__init__()

        do_residual = config.do_residual
        self.emb_wrapper = EmbeddingWrapper(config, emb_init)
        self.lsmid_layer_wrapper = nn.ModuleList(
            [
                GATLayerWrapper(config, do_residual=do_residual)
                for _ in range(config.nmid_layers)
            ]
        )
        self.last_layer = GATLayerWrapper(config, do_residual=False, concat=False)

    def forward(self, tcword_id: Tensor, adj: Tensor) -> Tensor:  # type: ignore
        h = self.emb_wrapper(tcword_id)

        for layer_wrapper in self.lsmid_layer_wrapper:
            h = layer_wrapper(h, adj)

        h = self.last_layer(h, adj)

        return h  # type: ignore


class GATModel(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig, emb_init: Optional[Tensor]):
        super().__init__()
        self.cls_id = config.cls_id
        self.gat_layered = GATLayered(config, emb_init)

    def prepare_batch(
        self, batch: List[Tuple[List[Node], List[Edge], List[Node]]]
    ) -> Tuple[List[Node], List[Edge], List[List[Node]]]:
        """
        Increment the relative node numbers in the adjacency list, and the list of key nodes
        """

        lsglobal_node: List[Node] = []
        lsedge_index: List[Edge] = []
        lslshead_node: List[List[Node]] = []
        counter = 0

        assert self.cls_id not in lsglobal_node

        for one_lsglobal_node, one_lsedge_index, one_lshead_node in batch:
            # Extend global node ids
            lsglobal_node.extend(one_lsglobal_node)

            # Extend edge index, but increment the numbers
            lsedge_index.extend(
                [(edge[0] + counter, edge[1] + counter) for edge in one_lsedge_index]
            )

            # Extend lslshead node as well, but increment the numbers
            lslshead_node.append([node + counter for node in one_lshead_node])

            # Increment nodee counter
            counter += len(one_lsglobal_node)

        # Make a non sparse adjacency matrix

        return lsglobal_node, lsedge_index, lslshead_node


class GATForSeqClsf(GATModel):
    def __init__(self, config: GATForSeqClsfConfig, emb_init: Optional[Tensor]) -> None:
        super().__init__(config, emb_init=emb_init)
        nhid = config.nhid
        nclass = config.nclass
        feat_dropout_p = config.feat_dropout_p

        self.linear = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(p=feat_dropout_p)
        self.crs_entrpy = nn.CrossEntropyLoss()

    def prepare_batch_for_seq_clsf(
        self, batch: List[Tuple[List[Node], List[Edge], List[Node]]]
    ) -> Tuple[List[Node], List[Edge], List[List[Node]]]:
        """
         For each graph in batch
            connect each key node to a new, "CLS" node
            make that "CLS" node the only node in list of key nodes
        do super().prepare_batch()
        """

        # Connect all the "head nodes" to a new [CLS] node
        new_batch: List[Tuple[List[Node], List[Edge], List[Node]]] = []
        for one_lsglobal_node, one_lsedge_index, one_lshead_node in batch:
            new_one_lsglobal_node = one_lsglobal_node + [self.cls_id]

            new_cls_node = len(new_one_lsglobal_node) - 1
            head_to_cls_edges = [(node, new_cls_node) for node in one_lshead_node]
            new_one_lsedge_index = one_lsedge_index + to_undirected(head_to_cls_edges)

            new_one_lshead_node = [new_cls_node]

            new_batch.append(
                (new_one_lsglobal_node, new_one_lsedge_index, new_one_lshead_node)
            )

        return self.prepare_batch(new_batch)

    def forward(self, X: List[Tuple[List[Node], List[Edge], List[Node]]], y: Optional[List[int]]) -> Tuple[Tensor, ...]:  # type: ignore
        new_X = self.prepare_batch_for_seq_clsf(X)

        lsglobal_node, lsedge_index, lscls_node = new_X
        assert set(map(len, lscls_node)) == {1}

        # TODO: Maybe it will help with speed if you don't create a new tensor during every forward() call?
        word_ids = torch.tensor(lsglobal_node)

        N = len(lsglobal_node)
        adj: torch.Tensor = torch.zeros(N, N, dtype=torch.float)
        adj[list(zip(*lsedge_index))] = 1

        h = self.gat_layered(word_ids, adj)

        cls_id_h = h[:-1]

        cls_id_h = self.dropout(cls_id_h)
        logits = self.linear(cls_id_h)

        if y is not None:
            new_y = torch.tensor(y)
            loss = self.crs_entrpy(logits, new_y)
            return logits, loss
        else:
            return (logits,)
