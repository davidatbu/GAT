from typing import List
from typing import Tuple
from typing import TypeVar

import torch
import torch.nn as nn

from config import GATConfig
from layers import EmbeddingWrapper
from layers import GATLayerWrapper
from sent2graph import Edge
from sent2graph import Node


_T = TypeVar("_T")


class GATModel(nn.Module):  # type: ignore
    def __init__(self, config: GATConfig, emb_init: torch.Tensor):

        self.emb_wrapper = EmbeddingWrapper(config, emb_init)
        self.lsmid_layer_wrapper = nn.ModuleList(
            [GATLayerWrapper(config) for _ in range(config.nmid_layers)]
        )

    @staticmethod
    def prepare_batch(
        batch: List[Tuple[List[Node], List[Edge], List[Node], _T]]
    ) -> Tuple[List[Node], torch.Tensor, List[List[Node]], List[_T]]:

        lsglobal_node: List[Node] = []
        lsedge_index: List[Edge] = []
        lslshead_node: List[List[Node]] = []
        lslbl: List[_T] = []
        counter = 0
        for one_lsglobal_node, one_lsedge_index, one_lshead_node, one_lbl in batch:
            # Extend global node ids
            lsglobal_node.extend(one_lsglobal_node)

            # Extend edge index, but increment the numbers
            lsedge_index.extend(
                [(edge[0] + counter, edge[1] + counter) for edge in one_lsedge_index]
            )

            # Extend lslshead node as well, but increment the numbers
            lslshead_node.append([node + counter for node in one_lshead_node])

            # Append lbl
            lslbl.append(one_lbl)

            # Increment nodee counter
            counter += len(one_lsglobal_node)

        # Make a non sparse adjacency matrix
        N = len(lsglobal_node)
        tcadj: torch.Tensor = torch.zeros(N, N, dtype=torch.float)
        tcadj[list(zip(*lsedge_index))] = 1

        return lsglobal_node, tcadj, lslshead_node, lslbl
