from typing import List
from typing import Tuple

Edge = Tuple[int, int]
Node = int
EdgeType = int
Slice = Tuple[int, int]


def to_undirected(lsedge_index: List[Edge]) -> List[Edge]:
    # type ignore is cuz mypy can't figure out the length of a sorted list doesn't change
    directed_edge_index: List[Edge] = sorted(
        set([tuple(sorted(e)) for e in lsedge_index])  # type: ignore
    )
    undirected_edge_index = directed_edge_index + [
        (edge[1], edge[0]) for edge in directed_edge_index
    ]
    return undirected_edge_index
