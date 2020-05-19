from itertools import zip_longest
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import TypeVar

Edge = Tuple[int, int]
EdgeList = List[Edge]
Node = int
NodeList = List[Node]
EdgeType = int
EdgeTypeList = List[EdgeType]
Slice = Tuple[int, int]


class SentGraph(NamedTuple):
    lsedge: EdgeList
    lsedge_type: EdgeTypeList
    lsimp_node: NodeList
    nodeid2wordid: Optional[List[int]]


class SentExample(NamedTuple):
    lssent: Tuple[str, ...]
    lbl: str


class SentgraphExample(NamedTuple):
    lssentgraph: Tuple[SentGraph, ...]
    lbl_id: int


def to_undirected(lsedge_index: List[Edge]) -> List[Edge]:
    # type ignore is cuz mypy can't figure out the length of a sorted list doesn't change
    directed_edge_index: List[Edge] = sorted(
        set([tuple(sorted(e)) for e in lsedge_index])  # type: ignore
    )
    undirected_edge_index = directed_edge_index + [
        (edge[1], edge[0]) for edge in directed_edge_index
    ]
    return undirected_edge_index


_T = TypeVar("_T")


def grouper(
    iterable: Iterable[_T], n: int, fillvalue: Optional[_T] = None
) -> Iterable[Tuple[Optional[_T], ...]]:
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
