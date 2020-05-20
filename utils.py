from itertools import zip_longest
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
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
    lssent: List[str]
    lbl: str


class SentgraphExample(NamedTuple):
    lssentgraph: List[SentGraph]
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


def sorted_directed(lsedge: List[Edge]) -> List[Edge]:
    dict_edge: Dict[Node, Node] = {}
    for node1, node2 in lsedge:
        if node1 > node2:
            node2, node1 = node1, node2
        dict_edge[node1] = node2
    return list(dict_edge.items())


_T = TypeVar("_T")


def grouper(iterable: Iterable[_T], n: int) -> Iterator[List[_T]]:

    cur_batch = []
    for i, item in enumerate(iter(iterable), start=1):
        cur_batch.append(item)
        if i % n == 0:
            yield cur_batch
            cur_batch = []


def is_seq(item: Any) -> bool:
    if isinstance(item, (list, tuple)):
        return True
    return False


def flatten(ls: Iterable[Any]) -> Iterator[Any]:
    for i in ls:
        if is_seq(i):
            for j in flatten(i):
                yield j
        else:
            yield i


def reshape_like(to_reshape: Iterable[Any], model: Any) -> Tuple[Any, int]:
    flat = flatten(to_reshape)
    return _reshape_like(flat, model)


def _reshape_like(flat: Iterator[Any], model: Any) -> Tuple[Any, int]:

    consumed = 0
    reshaped = []

    for child in model:
        if is_seq(child):
            child_reshaped, child_consumed = reshape_like(flat, child)
            consumed += child_consumed
        else:
            child_reshaped = next(flat)
            consumed += 1
        reshaped.append(child_reshaped)

    return reshaped, consumed
