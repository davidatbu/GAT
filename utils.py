from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import plotly.figure_factory as ff  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
from bs4 import NavigableString

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


CellContent = Union[str, int, float]


def html_table(
    rows: List[Tuple[CellContent, ...]],
    headers: Tuple[CellContent, ...],
    row_colors: List[Optional[str]] = [],
) -> str:
    head_sp: BeautifulSoup = BeautifulSoup("<html><table></table></html>", "lxml")
    table_sp: BeautifulSoup = head_sp.find("table")

    def append(
        sp: BeautifulSoup,
        tag: Optional[str] = None,
        content: Optional[CellContent] = None,
        style: Optional[str] = None,
    ) -> BeautifulSoup:

        if style is None:
            attrs: Dict[str, str] = {}
        else:
            attrs = {"style": style}

        if tag is None:
            new_sp = NavigableString(str(content))
        else:
            new_sp = head_sp.new_tag(tag, **attrs)
            if content is not None:
                new_sp.append(str(content))

        sp.append(new_sp)
        return new_sp

    header_row_sp = append(table_sp, tag="tr")
    for hdr_cell in headers:
        append(header_row_sp, tag="th", content=hdr_cell)

    if not row_colors:
        row_colors = [None] * len(rows)
    for color, row in zip(row_colors, rows):
        if color:
            style: Optional[str] = f"color: {color}"
        else:
            style = None
        row_sp = append(table_sp, tag="tr", style=style)
        for cell in row:
            append(row_sp, tag="td", content=cell)

    return str(table_sp)


def plotly_cm(
    cm: np.ndarray, labels: List[str], title: str = "Confusion matrix"
) -> Any:

    # change each element of z to type string for annotations
    scaled = cm * 100 / cm.sum()
    z_text = [[str(y) for y in x] for x in scaled]

    # set up figure
    fig = ff.create_annotated_heatmap(
        cm, x=labels, y=labels, annotation_text=z_text, colorscale="Viridis"
    )

    # add title
    fig.update_layout(title_text=f"<i><b>{title}</b></i>",)

    # add custom xaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Predicted value",
            xref="paper",
            yref="paper",
        )
    )

    # add custom yaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=-0.35,
            y=0.5,
            showarrow=False,
            text="Real value",
            textangle=-90,
            xref="paper",
            yref="paper",
        )
    )

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig["data"][0]["showscale"] = True
    return fig
