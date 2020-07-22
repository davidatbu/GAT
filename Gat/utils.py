"""Some datastructures, some convinience functions, can probably be broken up."""
from __future__ import annotations

import base64
import typing as T

import lazy_import

if T.TYPE_CHECKING:
    import networkx as nx  # type: ignore
else:
    nx = lazy_import.lazy_module("networkx")
import numpy as np  # type: ignore
import plotly.figure_factory as ff  # type: ignore
from bs4 import BeautifulSoup as BS  # type: ignore
from bs4 import NavigableString
from dataclasses import dataclass

Edge = T.Tuple[int, int]
EdgeList = T.List[Edge]
Node = int
NodeList = T.List[Node]
EdgeType = int
EdgeTypeList = T.List[EdgeType]
Slice = T.Tuple[int, int]
NodeName = T.TypeVar("NodeName")
EdgeName = T.TypeVar("EdgeName")
NamedEdgesWithTypes = T.FrozenSet[T.Tuple[T.Tuple[NodeName, NodeName], EdgeName]]
NamedImpNodes = T.FrozenSet[NodeName]


class Graph:
    """A representation of a graph, amenable to representing a graph of words.

    Attributes:
        lsedge: A list of tuples. `(n1,n2)` means there is an edge between 
            nodeid2wordid[n1] and nodeid2wordid[n2].
        lsedge_type: A list of edge types.
        lsimp_node: A list of "important nodes". `[n1]` means nodeid2wordid[n1] is an
            important node.
        nodeid2wordid: A mapping from the zero based node indices used in above to a
            global "vocabulary" of some kind.
    """

    __slots__ = (
        "lsedge",
        "lsedge_type",
        "lsimp_node",
        "nodeid2wordid",
    )

    def __init__(
        self,
        lsedge: EdgeList,
        lsedge_type: EdgeTypeList,
        lsimp_node: NodeList,
        nodeid2wordid: T.Optional[T.List[int]],
    ) -> None:

        self.lsedge = lsedge
        self.lsedge_type = lsedge_type
        self.lsimp_node = lsimp_node
        self.nodeid2wordid = nodeid2wordid

    def copy(
        self,
        *,
        lsedge: T.Optional[EdgeList] = None,
        lsedge_type: T.Optional[EdgeTypeList] = None,
        lsimp_node: T.Optional[NodeList] = None,
        nodeid2wordid: T.Optional[T.List[Node]] = None,
    ) -> Graph:

        graph_class = type(self)
        return graph_class(
            lsedge=lsedge if lsedge is not None else self.lsedge,
            lsedge_type=lsedge_type if lsedge_type is not None else self.lsedge_type,
            lsimp_node=lsimp_node if lsimp_node is not None else self.lsimp_node,
            nodeid2wordid=nodeid2wordid
            if nodeid2wordid is not None
            else self.nodeid2wordid,
        )

    def __hash__(self) -> int:
        """Needed to use as a key in lru_cache."""
        nodeid2wordid = self.nodeid2wordid
        if nodeid2wordid is None:
            nodeid2wordid = []
        to_hash: T.List[T.List[T.Any]] = [
            self.lsedge,
            self.lsedge_type,
            self.lsimp_node,
            nodeid2wordid,
        ]

        return hash(tuple(tuple(ls) for ls in to_hash))

    def _named_edges_and_imp_nodes(
        self,
        node_namer: T.Callable[[int], NodeName],
        edge_namer: T.Callable[[int], EdgeName],
    ) -> T.Tuple[NamedEdgesWithTypes[NodeName, EdgeName], NamedImpNodes[NodeName]]:
        """"
        
        Returns:
            named_edges_with_types:
            named_imp_nodes:
        """
        assert self.nodeid2wordid
        lsnode_named = list(map(node_namer, self.nodeid2wordid))
        lsedge_named = [(lsnode_named[e1], lsnode_named[e2]) for e1, e2 in self.lsedge]
        lsimp_node_named = frozenset(lsnode_named[n] for n in self.lsimp_node)
        lsedge_type_named = list(map(edge_namer, self.lsedge_type))
        lsedge_with_edge_type_named = frozenset(zip(lsedge_named, lsedge_type_named))

        return lsedge_with_edge_type_named, lsimp_node_named

    def equal_to(
        self,
        other: Graph,
        node_namer: T.Callable[[int], NodeName],
        edge_namer: T.Callable[[int], EdgeName],
    ) -> bool:
        """Compares two graphs and checks equality.

        Args:
            node_namer: the global node ids usually correspond to something like a 
                word token. We dont want to compare based on the global node id, 
                but the token itself.
            edge_namer: Samet thing as node_namer.
        """

        assert (
            self.nodeid2wordid is not None and other.nodeid2wordid is not None
        ), "Need global node ids to determine graph equality"

        self_named = self._named_edges_and_imp_nodes(node_namer, edge_namer)
        other_named = other._named_edges_and_imp_nodes(node_namer, edge_namer)
        return self_named == other_named

    def to_svg(
        self,
        node_namer: T.Callable[[int], str] = lambda i: str(i),
        edge_namer: T.Callable[[int], str] = lambda i: str(i),
    ) -> str:
        """Draw an SVG image using networkx.

        Args:
            node_namer: Turn the global node ids to human readable names.
            edge_namer: Same as above, but for the edges.

        Returns:
            svg_str: An SVG string.
        """

        g = nx.DiGraph()

        def quote(s: str) -> str:
            """Because of a PyDot bug, we need this."""
            return '"' + s.replace('"', '"') + '"'

        assert self.nodeid2wordid is not None

        # NetworkX format
        lsnode_id_and_nx_dict: T.List[T.Tuple[int, T.Dict[str, str]]] = [
            (node_id, {"label": quote(name)})
            for node_id, name in enumerate(map(node_namer, self.nodeid2wordid))
        ]

        # Mark the "important nodes"
        print("about to check for head nodes.")
        for node_id, nx_dict in lsnode_id_and_nx_dict:
            if node_id in self.lsimp_node:
                print("found head node.")
                nx_dict["label"] += ": IMP node"

        # Edges in nx format
        lsedge_name: T.List[T.Tuple[int, int, T.Dict[str, str]]] = [
            (n1, n2, {"label": quote(edge_namer(edge_id))})
            for (n1, n2), edge_id in zip(self.lsedge, self.lsedge_type)
        ]
        g.add_nodes_from(lsnode_id_and_nx_dict)
        g.add_edges_from(lsedge_name)
        p = nx.drawing.nx_pydot.to_pydot(g)
        return p.create_svg().decode()  # type: ignore


class SentExample(T.NamedTuple):
    """A list of sentences and a label."""

    lssent: T.List[str]
    lbl: str


class GraphExample(T.NamedTuple):
    """A list of graphs and a label."""

    lsgraph: T.List[Graph]
    lbl_id: int


def sorted_directed(lsedge: T.List[Edge]) -> T.List[Edge]:
    dict_edge: T.Dict[Node, Node] = {}
    for node1, node2 in lsedge:
        if node1 > node2:
            node2, node1 = node1, node2
        dict_edge[node1] = node2
    return list(dict_edge.items())


_T = T.TypeVar("_T")


def grouper(iterable: T.Iterable[_T], n: int) -> T.Iterator[T.List[_T]]:

    cur_batch = []
    for i, item in enumerate(iter(iterable), start=1):
        cur_batch.append(item)
        if i % n == 0:
            yield cur_batch
            cur_batch = []


def is_seq(item: T.Any) -> bool:
    if isinstance(item, (list, tuple)):
        return True
    return False


def flatten(ls: T.Iterable[T.Any]) -> T.Iterator[T.Any]:
    for i in ls:
        if is_seq(i):
            for j in flatten(i):
                yield j
        else:
            yield i


def reshape_like(to_reshape: T.Iterable[T.Any], model: T.Any) -> T.Tuple[T.Any, int]:
    flat = flatten(to_reshape)
    return _reshape_like(flat, model)


def _reshape_like(flat: T.Iterator[T.Any], model: T.Any) -> T.Tuple[T.Any, int]:

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


class Cell:
    def __init__(self) -> None:
        pass

    def sp(self, root_sp: BS) -> BS:
        raise NotImplementedError()


class TextCell(Cell):
    def __init__(self, content: str):
        self._content = content

    def sp(self, root_sp: BS) -> BS:
        return NavigableString(self._content)


class NumCell(Cell):
    def __init__(self, content: T.Union[int, float]):
        self._content = content

    def sp(self, root_sp: BS) -> BS:
        return NavigableString(str(self._content))


class PngCell(Cell):
    def __init__(self, content: bytes):
        self._content = content

    def sp(self, root_sp: BS) -> BS:
        encoded = base64.encodebytes(self._content).decode()
        img_sp = root_sp.new_tag(
            "img", src=f"data:image/png;base64,{encoded}", width="900px"
        )
        return img_sp


class SvgCell(Cell):
    def __init__(self, content: str):
        self._content = content

    def sp(self, root_sp: BS) -> BS:
        svg_doc_sp = BS(self._content)
        svg_sp = svg_doc_sp.find("svg")
        svg_sp["style"] = "width: 700px"
        for attr in ["width", "height"]:
            if attr in svg_sp.attrs:
                del svg_sp.attrs[attr]
        return svg_sp


def html_table(
    rows: T.List[T.Tuple[Cell, ...]],
    headers: T.Tuple[Cell, ...],
    row_colors: T.List[T.Optional[str]] = [],
) -> str:
    root_sp: BS = BS("<html><table></table></html>", "lxml")
    table_sp: BS = root_sp.find("table")

    header_row_sp = root_sp.new_tag("tr")
    table_sp.append(header_row_sp)

    for hdr_cell in headers:
        th_sp = root_sp.new_tag("th")
        th_sp.append(hdr_cell.sp(root_sp))
        header_row_sp.append(th_sp)

    if not row_colors:
        row_colors = [None] * len(rows)
    for color, row in zip(row_colors, rows):
        if color:
            attrs: T.Dict[str, str] = {"style": f"color: {color}"}
        else:
            attrs = {}

        row_sp = root_sp.new_tag("tr", attrs=attrs)
        table_sp.append(row_sp)
        for cell in row:
            td_sp = root_sp.new_tag("td")
            td_sp.append(cell.sp(root_sp))
            row_sp.append(td_sp)

    return str(table_sp)


def plotly_cm(
    cm: np.ndarray, labels: T.List[str], title: str = "Confusion matrix"
) -> T.Any:

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
