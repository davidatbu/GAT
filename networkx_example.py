from bs4 import BeautifulSoup as BS  # type: ignore


def networkx() -> None:
    import networkx as nx  # type: ignore

    g = nx.DiGraph()
    g.add_nodes_from([(1, {"label": "a"}), (2, {"label": "my"})])
    g.add_edges_from(
        [
            (1, 2, {"label": "ADV"}),
            (1, 3),
            (1, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (3, 8),
            (3, 9),
            (4, 10),
            (5, 11),
            (5, 12),
            (6, 13),
        ]
    )
    p = nx.drawing.nx_pydot.to_pydot(g)
    svg = p.create_svg().decode()
    root_sp = BS("<table></table>", "lxml")
    svg_doc_sp = BS(svg)
    svg_sp = svg_doc_sp.find("svg")
    root_sp.append(svg_sp)
    print(root_sp)
    print(type(root_sp))


if __name__ == "__main__":
    networkx()
