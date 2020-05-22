def networkx() -> None:
    import networkx as nx  # type: ignore

    g = nx.DiGraph()
    g.add_nodes_from([(1, {"label": "hi"}), (2, {"label": "my"})])
    g.add_edges_from(
        [
            (1, 2),
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
    g.nodes()
    p = nx.drawing.nx_pydot.to_pydot(g)
    p.write_png("example.png")


if __name__ == "__main__":
    networkx()
