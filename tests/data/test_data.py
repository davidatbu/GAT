"""Tests for base.py."""
import unittest
from pathlib import Path

from Gat import data
from Gat.data import datasets


class TestDatasets(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        datasets_per_split, _, _ = data.utils.load_splits(
            # Path("data/glue_data/SST-2"),
            Path("actual_data/SST-2_tiny"),
            splits=["train", "dev"],
            lstxt_col=["sentence"],
            sent2graph_name="dep",
            unk_thres=1,
        )

        self._dataset = datasets_per_split["train"]

    def tearDown(self) -> None:
        super().tearDown()

    def test_draw_svg(self) -> None:
        example = self._dataset[0]

        graph = example.lsgraph[0]

        svg_content = graph.to_svg(
            node_namer=lambda node_id: self._dataset.vocab.get_tok(node_id),
            edge_namer=lambda edge_id: self._dataset.id2edge_type[edge_id],
        )

        with open("graph.svg", "w") as f:
            f.write(svg_content)

    def test_connect_to_cls(self) -> None:

        dataset = datasets.ConnectToClsDataset(self._dataset)

        graph = dataset[0].lsgraph[0]
        svg_content = graph.to_svg(
            node_namer=lambda node_id: self._dataset.vocab.get_tok(node_id),
            edge_namer=lambda edge_id: dataset.id2edge_type[edge_id],
        )
        with open("graph_with_cls.svg", "w") as f:
            f.write(svg_content)

    def test_undirected(self) -> None:

        dataset = datasets.UndirectedDataset(self._dataset)

        graph = dataset[0].lsgraph[0]
        svg_content = graph.to_svg(
            node_namer=lambda node_id: self._dataset.vocab.get_tok(node_id),
            edge_namer=lambda edge_id: dataset.id2edge_type[edge_id],
        )
        with open("graph_undirected.svg", "w") as f:
            f.write(svg_content)

    def test_undirected_connect_to_cls(self) -> None:

        dataset = datasets.UndirectedDataset(
            datasets.ConnectToClsDataset(self._dataset)
        )

        graph = dataset[0].lsgraph[0]
        svg_content = graph.to_svg(
            node_namer=lambda node_id: self._dataset.vocab.get_tok(node_id),
            edge_namer=lambda edge_id: dataset.id2edge_type[edge_id],
        )
        with open("graph_with_cls_undirected.svg", "w") as f:
            f.write(svg_content)
