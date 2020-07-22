import dataclasses
import typing as T
import unittest

from Gat.utils import Graph


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._node_namer = lambda x: f"modulo 10 was {x % 10}"
        self._edge_namer = lambda x: f"modulo 10 was {x % 10}"
        self._graph1 = Graph(
            lsedge=[(0, 1), (1, 2), (2, 3), (3, 0)],
            lsimp_node=[],
            lsedge_type=[0, 1, 2, 3],
            nodeid2wordid=[0, 1, 2, 3],
        )

    def test_equality(self) -> None:
        assert self._graph1.nodeid2wordid is not None
        with self.subTest("Just another copy"):
            self.assertTrue(
                self._graph1.equal_to(
                    dataclasses.replace(self._graph1),
                    self._node_namer,
                    self._edge_namer,
                )
            )
        with self.subTest("imp_node different"):
            graph2 = dataclasses.replace(self._graph1, lsimp_node=[0])
            self.assertFalse(
                self._graph1.equal_to(graph2, self._node_namer, self._edge_namer)
            )

        global_node_id_different = dataclasses.replace(
            self._graph1, nodeid2wordid=[10, 11, 12, 13]
        )
        with self.subTest("Node namer has no modulo"):
            self.assertFalse(
                self._graph1.equal_to(
                    global_node_id_different, lambda x: str(x), self._edge_namer
                )
            )
        with self.subTest("Node namer has modulo"):
            self.assertTrue(
                self._graph1.equal_to(
                    global_node_id_different, self._node_namer, self._edge_namer
                )
            )

            _T = T.TypeVar("_T")

        def reorder(ls: T.Sequence[_T], indices: T.List[int]) -> T.List[_T]:
            return [ls[i] for i in indices]

        indices = sorted(range(len(self._graph1.lsedge)), key=lambda x: x * 234 % 31)

        edge_order_different = dataclasses.replace(
            self._graph1,
            lsedge=reorder(self._graph1.lsedge, indices),
            lsedge_type=reorder(self._graph1.lsedge_type, indices),
        )
        with self.subTest("Edges reordered"):
            self.assertTrue(
                self._graph1.equal_to(
                    edge_order_different, self._node_namer, self._edge_namer
                )
            )
