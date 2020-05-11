import logging
import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TypeVar

import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
from allennlp.predictors.predictor import Predictor

GLOBAL_DEBUG_FLAG = False


class FakeAllen(Predictor):
    def __init__(self) -> None:
        pass

    def predict(self, sent: str) -> Dict[str, Any]:
        return {
            "verbs": [
                {
                    "verb": "do",
                    "description": "[V: do] not love the world",
                    "tags": ["B-V", "O", "O", "O", "O"],
                },
                {
                    "verb": "love",
                    "description": "do [ARGM-NEG: not] [V: love] [ARG1: the world]",
                    "tags": ["O", "B-ARGM-NEG", "B-V", "B-ARG1", "I-ARG1"],
                },
            ],
            "words": ["do", "not", "love", "the", "world"],
        }


from allennlp.predictors.predictor import Predictor
from typing_extensions import Literal


logging.basicConfig()
logger = logging.getLogger("__main__")
logger.setLevel(logging.DEBUG)


V = TypeVar("V")


class PerSplit(Dict[Literal["train", "val", "test"], V]):
    def __setitem__(self, k: Literal["train", "val", "test"], v: V) -> None:
        if k not in ["train", "val", "test"]:
            raise Exception("Nope.")
        super().__setitem__(k, v)


Edge = Tuple[int, int]
Node = int
EdgeType = int
Slice = Tuple[int, int]


class SentenceToGraph:
    def __init__(self, cache_dir: Path, ignore_cache: bool) -> None:
        self.cache_fp = cache_dir / (str(self) + ".pkl")
        self.cache: Dict[Tuple[str, ...], Tuple[List[int], List[Edge], List[int]]] = {}

        if not ignore_cache:

            if not self.cache_fp.exists():
                logger.warning(
                    f"{str(self.cache_fp)} does not exist, but asked to read from cache. Ignoring."
                )
            else:
                with self.cache_fp.open("rb") as fb:
                    pkl_content = pkl.load(fb)
                    assert isinstance(pkl_content, dict)
                    self.cache = pkl_content

    def __repr__(self) -> str:
        raise NotImplementedError()

    def to_graph(
        self, lsword: Tuple[str, ...]
    ) -> Tuple[List[int], List[Edge], List[int]]:
        if lsword in self.cache:
            return self.cache[lsword]
        else:
            graph = self._to_graph(lsword)
            self.cache[lsword] = graph
            return self.cache[lsword]

    def batch_to_graph(
        self, lslsword: List[Tuple[str, ...]]
    ) -> List[Tuple[List[int], List[Edge], List[int]]]:
        return [self.to_graph(lsword) for lsword in lslsword]

    def _to_graph(
        self, lsword: Tuple[str, ...]
    ) -> Tuple[List[int], List[Edge], List[int]]:

        raise NotImplementedError()

    def save_cache(self) -> None:
        with self.cache_fp.open("wb") as fb:
            pkl.dump(self.cache, fb)


_id2role: List[str] = [
    "ARG0",
    "ARG1",
    "ARG2",
    "ARG3",
    "ARG4",
    "ARGM-ADJ",
    "ARGM-ADV",
    "ARGM-CAU",
    "ARGM-COM",
    "ARGM-DIR",
    "ARGM-DIS",
    "ARGM-DSP",
    "ARGM-EXT",
    "ARGM-GOL",
    "ARGM-LOC",
    "ARGM-MNR",
    "ARGM-MOD",
    "ARGM-NEG",
    "ARGM-PRD",
    "ARGM-PRP",
    "ARGM-PRR",
    "ARGM-REC",
    "ARGM-TMP",
    "ARGA",
    "LINK-PRO",
    "LINK-PSV",
    "LINK-SLC",
]
_role2id: Dict[str, int] = {role: i for i, role in enumerate(_id2role)}


class SRLSentenceToGraph(SentenceToGraph):
    _id2role = _id2role
    _role2id = _role2id

    def __init__(
        self, cache_dir: Path, ignore_cache: bool, use_cache_only: bool
    ) -> None:
        assert not (ignore_cache and use_cache_only)
        self.use_cache_only = use_cache_only

        super().__init__(cache_dir, ignore_cache)
        self.allen: Optional[Predictor] = None
        if not use_cache_only:
            if GLOBAL_DEBUG_FLAG:
                self.allen = FakeAllen()
            else:
                self.allen = Predictor.from_path(
                    "/projectnb/llamagrp/davidat/pretrained_models/allenlp/bert-base-srl-2019.06.17.tar.gz"
                )

            # self.allen = FakeAllen()

    def __repr__(self) -> str:
        return "BERT_SRL_SEN2GRAPH"

    def _to_graph(
        self, lsword: Tuple[str, ...]
    ) -> Tuple[List[int], List[Edge], List[int]]:
        if self.use_cache_only:
            raise Exception(
                f"self.use_cache_only=True, but {lsword[:10]}... not found in cache"
            )

        assert self.allen is not None
        srl_resp = self.allen.predict(" ".join(lsword))
        logger.debug(pformat(srl_resp, indent=2))
        # Sample response
        #         # Assert we had the same tokenization

        # To avoid "connections that skip levels", we need this
        # TTO get the "head" nodes, gotta keep track of which words are not head
        lspred_and_args: List[Tuple[Node, List[Tuple[EdgeType, Slice]]]] = []
        for srl_desc in srl_resp["verbs"]:

            cur_role: Optional[str] = None
            cur_beg: Optional[int] = None

            role2slice: Dict[str, Tuple[int, int]] = {}
            for i, tag in enumerate(srl_desc["tags"]):
                if tag[0] in ["O", "B"]:
                    if cur_beg is not None:  # We *just* ended a role
                        cur_end = i
                        cur_slice = (cur_beg, cur_end)
                        assert cur_role is not None
                        role2slice[cur_role] = cur_slice
                    cur_role = None
                    cur_beg = None

                if tag[0] == "B":  # We are beginning a role
                    cur_role = tag[2:]
                    cur_beg = i

            # Typical need-one-check after the loop finishes
            if cur_beg is not None:
                cur_end = len(srl_desc["tags"])
                assert cur_role is not None
                cur_slice = (cur_beg, cur_end)
                assert cur_role is not None
                role2slice[cur_role] = cur_slice

            # Check that the "predicate" is present, sometimes, allen dones't give me the predicate
            # predicate is marked with a "V" "role"
            if not "V" in role2slice:
                logger.warning(
                    f"NO PREDICATE IN PARSE OF {lsword}. Here is one of the returned structs: {srl_desc}"
                )
                continue

            # Make sure the predicate is one word
            pred_slice = role2slice.pop("V")
            if not pred_slice[1] - pred_slice[0] == 1:
                raise Exception("whaaaat, propbank has multiword predicates?")
            pred_node: Node = pred_slice[0]

            # Slightly restructure by converting roles to ids,
            # and placing the predicate as a node of it's own
            lsedge_type_and_arg: List[Tuple[EdgeType, Slice]] = []
            for role, arg_slice in role2slice.items():
                role_id = self._role2id[role]
                lsedge_type_and_arg.append((role_id, arg_slice))
            pred_and_args: Tuple[Node, List[Tuple[EdgeType, Slice]]] = (
                pred_node,
                lsedge_type_and_arg,
            )
            lspred_and_args.append(pred_and_args)

        lsedge: List[Edge] = []
        lsedge_type: List[EdgeType] = []

        # Begin building graph from smallest predicate structure
        setnode: Set[Node] = set(range(len(lsword)))
        lspred_and_args.sort(key=self._get_args_length)
        for pred_and_args in lspred_and_args:
            pred_node = pred_and_args[0]
            for edge_type, arg_slice in pred_and_args[1]:
                for arg_node in range(*arg_slice):
                    if arg_node in setnode:
                        lsedge.append((arg_node, pred_node))
                        lsedge_type.append(edge_type)
                        setnode.remove(arg_node)

        # Nodes that had no "parent" argument
        lshead_node = list(sorted(setnode))

        return (lshead_node, lsedge, lsedge_type)

    @staticmethod
    def _get_args_length(
        pred_and_args: Tuple[Node, List[Tuple[EdgeType, Slice]]]
    ) -> int:
        if len(pred_and_args[1]) == 0:
            return 0
        _, all_slices = zip(*pred_and_args[1])
        all_slice_end_points: List[Node] = [i for slice_ in all_slices for i in slice_]
        return max(all_slice_end_points) - min(all_slice_end_points)

    def draw_graph(self, lsword: Tuple[str, ...]) -> None:

        lshead_node, lsedge_index, lsrole_id = self.to_graph(lsword)
        lsnode = list(range(len(lsword)))
        node2label = {node: word for node, word in enumerate(lsword)}
        edge2label = {
            edge: self._id2role[role_id]
            for edge, role_id in zip(lsedge_index, lsrole_id)
        }
        lsnode_color = ["b" if i in lshead_node else "r" for i in lsnode]

        G = nx.Graph()
        G.add_nodes_from(lsnode)
        G.add_edges_from(lsedge_index)

        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos=pos,
            labels=node2label,
            node_color=lsnode_color,
            node_size=1000,
            size=10000,
        )
        nx.draw_networkx_edge_labels(
            G, pos=pos, edge_labels=edge2label,
        )
        plt.show()


def _test() -> None:
    dataset_dir = Path(
        "/projectnb/llamagrp/davidat/projects/graphs/data/ready/gv_2018_1160_examples"
    )
    sent2graph = SRLSentenceToGraph(
        cache_dir=dataset_dir, ignore_cache=False, use_cache_only=False
    )
    for lsword in [["do", "not", "love", "the", "world"]]:
        sent2graph.draw_graph(tuple(lsword))


def main() -> None:
    global GLOBAL_DEBUG_FLAG
    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    GLOBAL_DEBUG_FLAG = args.debug
    _test()


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    main()
