import logging
from multiprocessing import Process
from multiprocessing import Queue
from pprint import pformat
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import torch
from allennlp.predictors.predictor import Predictor
from typing_extensions import Literal

from ..utils.base import Edge
from ..utils.base import EdgeType
from ..utils.base import Node
from ..utils.base import SentGraph
from ..utils.base import Slice
from .base import SentenceToGraph

logger = logging.getLogger("__main__")

if TYPE_CHECKING:
    task_queue_t = Queue[Tuple[int, Union[Dict[str, Any], Literal["STOP"]]]]
    done_queue_t = Queue[Tuple[int, SentGraph]]
else:
    task_queue_t = done_queue_t = Queue

# I'm brekaing my own laws by using globals here. an FYI for future me.
_id2role: List[str] = [
    "ARG0",
    "ARG1",
    "ARG2",
    "ARG3",
    "ARG4",
    "ARG5",
    "ARGA",
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
    "ARGM-LVB",
    "ARGM-MNR",
    "ARGM-MOD",
    "ARGM-NEG",
    "ARGM-PNC",
    "ARGM-PRD",
    "ARGM-PRP",
    "ARGM-PRR",
    "ARGM-PRX",
    "ARGM-REC",
    "ARGM-TMP",
    "C-ARG0",
    "C-ARG1",
    "C-ARG2",
    "C-ARG3",
    "C-ARG4",
    "C-ARGM-ADJ",
    "C-ARGM-ADV",
    "C-ARGM-CAU",
    "C-ARGM-COM",
    "C-ARGM-DIR",
    "C-ARGM-DIS",
    "C-ARGM-DSP",
    "C-ARGM-EXT",
    "C-ARGM-LOC",
    "C-ARGM-MNR",
    "C-ARGM-MOD",
    "C-ARGM-NEG",
    "C-ARGM-PRP",
    "C-ARGM-TMP",
    "R-ARG0",
    "R-ARG1",
    "R-ARG2",
    "R-ARG3",
    "R-ARG4",
    "R-ARG5",
    "R-ARGM-ADV",
    "R-ARGM-CAU",
    "R-ARGM-COM",
    "R-ARGM-DIR",
    "R-ARGM-EXT",
    "R-ARGM-GOL",
    "R-ARGM-LOC",
    "R-ARGM-MNR",
    "R-ARGM-MOD",
    "R-ARGM-PNC",
    "R-ARGM-PRD",
    "R-ARGM-PRP",
    "R-ARGM-TMP",
    "V",
]
_role2id: Dict[str, int] = {role: i for i, role in enumerate(_id2role)}


class SRLSentenceToGraph(SentenceToGraph):
    @property
    def id2edge_type(self) -> List[str]:
        return _id2role

    @property
    def edge_type2id(self) -> Dict[str, int]:
        return _role2id

    def __init__(self, use_workers: bool = True) -> None:
        if torch.cuda.is_available():
            logger.info("Using CUDA for SRL")
            cuda_device = 0
        else:
            logger.warning("NOT USING CUDA FOR SRL")
            cuda_device = -1
        self.allen = Predictor.from_path(
            "/projectnb/llamagrp/davidat/pretrained_models/allenlp/bert-base-srl-2019.06.17.tar.gz",
            cuda_device=cuda_device,
        )

        self.use_workers = use_workers

    def init_workers(self) -> None:
        self.task_queue: task_queue_t = Queue()
        self.done_queue: done_queue_t = Queue()

        if self.use_workers:
            self.num_workers = 30
            for i in range(self.num_workers):
                Process(
                    target=_srl_resp_to_graph_worker,
                    args=(self.task_queue, self.done_queue),
                ).start()

    def __repr__(self) -> str:
        return "BSrl"

    def to_graph(self, lsword: List[str]) -> SentGraph:
        srl_resp = self.allen.predict(" ".join(lsword))
        return _srl_resp_to_graph(srl_resp)

    def batch_to_graph(self, lslsword: List[List[str]]) -> List[SentGraph]:
        req_json = [{"sentence": " ".join(lsword)} for lsword in lslsword]
        lssrl_resp = self.allen.predict_batch_json(req_json)
        if not self.use_workers:
            return [_srl_resp_to_graph(srl_resp) for srl_resp in lssrl_resp]
        else:
            assert self.done_queue.empty()
            for i, srl_resp in enumerate(lssrl_resp):
                self.task_queue.put((i, srl_resp))

            lsidx_sentgraph = [self.done_queue.get() for _ in range(len(lssrl_resp))]
            lsidx_sentgraph.sort(key=lambda tup: tup[0])
        _, lssentgraph = zip(*lsidx_sentgraph)
        return list(lssentgraph)

    def finish_workers(self) -> None:
        for i in range(self.num_workers):
            self.task_queue.put((999, "STOP"))

    def draw_graph(self, lsword: List[str]) -> None:
        import matplotlib.pyplot as plt
        import networkx as nx  # type: ignore

        lsedge, lsedge_type, lshead_node, _ = self.to_graph(lsword)
        lsnode = list(range(len(lsword)))
        node2label = {node: word for node, word in enumerate(lsword)}
        edge2label = {
            edge: self.id2edge_type[edge_type]
            for edge, edge_type in zip(lsedge, lsedge_type)
        }
        lsnode_color = ["b" if i in lshead_node else "r" for i in lsnode]

        G = nx.Graph()
        G.add_nodes_from(lsnode)
        G.add_edges_from(lsedge)

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


def _get_args_length(pred_and_args: Tuple[Node, List[Tuple[EdgeType, Slice]]]) -> int:
    if len(pred_and_args[1]) == 0:
        return 0
    _, all_slices = zip(*pred_and_args[1])
    all_slice_end_points: List[Node] = [i for slice_ in all_slices for i in slice_]
    return max(all_slice_end_points) - min(all_slice_end_points)


def _srl_resp_to_graph(srl_resp: Dict[str, Any]) -> SentGraph:
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

        # Typical need-one-check after the loop finish_workerses
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
                f"NO PREDICATE IN PARSE OF {srl_resp['words']}. Here is one of the returned structs: {srl_desc}"
            )
            continue

        # Make sure the predicate is one word
        pred_slice = role2slice.pop("V")
        if not pred_slice[1] - pred_slice[0] == 1:
            logger.warning(f"Whaaaat, propbank has multiword predicates? {srl_desc}")
            continue
        pred_node: Node = pred_slice[0]

        # Slightly restructure by converting roles to ids,
        # and placing the predicate as a node of it's own
        lsedge_type_and_arg: List[Tuple[EdgeType, Slice]] = []
        for role, arg_slice in role2slice.items():
            role_id = _role2id[role]
            lsedge_type_and_arg.append((role_id, arg_slice))
        pred_and_args: Tuple[Node, List[Tuple[EdgeType, Slice]]] = (
            pred_node,
            lsedge_type_and_arg,
        )
        lspred_and_args.append(pred_and_args)

    lsedge: List[Edge] = []
    lsedge_type: List[EdgeType] = []

    # Begin building graph from smallest predicate structure
    setnode: Set[Node] = set(range(len(srl_resp["words"])))
    lspred_and_args.sort(key=_get_args_length)
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

    # The nodeid2wordid is None
    return SentGraph(lsedge, lsedge_type, lshead_node, None)


def _srl_resp_to_graph_worker(
    task_queue: task_queue_t, done_queue: done_queue_t,
) -> None:
    for idx, srl_resp in iter(task_queue.get, None):
        if srl_resp == "STOP":
            break
        sentgraph = _srl_resp_to_graph(srl_resp)
        done_queue.put((idx, sentgraph))
