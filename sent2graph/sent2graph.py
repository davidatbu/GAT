import logging
from typing import Dict
from typing import List
from typing import TypeVar

from utils import SentGraph


logging.basicConfig()
logger = logging.getLogger("__main__")


V = TypeVar("V")


class SentenceToGraph:
    id2edge_type: List[str]
    edge_type2id: Dict[str, int]

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        raise NotImplementedError()

    def batch_to_graph(self, lslsword: List[List[str]]) -> List[SentGraph]:
        return [self.to_graph(lsword) for lsword in lslsword]

    def to_graph(self, lsword: List[str]) -> SentGraph:
        raise NotImplementedError()

    def init_workers(self) -> None:
        pass

    def finish_workers(self) -> None:
        pass
