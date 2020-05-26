import logging
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import TypeVar

from ..utils.base import SentGraph


logging.basicConfig()
logger = logging.getLogger("__main__")


V = TypeVar("V")


class SentenceToGraph(ABC):
    @property
    @abstractmethod
    def id2edge_type(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def edge_type2id(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def batch_to_graph(self, lslsword: List[List[str]]) -> List[SentGraph]:
        return [self.to_graph(lsword) for lsword in lslsword]

    @abstractmethod
    def to_graph(self, lsword: List[str]) -> SentGraph:
        pass

    def init_workers(self) -> None:
        pass

    def finish_workers(self) -> None:
        pass
