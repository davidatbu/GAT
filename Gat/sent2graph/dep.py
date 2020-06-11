from typing import Dict
from typing import List

import spacy  # type: ignore
from spacy.tokens import Doc  # type: ignore

from ..utils.base import EdgeList
from ..utils.base import EdgeTypeList
from ..utils.base import NodeList
from ..utils.base import SentGraph
from .base import SentenceToGraph

# https://github.com/explosion/spacy-models/releases/tag/en_core_web_md-2.2.0
_id2edge_type = [
    "acl",
    "acomp",
    "advcl",
    "advmod",
    "agent",
    "amod",
    "appos",
    "attr",
    "aux",
    "auxpass",
    "case",
    "cc",
    "ccomp",
    "compound",
    "conj",
    "csubj",
    "csubjpass",
    "dative",
    "dep",
    "det",
    "dobj",
    "expl",
    "intj",
    "mark",
    "meta",
    "neg",
    "nmod",
    "npadvmod",
    "nsubj",
    "nsubjpass",
    "nummod",
    "oprd",
    "parataxis",
    "pcomp",
    "pobj",
    "poss",
    "preconj",
    "predet",
    "prep",
    "prt",
    "punct",
    "quantmod",
    "relcl",
    "xcomp",
]

_edge_type2id = {edge_type: id_ for id_, edge_type in enumerate(_id2edge_type)}


class DepSentenceToGraph(SentenceToGraph):
    def __init__(self, spacy_mdl: str = "en_core_web_md") -> None:
        self._spacy_mdl = spacy_mdl
        self._nlp = spacy.load(self._spacy_mdl, disable=["tagger", "ner"])

    def __repr__(self) -> str:
        return f"DepS2G_{self._spacy_mdl}"

    @property
    def edge_type2id(self) -> Dict[str, int]:
        return _edge_type2id

    @property
    def id2edge_type(self) -> List[str]:
        return _id2edge_type

    def to_graph(self, lsword: List[str]) -> SentGraph:
        sent = " ".join(lsword)
        doc: Doc = self._nlp(sent)

        lsedge: EdgeList = []
        lsedge_type: EdgeTypeList = []
        lsimp_node: NodeList = []

        for i, token in enumerate(doc):
            assert i == token.i
            if token.dep_ == "ROOT":
                assert token.head == token
                lsimp_node.append(token.i)
            else:
                lsedge.append((token.i, token.head.i))
                lsedge_type.append(self.edge_type2id[token.dep_])

        assert lsimp_node != []

        return SentGraph(
            lsedge=lsedge,
            lsedge_type=lsedge_type,
            lsimp_node=lsimp_node,
            nodeid2wordid=None,
        )
