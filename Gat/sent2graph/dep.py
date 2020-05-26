from typing import Dict
from typing import List

import spacy
from spacy.tokens import Doc

from ..utils.base import EdgeList
from ..utils.base import EdgeTypeList
from ..utils.base import NodeList
from ..utils.base import SentGraph
from .base import SentenceToGraph

# https://spacy.io/api/annotation#dependency-parsing
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
    "cop",
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
    "nn",
    "nounmod",
    "npmod",
    "nsubj",
    "nsubjpass",
    "nummod",
    "oprd",
    "obj",
    "obl",
    "parataxis",
    "pcomp",
    "pobj",
    "poss",
    "preconj",
    "prep",
    "prt",
    "punct",
    "quantmod",
    "relcl",
    # "root",
    "xcomp",
    "",
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

        for token in doc:
            if token.dep_ == "ROOT":
                assert token.head == token
                lsimp_node.append(token.idx)
            else:
                lsedge.append((token.idx, token.head.idx))
                lsedge_type.append(token.dep_)

        return SentGraph(
            lsedge=lsedge,
            lsedge_type=lsedge_type,
            lsimp_node=lsimp_node,
            nodeid2wordid=None,
        )
