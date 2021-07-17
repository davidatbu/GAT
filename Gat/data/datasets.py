import abc
import logging
import typing as T
from pathlib import Path

from Gat.data import numerizer, sent2graphs, vocabs
from Gat.data.cacheable import Cacheable, TorchCachingTool
from Gat.data.sources import TextSource
from Gat.utils import Graph, GraphExample, SentExample, grouper
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

_T = T.TypeVar("_T")
_S = T.TypeVar("_S")
_D = T.TypeVar("_D")


class NiceDataset(Dataset, T.Iterable[_T], abc.ABC):
    """Inherits from T.Generic, adds __len__ and __iter__"""

    @abc.abstractmethod
    def __getitem__(self, i: int) -> _T:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self,) -> T.Iterator[_T]:
        for i in range(len(self)):
            yield self[i]


class TransformedDataset(NiceDataset[_S], T.Generic[_T, _S]):
    @abc.abstractmethod
    def _transform(self, example: _T) -> _S:
        pass

    @abc.abstractproperty
    def base_dataset(self) -> NiceDataset[_T]:
        pass

    def __getitem__(self, i: int) -> _S:
        return self._transform(self.base_dataset[i])

    def __len__(self) -> int:
        return len(self.base_dataset)


class CutDataset(NiceDataset[_T]):
    def __init__(self, base_dataset: NiceDataset[_T], total_len: int) -> None:
        """
        Args:
            total_len: length of cut dataset.
        """
        if total_len < 1:
            raise Exception()
        super().__init__()
        self._total_len = total_len
        self._base_dataset = base_dataset

    def __getitem__(self, i: int) -> _T:
        if i < self._total_len:
            return self._base_dataset[i]
        else:
            raise IndexError()

    def __len__(self) -> int:
        return self._total_len


_NumerizerSub = T.TypeVar("_NumerizerSub", bound=numerizer.Numerizer, covariant=True)


class BaseGraphDataset(NiceDataset[GraphExample], T.Generic[_NumerizerSub]):
    @abc.abstractproperty
    def sent2graph(self) -> sent2graphs.SentenceToGraph:
        pass

    @abc.abstractproperty
    def numerizer(self) -> _NumerizerSub:
        pass

    @abc.abstractproperty
    def id2edge_type(self) -> T.List[str]:
        pass

    @abc.abstractproperty
    def edge_type2id(self) -> T.Dict[str, int]:
        pass


_VocabSub = T.TypeVar("_VocabSub", bound=vocabs.Vocab, covariant=True)


class BaseSentenceToGraphDataset(BaseGraphDataset[_VocabSub]):
    @abc.abstractproperty
    def sent2graph(self) -> sent2graphs.SentenceToGraph:
        pass

    @abc.abstractproperty
    def numerizer(self) -> _VocabSub:
        pass

    @abc.abstractproperty
    def id2edge_type(self) -> T.List[str]:
        pass

    @abc.abstractproperty
    def edge_type2id(self) -> T.Dict[str, int]:
        pass


class UndirectedDataset(
    TransformedDataset[GraphExample, GraphExample],
    BaseSentenceToGraphDataset[_VocabSub],
):
    _REVERSED = "_REVERSED"

    def __init__(self, base_dataset: BaseSentenceToGraphDataset[_VocabSub]):
        self._base_dataset = base_dataset
        self._id2edge_type = self._base_dataset.id2edge_type + [
            edge_type + self._REVERSED for edge_type in base_dataset.id2edge_type
        ]
        self._edge_type2id = {
            edge_type: i for i, edge_type in enumerate(self._id2edge_type)
        }

    @property
    def base_dataset(self) -> BaseSentenceToGraphDataset[_VocabSub]:
        return self._base_dataset

    def _transform(self, example: GraphExample) -> GraphExample:
        new_example = GraphExample(
            [self._graph_to_undirected(g) for g in example.lsgraph], example.lbl_id,
        )
        return new_example

    def _graph_to_undirected(self, g: Graph) -> Graph:
        new_graph = Graph(
            lsedge=g.lsedge + [(n2, n1) for n1, n2 in g.lsedge],
            lsedge_type=g.lsedge_type
            + [
                self.edge_type2id[self.id2edge_type[edge_id] + self._REVERSED]
                for edge_id in g.lsedge_type
            ],
            lsimp_node=g.lsimp_node,
            nodeid2wordid=g.lsglobal_id,
        )
        return new_graph

    @property
    def sent2graph(self) -> sent2graphs.SentenceToGraph:
        # Technically not true, since we modify the graph in this class after getting
        # it from this sent2graph
        return self._base_dataset.sent2graph

    @property
    def numerizer(self) -> _VocabSub:
        return self._base_dataset.numerizer

    @property
    def id2edge_type(self) -> T.List[str]:
        return self._id2edge_type

    @property
    def edge_type2id(self) -> T.Dict[str, int]:
        return self._edge_type2id


class ConnectToClsDataset(
    TransformedDataset[GraphExample, GraphExample],
    BaseSentenceToGraphDataset[_VocabSub],
):
    def __init__(self, base_dataset: BaseSentenceToGraphDataset[_VocabSub]):
        cls_edge_name = "CLS_EDGE"
        assert cls_edge_name not in base_dataset.id2edge_type
        self._base_dataset = base_dataset
        self._id2edge_type = self._base_dataset.id2edge_type + [cls_edge_name]
        self._cls_edge_id = len(self._id2edge_type) - 1
        self._edge_type2id = self._base_dataset.edge_type2id.copy()
        self._edge_type2id.update({cls_edge_name: self._cls_edge_id})

    @property
    def base_dataset(self) -> BaseSentenceToGraphDataset[_VocabSub]:
        return self._base_dataset

    def _transform(self, example: GraphExample) -> GraphExample:
        cls_tok_id = self.base_dataset.numerizer.get_tok_id(
            self.base_dataset.numerizer.cls_tok
        )
        new_example = GraphExample(
            [
                self._connect_imp_nodes_to_new_node(g, cls_tok_id, self._cls_edge_id)
                for g in example.lsgraph
            ],
            example.lbl_id,
        )
        return new_example

    @property
    def sent2graph(self) -> sent2graphs.SentenceToGraph:
        # Technically not true, since we modify the graph in this class after getting
        # it from this sent2graph
        return self._base_dataset.sent2graph

    @property
    def numerizer(self) -> _VocabSub:
        return self._base_dataset.numerizer

    @property
    def id2edge_type(self) -> T.List[str]:
        return self._id2edge_type

    @property
    def edge_type2id(self) -> T.Dict[str, int]:
        return self._edge_type2id

    @staticmethod
    def _connect_imp_nodes_to_new_node(
        g: Graph, new_node_global_id: int, new_edge_global_id: int
    ) -> Graph:
        """Connect g.lsimp_node to a new node, and make the new node the only imp node.

        Used to connect "important nodes" (like dependency head nodes) to a CLS node.

        The CLS node will be inserted at the beginning of the list of nodes.

        For every node n in g.lsimp_node, adds edge (g.lsimp_node, new_node_global_id)
        # TODO: This actually needs to be the other way around. But I need to change
        this in a couple of other places. It's better to change it all at the same itme.
        """

        assert g.lsglobal_id is not None
        if new_node_global_id in g.lsglobal_id:
            raise Exception("new node to be added in graph already exists.")

        new_node_local_id = 0

        # Inserting CLS at the beginning
        lsedge_shifted = [(n1 + 1, n2 + 1) for n1, n2 in g.lsedge]
        lsimp_node_shifted = [n + 1 for n in g.lsimp_node]

        new_graph = Graph(
            lsedge=[(imp_node, new_node_local_id) for imp_node in lsimp_node_shifted]
            + lsedge_shifted,
            lsedge_type=[new_edge_global_id] * len(lsimp_node_shifted) + g.lsedge_type,
            lsimp_node=[new_node_local_id],
            nodeid2wordid=[new_node_global_id] + g.lsglobal_id,
        )
        return new_graph


class SentenceGraphDataset(BaseSentenceToGraphDataset[vocabs.BasicVocab], Cacheable):
    """A dataset that turns a TextSource into a dataset of `GraphExample`s.

    We handle here:
        1. That a TextSource might have multiple text columns per example, and therefore
           batched processing (necessary for speed for SRL for example) needs special
           care.
        2. That we definitely want to cache processed results(notice we inherit from
           Cacheable).
    """

    def __init__(
        self,
        txt_src: TextSource,
        sent2graph: sent2graphs.SentenceToGraph,
        vocab: vocabs.BasicVocab,
        cache_dir: Path,
        ignore_cache: bool = False,
        processing_batch_size: int = 800,
    ):
        """.

        Args:
            txt_src:
            sent2graph:
            vocab: We make sure here that the Sent2Graph passed and the Vocab passed
                   use the same tokenizer.
                   Also, we use vocab.tokenizer to tokenize before doing
                   sent2graph.to_graph().
            cache_dir: Look at Cacheable doc.
            ignore_cache: Look at Cacheable doc.
        """
        self._sent2graph = sent2graph
        self._txt_src = txt_src
        self._vocab = vocab
        self._processing_batch_size = processing_batch_size

        Cacheable.__init__(self, cache_dir=cache_dir, ignore_cache=ignore_cache)

    @property
    def sent2graph(self) -> sent2graphs.SentenceToGraph:
        return self._sent2graph

    @property
    def vocab(self) -> vocabs.BasicVocab:
        return self._vocab

    @property
    def id2edge_type(self) -> T.List[str]:
        return self._sent2graph.id2edge_type

    @property
    def edge_type2id(self) -> T.Dict[str, int]:
        return self._sent2graph.edge_type2id

    @property
    def _cached_attrs(self) -> T.Tuple[T.Tuple[str, TorchCachingTool], ...]:
        return (("_lssentgraph_ex", TorchCachingTool()),)

    def __repr__(self) -> str:
        return (
            "SentenceGraphDataset"
            f"-sent2graph_{self._sent2graph}"
            f"-txt_src_{self._txt_src}"
            f"-vocab_{self._vocab}"
        )

    def process(self) -> None:
        """This is where it all happens."""
        logger.info("Getting sentence graphs ...")
        batch_size = min(self._processing_batch_size, len(self._txt_src))
        # Do batched SRL prediction
        lslssent: T.List[T.List[str]]
        # Group the sent_ex's into batches
        batched_lssent_ex = grouper(self._txt_src, n=batch_size)

        lssentgraph_ex: T.List[GraphExample] = []
        num_batches = (len(self._txt_src) // batch_size) + int(
            len(self._txt_src) % batch_size != 0
        )
        self._sent2graph.init_workers()
        for lssent_ex in tqdm(
            batched_lssent_ex,
            desc=f"Turning sentence into graphs with batch size {batch_size}",
            total=num_batches,
        ):
            one_batch_lssentgraph_ex = self._batch_process_sent_ex(lssent_ex)
            lssentgraph_ex.extend(one_batch_lssentgraph_ex)

        self._sent2graph.finish_workers()
        self._lssentgraph_ex = lssentgraph_ex

    def _batch_process_sent_ex(
        self, lssent_ex: T.List[SentExample]
    ) -> T.List[GraphExample]:

        lslssent: T.List[T.List[str]]
        lslbl: T.List[str]
        lslssent, lslbl = map(list, zip(*lssent_ex))  # type: ignore

        # This part is complicated because we allow an arbitrary number of sentences
        # per example, AND we want to do batched prediction.
        # The trick is, for each batch do one "column" of the batch at a time.

        lslbl_id = self._vocab.labels.batch_get_lbl_id(lslbl)

        lsper_column_sentgraph: T.List[T.List[Graph]] = []
        for per_column_lssent in zip(*lslssent):

            # For turning the sentence into a graph
            # Note that we are bypassing any vocab filtering(like replacing with UNK)
            per_column_lsword = self._vocab.batch_tokenize(
                self._vocab.batch_simplify_txt(list(per_column_lssent))
            )
            per_column_sentgraph = self._sent2graph.batch_to_graph(per_column_lsword)

            # But we do want the UNK tokens for nodeid2wordid
            per_column_nodeid2wordid = self._vocab.get_lslstok_id(per_column_lsword)
            per_column_sentgraph = [
                Graph(
                    lsedge=lsedge,
                    lsedge_type=lsedge_type,
                    lsimp_node=lsimp_node,
                    nodeid2wordid=nodeid2wordid,
                )
                for (lsedge, lsedge_type, lsimp_node, _), nodeid2wordid in zip(
                    per_column_sentgraph, per_column_nodeid2wordid
                )
            ]

            lsper_column_sentgraph.append(per_column_sentgraph)
        lslssentgraph: T.List[T.List[Graph]] = list(
            map(list, zip(*lsper_column_sentgraph))
        )
        res = [
            GraphExample(lsgraph=lssentgraph, lbl_id=lbl_id)
            for lssentgraph, lbl_id in zip(lslssentgraph, lslbl_id)
        ]
        return res

    def _process_lssent_ex(self, sent_ex: SentExample) -> GraphExample:
        lssent, lbl = sent_ex
        lssentgraph: T.List[Graph] = []
        for sent in lssent:
            lsword = self._vocab.tokenize(self._vocab.simplify_txt(sent))
            lsedge, lsedge_type, lsimp_node, _ = self._sent2graph.to_graph(lsword)
            nodeid2wordid = self._vocab.tokenize_and_get_lstok_id(sent)
            sentgraph = Graph(
                lsedge=lsedge,
                lsedge_type=lsedge_type,
                lsimp_node=lsimp_node,
                nodeid2wordid=nodeid2wordid,
            )
            lssentgraph.append(sentgraph)
        lbl_id = self._vocab.labels.get_lbl_id(lbl)
        sentgraph_ex = GraphExample(lsgraph=lssentgraph, lbl_id=lbl_id)
        return sentgraph_ex

    def __len__(self) -> int:
        return len(self._lssentgraph_ex)

    def __getitem__(self, idx: int) -> GraphExample:
        if idx > len(self):
            raise IndexError(f"{idx} is >= {len(self)}")

        return self._lssentgraph_ex[idx]
