import csv
import hashlib
import json
import logging
import pickle as pkl
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sized
from typing import Tuple
from typing import Type

import torch
from allennlp.data.tokenizers import SpacyTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import Counter
from typing_extensions import Literal

from ..embeddings.base import WordToVec
from ..embeddings.glove import GloveWordToVec
from ..sent2graph.base import SentenceToGraph
from ..sent2graph.dep import DepSentenceToGraph
from ..sent2graph.srl import SRLSentenceToGraph
from ..utils.base import grouper
from ..utils.base import SentExample
from ..utils.base import SentGraph
from ..utils.base import SentgraphExample


logger = logging.getLogger("__main__")


class TextSource(Iterable[SentExample], Sized):
    def __getitem__(self, idx: int) -> SentExample:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[SentExample]:
        for i in range(len(self)):
            yield self[i]


class FromIterableTextSource(TextSource):
    def __init__(self, iterable: Iterable[SentExample]) -> None:
        self._ls = list(iterable)

    def __len__(self) -> int:
        return len(self._ls)

    def __getitem__(self, idx: int) -> SentExample:
        if idx < 0 or idx > len(self):
            raise IndexError(
                f"f{self.__class__.__name__} has only {len(self)} items. {idx} was asked, which is either negative or gretaer than length."
            )
        return self._ls[idx]

    def __repr__(self) -> str:
        return hashlib.sha1(str(self._ls).encode()).hexdigest()


class ConcatTextSource(TextSource):
    def __init__(self, arg: TextSource, *args: TextSource):
        self.lstxt_src = (arg,) + args
        self.lens = list(map(len, self.lstxt_src))

    def __getitem__(self, idx: int) -> SentExample:
        cur_txt_src_i = 0
        cur_len = len(self.lstxt_src[cur_txt_src_i])
        while idx >= cur_len:
            idx -= cur_len
            cur_txt_src_i += 1
        return self.lstxt_src[cur_txt_src_i][idx]

    def __len__(self) -> int:
        return sum(self.lens)

    def __repr__(self) -> str:
        return (
            "ConcatTextSource"
            + "-"
            + "-".join(str(txt_src) for txt_src in self.lstxt_src)
        )


class CsvTextSource(TextSource):
    def __init__(
        self,
        fp: Path,
        lstxt_col: List[str],
        lbl_col: str,
        allow_unlablled: bool,
        csv_reader_kwargs: Dict[str, Any] = {},
    ) -> None:

        self.fp = fp

        with fp.open() as f:
            reader = csv.reader(f, **csv_reader_kwargs)
            headers = next(reader)
            for txt_col in lstxt_col:
                if headers.count(txt_col) != 1 or headers.count(lbl_col) != 1:
                    raise Exception(
                        f"{txt_col} or {lbl_col} not found as a header in csv flie {str(fp)}, or were found more than once."
                    )
            lstxt_col_i = [headers.index(txt_col) for txt_col in lstxt_col]
            lbl_col_i = headers.index(lbl_col)

            self.rows = [
                SentExample(
                    lssent=[row[txt_col_i] for txt_col_i in lstxt_col_i],
                    lbl=row[lbl_col_i],
                )
                for row in reader
            ]

    def __repr__(self) -> str:
        return f"CsvTextSource-fp_{str(self.fp)}"

    def __getitem__(self, idx: int) -> SentExample:
        return self.rows[idx]

    def __len__(self) -> int:
        return len(self.rows)


class Cacheable:
    """Subclassers must implement a meaningful __repr___"""

    def __init__(self, cache_dir: Path, ignore_cache: bool) -> None:

        # Use the  repr to create a cache dir
        obj_repr_hash = hashlib.sha1(str(self).encode()).hexdigest()
        self.specific_cache_dir = cache_dir / obj_repr_hash
        self.specific_cache_dir.mkdir(exist_ok=True)

        if self.cached_exists() and not (ignore_cache):
            logger.info(f"{obj_repr_hash} found cached.")
            self.from_cache()
        else:
            logger.info(f"{obj_repr_hash} not found cached. Processing ...")
            self.process()
            self.to_cache()

    @property
    def cached_attrs(self) -> List[Tuple[Literal["torch", "pkl", "json"], str]]:
        raise NotImplementedError()

    def process(self) -> None:
        raise NotImplementedError()

    def cached_exists(self) -> bool:
        return all(
            [
                self._cache_fp_for_attr(pkling_method, attr_name).exists()
                for pkling_method, attr_name in self.cached_attrs
            ]
        )

    def from_cache(self) -> None:
        for pkling_method, attr_name in self.cached_attrs:
            fp = self._cache_fp_for_attr(pkling_method, attr_name)

            if pkling_method == "torch":
                with fp.open("rb") as fb:
                    obj = torch.load(fb)  # type: ignore
            elif pkling_method == "pkl":
                with fp.open("rb") as fb:
                    obj = pkl.load(fb)
            elif pkling_method == "json":
                with fp.open() as f:
                    obj = json.load(f)
            else:
                raise Exception("pkcling method")
            setattr(self, attr_name, obj)

    def _cache_fp_for_attr(self, pkling_method: str, attr_name: str) -> Path:
        return self.specific_cache_dir / f"{attr_name}.{pkling_method}"

    def to_cache(self) -> None:
        for pkling_method, attr_name in self.cached_attrs:

            fp = self._cache_fp_for_attr(pkling_method, attr_name)
            obj = getattr(self, attr_name)
            if pkling_method == "torch":
                with fp.open("wb") as fb:
                    torch.save(obj, fb)  # type: ignore
            elif pkling_method == "pkl":
                with fp.open("wb") as fb:
                    pkl.dump(obj, fb)
            elif pkling_method == "json":
                with fp.open() as f:
                    json.dump(obj, f)
            else:
                raise Exception("pkcling method")


class VocabAndEmb(Cacheable):
    def __init__(
        self,
        txt_src: TextSource,
        cache_dir: Path,
        embedder: Optional[WordToVec],
        lower_case: bool = True,
        unk_thres: int = 1,
        ignore_cache: bool = False,
    ) -> None:
        self.lower_case = lower_case
        self.embedder = embedder
        self.unk_thres = unk_thres
        self.txt_src = txt_src
        self.tokenizer = SpacyTokenizer()

        self._pad_id = 0
        self._cls_id = 1
        self._unk_id = 2
        self._real_tokens_start = 3

        Cacheable.__init__(self, cache_dir=cache_dir, ignore_cache=ignore_cache)

        self._lbl2id: Dict[str, int] = {
            lbl: id_ for id_, lbl in enumerate(self._id2lbl)
        }
        self._word2id: Dict[str, int] = {
            word: id_ for id_, word in enumerate(self._id2word)
        }

    @property
    def cached_attrs(self) -> List[Tuple[Literal["torch", "json", "pkl"], str]]:
        return [("pkl", "_id2word"), ("pkl", "_id2lbl"), ("torch", "_embs")]

    def __repr__(self) -> str:
        return f"VocabAndEmb-lower_case_{self.lower_case}-embedder_{self.embedder}_unk_thres_{self.unk_thres}_txt_src_{self.txt_src}"

    def process(self) -> None:

        # Compute word2id

        word_counts: Counter[str] = Counter()
        lslbl: List[str] = []

        for lssent, lbl in self.txt_src:
            lslbl.append(lbl)
            for sent in lssent:
                if self.lower_case:
                    sent = sent.lower()
                    lstoken = self.tokenizer.tokenize(sent)
                    lsword = [token.text for token in lstoken]
                    word_counts.update(lsword)

        self._id2word = [
            word for word, count in word_counts.items() if count >= self.unk_thres
        ]
        self._id2word = ["[PAD]", "[CLS]", "[UNK]"] + self._id2word
        self._id2lbl = list(sorted(set(lslbl)))
        self._lblcnt = Counter(lslbl)
        logger.info(f"Made id2word of length {len(self._id2word)}")
        logger.info(f"Made id2lbl of length {len(self._id2lbl)}")

        if self.embedder is not None:
            embs = torch.zeros((len(self._id2word), self.embedder.dim))
            self.embedder.prefetch_lsword(self._id2word[self._real_tokens_start :])
            self.embedder.set_unk_as_avg()
            embs[self._real_tokens_start :] = self.embedder.for_lsword(
                self._id2word[self._real_tokens_start :]
            )
            embs[self._unk_id] = self.embedder.for_unk()
            self._embs = embs
            logger.info(f"Got vocabulary embeddings of shape {embs.shape}")
        else:
            logger.info("Not getting vecs")
            self._embs = torch.tensor([0])

    @property
    def embs(self) -> torch.Tensor:
        if self.embedder is None:
            raise Exception("No embedder was provided, so embs is not set.")
        return self._embs

    def word2id(self, word: str) -> int:
        return self._word2id.get(word, self._unk_id)

    def batch_id2word(self, lsword_id: List[int]) -> List[str]:
        return [self._id2word[word_id] for word_id in lsword_id]

    def lbl2id(self, lbl: str) -> int:
        return self._lbl2id[lbl]

    def tokenize_before_unk(self, sent: str) -> List[str]:
        if self.lower_case:
            sent = sent.lower()
        before_unk = [token.text for token in self.tokenizer.tokenize(sent)]
        return before_unk

    def batch_tokenize_before_unk(self, lssent: List[str]) -> List[List[str]]:
        return [self.tokenize_before_unk(sent) for sent in lssent]


class SliceDataset(Dataset):  # type: ignore
    def __init__(self, orig_ds: Dataset, n: int) -> None:  # type: ignore
        assert len(orig_ds) >= n
        self.orig_ds = orig_ds
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Any:
        if i >= len(self):
            raise IndexError("SliceDataset ended.")
        return self.orig_ds[i]


class SentenceGraphDataset(Dataset, Cacheable):  # type: ignore
    def __init__(
        self,
        cache_dir: Path,
        txt_src: TextSource,
        sent2graph: SentenceToGraph,
        vocab_and_emb: VocabAndEmb,
        undirected_edges: bool = True,
        ignore_cache: bool = False,
        unk_thres: Optional[int] = None,
        processing_batch_size: int = 800,
    ):

        self.sent2graph = sent2graph
        self.txt_src = txt_src
        self.vocab_and_emb = vocab_and_emb
        self._undirected_edges = undirected_edges
        self._processing_batch_size = processing_batch_size

        Cacheable.__init__(self, cache_dir=cache_dir, ignore_cache=ignore_cache)

    @property
    def cached_attrs(self) -> List[Tuple[Literal["torch", "json", "pkl"], str]]:
        return [
            ("pkl", "_lssentgraph_ex"),
        ]

    def __repr__(self) -> str:
        return f"SentenceGraphDataset-sent2graph_{self.sent2graph}-txt_src_{self.txt_src}-vocab_and_emb_{self.vocab_and_emb}"

    def process(self) -> None:
        logger.info("Getting sentence graphs ...")
        batch_size = min(self._processing_batch_size, len(self.txt_src))
        # Do batched SRL prediction
        lslssent: List[List[str]]
        # Group the sent_ex's into batches
        batched_lssent_ex = grouper(self.txt_src, n=batch_size)

        lssentgraph_ex: List[SentgraphExample] = []
        num_batches = (len(self.txt_src) // batch_size) + int(
            len(self.txt_src) % batch_size != 0
        )
        self.sent2graph.init_workers()
        for lssent_ex in tqdm(
            batched_lssent_ex,
            desc=f"Turning sentence into graphs with batch size {batch_size}",
            total=num_batches,
        ):
            one_batch_lssentgraph_ex = self._batch_process_sent_ex(lssent_ex)
            lssentgraph_ex.extend(one_batch_lssentgraph_ex)

        self.sent2graph.finish_workers()
        self._lssentgraph_ex = lssentgraph_ex

    def _batch_process_sent_ex(
        self, lssent_ex: List[SentExample]
    ) -> List[SentgraphExample]:

        lslssent: List[List[str]]
        lslbl: List[str]
        lslssent, lslbl = map(list, zip(*lssent_ex))  # type: ignore

        # Easily deadl with lblids
        lslbl_id = [self.vocab_and_emb.lbl2id(lbl) for lbl in lslbl]

        lsper_column_sentgraph: List[List[SentGraph]] = []
        for per_column_lssent in zip(*lslssent):
            per_column_lsword = self.vocab_and_emb.batch_tokenize_before_unk(
                list(per_column_lssent)
            )
            per_column_nodeid2wordid = [
                [self.vocab_and_emb.word2id(word) for word in lsword]
                for lsword in per_column_lsword
            ]
            per_column_sentgraph = self.sent2graph.batch_to_graph(per_column_lsword)

            per_column_sentgraph = [
                SentGraph(
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
        lslssentgraph: List[List[SentGraph]] = list(
            map(list, zip(*lsper_column_sentgraph))
        )
        res = [
            SentgraphExample(lssentgraph=lssentgraph, lbl_id=lbl_id)
            for lssentgraph, lbl_id in zip(lslssentgraph, lslbl_id)
        ]
        return res

    def _process_lssent_ex(self, sent_ex: SentExample) -> SentgraphExample:
        lssent, lbl = sent_ex
        lssentgraph: List[SentGraph] = []
        for sent in lssent:
            lsword = self.vocab_and_emb.tokenize_before_unk(sent)
            lsedge, lsedge_type, lsimp_node, _ = self.sent2graph.to_graph(lsword)
            # We get indices relative to sentence beginnig, convert these to global ids
            nodeid2wordid = [self.vocab_and_emb.word2id(word) for word in lsword]
            sentgraph = SentGraph(
                lsedge=lsedge,
                lsedge_type=lsedge_type,
                lsimp_node=lsimp_node,
                nodeid2wordid=nodeid2wordid,
            )
            lssentgraph.append(sentgraph)
        lbl_id = self.vocab_and_emb.lbl2id(lbl)
        sentgraph_ex = SentgraphExample(lssentgraph=lssentgraph, lbl_id=lbl_id)
        return sentgraph_ex

    def __len__(self) -> int:
        return len(self._lssentgraph_ex)

    def __getitem__(self, idx: int) -> SentgraphExample:
        if idx > len(self):
            raise IndexError(f"{idx} is >= {len(self)}")

        return self._lssentgraph_ex[idx]

    def sentgraph_ex_to_sent_ex(self, sent_graph_ex: SentgraphExample) -> SentExample:
        lslsword_id = [
            sentgraph.nodeid2wordid for sentgraph in sent_graph_ex.lssentgraph
        ]
        lssent: List[str] = []
        for lsword_id in lslsword_id:
            assert lsword_id is not None
            lssent.append(
                " ".join(self.vocab_and_emb._id2word[word_id] for word_id in lsword_id)
            )
        lbl = self.vocab_and_emb._id2lbl[sent_graph_ex.lbl_id]

        return SentExample(lssent=lssent, lbl=lbl)

    def __iter__(self,) -> Iterator[SentgraphExample]:
        for i in range(len(self)):
            yield self[i]

    def sentgraph_to_svg(self, sentgraph: SentGraph) -> str:
        import networkx as nx  # type: ignore

        g = nx.DiGraph()

        def quote(s: str) -> str:
            return '"' + s.replace('"', '"') + '"'

        assert sentgraph.nodeid2wordid is not None
        # NetworkX format
        lsnode_word: List[Tuple[int, Dict[str, str]]] = [
            (node_id, {"label": quote(self.vocab_and_emb._id2word[word_id])})
            for node_id, word_id in enumerate(sentgraph.nodeid2wordid)
        ]

        # Edges in nx format
        lsedge_role: List[Tuple[int, int, Dict[str, str]]] = [
            (n1, n2, {"label": quote(self.sent2graph.id2edge_type[edge_id])})
            for (n1, n2), edge_id in zip(sentgraph.lsedge, sentgraph.lsedge_type)
        ]
        g.add_nodes_from(lsnode_word)
        g.add_edges_from(lsedge_role)
        p = nx.drawing.nx_pydot.to_pydot(g)
        return p.create_svg().decode()  # type: ignore

    @staticmethod
    def collate_fn(
        batch: List[SentgraphExample],
    ) -> Tuple[List[List[SentGraph]], List[int]]:
        X, y = zip(*batch)
        return list(X), list(y)


SENT2GRAPHS: Dict[str, Type[SentenceToGraph]] = {
    "srl": SRLSentenceToGraph,
    "dep": DepSentenceToGraph,
}


def load_splits(
    dataset_dir: Path,
    sent2graph_name: Literal["srl", "dep"],
    splits: List[str] = ["train", "val"],
    fp_ending: str = "tsv",
    lstxt_col: List[str] = ["sentence1", "sentence2"],
    lbl_col: str = "label",
    delimiter: str = "\t",
    unk_thres: int = 2,
) -> Tuple[Dict[str, SentenceGraphDataset], Dict[str, CsvTextSource], VocabAndEmb]:

    assert "train" in splits
    txt_srcs = {
        split: CsvTextSource(
            fp=(dataset_dir / f"{split}.{fp_ending}"),
            lstxt_col=lstxt_col,
            lbl_col=lbl_col,
            allow_unlablled=False,
            csv_reader_kwargs={"delimiter": delimiter},
        )
        for split in splits
    }

    vocab_and_emb = VocabAndEmb(
        txt_src=txt_srcs["train"],
        cache_dir=dataset_dir,
        embedder=GloveWordToVec(),
        unk_thres=unk_thres,
    )

    cls_sent2graph = SENT2GRAPHS[sent2graph_name]
    split_datasets = {  # noqa:
        split: SentenceGraphDataset(
            cache_dir=dataset_dir,
            txt_src=txt_src,
            sent2graph=cls_sent2graph(),
            vocab_and_emb=vocab_and_emb,
            processing_batch_size=1000,
        )
        for split, txt_src in txt_srcs.items()
    }

    logger.info("First 10 of each split")
    for split, dataset in split_datasets.items():
        logger.info(f"{split}")
        for i in range(min(len(dataset), 5)):
            lssentgraph, lbl_id = dataset[i]
            logger.info(f"{vocab_and_emb._id2lbl[lbl_id]}")
            for _, _, _, lsword_id in lssentgraph:
                logger.info(f"\t{vocab_and_emb.batch_id2word(lsword_id)}")  # type: ignore

    return split_datasets, txt_srcs, vocab_and_emb
