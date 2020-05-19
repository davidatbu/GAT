import csv
import hashlib
import json
import logging
import pickle as pkl
from itertools import starmap
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import torch
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import Counter
from typing_extensions import Literal

from embeddings import WordToVec
from glove_embeddings import GloveWordToVec
from sent2graph import SentenceToGraph
from sent2graph import SRLSentenceToGraph
from utils import EdgeList
from utils import grouper
from utils import Node
from utils import NodeList
from utils import SentExample
from utils import SentGraph
from utils import SentgraphExample
from utils import to_undirected

# from p_tqdm import p_map  # type: ignore

logger = logging.getLogger("__main__")


class TextSource:
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
        return "Cat" + "-" + "-".join(str(txt_src) for txt_src in self.lstxt_src)


class CsvTextSource(TextSource):
    def __init__(
        self,
        fp: Path,
        lstxt_col: List[str],
        lbl_col: str,
        allow_unlablled: bool,
        csv_reader_kwargs: Dict[str, Any] = {},
    ) -> None:

        self.fp_stem = fp.stem

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
                    lssent=tuple(row[txt_col_i] for txt_col_i in lstxt_col_i),
                    lbl=row[lbl_col_i],
                )
                for row in reader
            ]

    def __repr__(self) -> str:
        return f"Csv_{self.fp_stem}"

    def __getitem__(self, idx: int) -> SentExample:
        return self.rows[idx]

    def __len__(self) -> int:
        return len(self.rows)


class Cacheable:
    def __init__(self, cache_dir: Path, ignore_cache: bool) -> None:
        self.specific_cache_dir = cache_dir / str(self)
        self.specific_cache_dir.mkdir(exist_ok=True)
        if self.cached_exists() and not (ignore_cache):
            logger.info(f"{str(self)} found cached.")
            self.from_cache()
        else:
            logger.info(f"{str(self)} not found cached. Processing ...")
            self.process()
            self.to_cache()

    @property
    def cached_attrs(self) -> List[Tuple[Literal["torch", "pkl", "json"], str]]:
        raise NotImplementedError()

    @property
    def lscache_uniquer_attr(self) -> List[str]:
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

    def __repr__(self) -> str:
        return (
            type(self).__name__
            + "-"
            + "-".join(
                [
                    f"{attr[:4]}_{str(getattr(self, attr))}"
                    for attr in self.lscache_uniquer_attr
                ]
            )
        )


class VocabAndEmb(Cacheable):
    def __init__(
        self,
        txt_src: TextSource,
        cache_dir: Path,
        embedder: WordToVec,
        lower_case: bool = True,
        unk_thres: int = 1,
        ignore_cache: bool = False,
    ) -> None:
        self.lower_case = lower_case
        self.embedder = embedder
        self.unk_thres = unk_thres
        self.txt_src = txt_src
        self.splitter = SpacyWordSplitter()

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
        return [("pkl", "_id2word"), ("pkl", "_id2lbl"), ("torch", "embs")]

    @property
    def lscache_uniquer_attr(self) -> List[str]:
        return ["lower_case", "embedder", "unk_thres", "txt_src"]

    def process(self) -> None:

        # Compute word2id

        word_counts: Counter[str] = Counter()
        lslbl: List[str] = []

        for lssent, lbl in self.txt_src:
            lslbl.append(lbl)
            for sent in lssent:
                if self.lower_case:
                    sent = sent.lower()
                    lstoken = self.splitter.split_words(sent)
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

        embs = torch.zeros((len(self._id2word), self.embedder.dim))
        self.embedder.prefetch_lsword(self._id2word[self._real_tokens_start :])
        self.embedder.set_unk_as_avg()
        embs[self._real_tokens_start :] = self.embedder.for_lsword(
            self._id2word[self._real_tokens_start :]
        )
        embs[self._unk_id] = self.embedder.for_unk()
        self.embs = embs
        logger.info(f"Got vocabulary embeddings of shape {self.embs.shape}")

    def word2id(self, word: str) -> int:
        return self._word2id.get(word, self._unk_id)

    def batch_id2word(self, lsword_id: List[int]) -> List[str]:
        return [self._id2word[word_id] for word_id in lsword_id]

    def lbl2id(self, lbl: str) -> int:
        return self._lbl2id[lbl]

    def tokenize_before_unk(self, sent: str) -> Tuple[str, ...]:
        if self.lower_case:
            sent = sent.lower()
        before_unk = tuple(token.text for token in self.splitter.split_words(sent))
        return before_unk


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
    ):

        self.sent2graph = sent2graph
        self.txt_src = txt_src
        self.vocab_and_emb = vocab_and_emb
        self._undirected_edges = undirected_edges

        Cacheable.__init__(self, cache_dir=cache_dir, ignore_cache=ignore_cache)

        self._lslssentgraph_ex: List[SentgraphExample] = []

    @property
    def cached_attrs(self) -> List[Tuple[Literal["torch", "json", "pkl"], str]]:
        return [
            ("pkl", "_lslssentgraph_ex"),
        ]

    @property
    def lscache_uniquer_attr(self) -> List[str]:
        return [
            "sent2graph",
            "txt_src",
            "vocab_and_emb",
        ]

    def process(self) -> None:
        logger.info("Getting sentence graphs ...")
        lssentgraph_ex: List[SentgraphExample] = list(
            tqdm(
                starmap(self._process_lssent, *zip(*self.txt_src),),
                desc="Tokenizing and graphizing",
            )
        )

        self._lslssentgraph_ex = lssentgraph_ex

    def _process_lssent(self, lssent: List[str], lbl: str) -> SentgraphExample:
        lssentgraph: List[SentGraph] = []
        for sent in lssent:
            lsword = self.vocab_and_emb.tokenize_before_unk(sent)
            lsedge, lsedge_type, lsimp_node, _ = self.sent2graph.to_graph(lsword)
            # We get indices relative to sentence beginngig, convert these to global ids
            nodeid2wordid = [self.vocab_and_emb.word2id(word) for word in lsword]
            sentgraph = SentGraph(
                lsedge=lsedge,
                lsedge_type=lsedge_type,
                lsimp_node=lsimp_node,
                nodeid2wordid=nodeid2wordid,
            )
            lssentgraph.append(sentgraph)
        lbl_id = self.vocab_and_emb.lbl2id(lbl)
        sentgraph_ex = SentgraphExample(lssentgraph=tuple(lssentgraph), lbl_id=lbl_id)
        return sentgraph_ex

    def __len__(self) -> int:
        return len(self._lslssentgraph_ex)

    def __getitem__(self, idx: int) -> SentgraphExample:
        if idx > len(self):
            raise IndexError(f"{idx} is >= {len(self)}")

        return self._lslssentgraph_ex[idx]

    def __iter__(self,) -> Iterator[SentgraphExample]:
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def batch_to_undirected(cls, lslsedge_index: List[EdgeList]) -> List[EdgeList]:
        return [to_undirected(lsedge_index) for lsedge_index in lslsedge_index]

    @staticmethod
    def collate_fn(
        batch: List[Tuple[Tuple[NodeList, EdgeList, NodeList], int]]
    ) -> Tuple[List[Tuple[NodeList, EdgeList, NodeList]], List[int]]:

        X, y = zip(*batch)
        return list(X), list(y)

    def draw_networkx_graph(
        self,
        lsglobal_node: NodeList,
        tcadj: torch.Tensor,
        lslbled_node: List[Tuple[Node, int]],
    ) -> None:
        import matplotlib.pyplot as plt
        import networkx as nx  # type: ignore

        lsedge_index = list(map(tuple, torch.nonzero(tcadj, as_tuple=False).tolist()))

        # Create nicer names
        node2word = {
            i: self.vocab_and_emb._id2word[word_id]
            for i, word_id in enumerate(lsglobal_node)
        }
        lsnode = list(range(len(lsglobal_node)))

        lsimp_node, lslbl = zip(*lslbled_node)
        lsnode_color: List[str] = [
            "b" if node in lsimp_node else "y" for node in lsnode
        ]

        # Append node label to nx "node labels"
        for imp_node, lbl in lslbled_node:
            node2word[imp_node] += f"|label={lbl}"

        G = nx.Graph()
        G.add_nodes_from(lsnode)
        G.add_edges_from(lsedge_index)

        pos = nx.planar_layout(G)
        nx.draw(
            G,
            pos=pos,
            labels=node2word,
            node_color=lsnode_color,  # node_size=1000, # size=10000,
        )
        plt.show()


def load_splits(
    dataset_dir: Path,
    splits: List[str] = ["train", "val"],
    fp_ending: str = "tsv",
    lstxt_col: List[str] = ["sentence1", "sentence2"],
    lbl_col: str = "label",
    delimiter: str = "\t",
) -> Tuple[Dict[str, SentenceGraphDataset], VocabAndEmb]:

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
        unk_thres=2,
    )

    split_datasets = {  # noqa:
        split: SentenceGraphDataset(
            cache_dir=dataset_dir,
            txt_src=txt_src,
            sent2graph=SRLSentenceToGraph(),
            vocab_and_emb=vocab_and_emb,
        )
        for split, txt_src in txt_srcs.items()
    }

    logger.info("First 10 of each split")
    for split, dataset in split_datasets.items():
        logger.info(f"{split}")
        for i in range(min(len(dataset), 5)):
            lssentgraph, lbl_id = dataset[i]
            print(f"{vocab_and_emb._id2lbl[lbl_id]}")
            for _, _, _, lsword_id in lssentgraph:
                print(f"\t{vocab_and_emb.batch_id2word(lsword_id)}")  # type: ignore

    return split_datasets, vocab_and_emb


def main() -> None:
    dataset_dir = Path("data/paraphrase/paws/")  # noqa:
    load_splits(
        dataset_dir,
        splits=["train", "val"],
        # fp_ending="csv",
        # lstxt_col=["news_title"],
        # lbl_col="Q3 Theme1",
        # delimiter=",",
    )


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
