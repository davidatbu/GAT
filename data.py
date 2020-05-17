import csv
import hashlib
import json
import logging
import pickle as pkl
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sized
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
from utils import Edge
from utils import Node
from utils import to_undirected

logger = logging.getLogger("__main__")


class TextSource(Iterable[Tuple[str, str]], Sized):
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        for i in range(len(self)):
            yield self[i]


class FromIterableTextSource(TextSource):
    def __init__(self, iterable: Iterable[Tuple[str, str]]) -> None:
        self._ls = list(iterable)

    def __len__(self) -> int:
        return len(self._ls)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
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

    def __getitem__(self, idx: int) -> Tuple[str, str]:
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
        self, fp: Path, txt_col: str, lbl_col: str, allow_unlablled: bool
    ) -> None:

        self.fp_stem = fp.stem

        with fp.open() as f:
            reader = csv.reader(f)
            headers = next(reader)
            if headers.count(txt_col) != 1 or headers.count(lbl_col) != 1:
                raise Exception(
                    f"{txt_col} or {lbl_col} not found as a header in csv flie {str(fp)}, or were found more than once."
                )
            txt_col_i = headers.index(txt_col)
            lbl_col_i = headers.index(lbl_col)

            self.rows = [(row[txt_col_i], row[lbl_col_i]) for row in reader]

    def __repr__(self) -> str:
        return f"Csv_{self.fp_stem}"

    def __getitem__(self, idx: int) -> Tuple[str, str]:
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
        lstxt, lslbl = zip(*self.txt_src)

        if self.lower_case:
            lower_func: Callable[[str], str] = lambda s: s.lower()
            lstxt = tuple(map(lower_func, lstxt))

        # Compute word2id
        lslsword: List[Tuple[str, ...]] = [
            tuple(token.text for token in lstoken)
            for lstoken in self.splitter.batch_split_words(lstxt)
        ]
        word_counts: Counter[str] = Counter()
        for lsword in lslsword:
            word_counts.update(lsword)
        assert self.unk_thres is not None
        self._id2word = [
            word for word, count in word_counts.items() if count >= self.unk_thres
        ]
        # Unk token is always last
        self._id2word = ["[PAD]", "[CLS]", "[UNK]"] + self._id2word
        self._id2lbl = list(sorted(set(lslbl)))
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

    def lbl2id(self, lbl: str) -> int:
        return self._lbl2id[lbl]

    def tokenize_before_unk(self, sent: str) -> Tuple[str, ...]:
        if self.lower_case:
            sent = sent.lower()
        before_unk = tuple(token.text for token in self.splitter.split_words(sent))
        return before_unk


class SentenceGraphDataset(
    Iterable[Tuple[Tuple[List[Node], List[Edge], List[Node]], int]],
    Sized,
    Cacheable,
    Dataset,  # type: ignore
):
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

        Cacheable.__init__(self, cache_dir=cache_dir, ignore_cache=ignore_cache)

        self._lslsedge_index: List[List[Edge]]  # just a type annotation
        if undirected_edges:
            self._lslsedge_index = self.batch_to_undirected(self._lslsedge_index)

    @property
    def cached_attrs(self) -> List[Tuple[Literal["torch", "json", "pkl"], str]]:
        return [
            ("pkl", "_lslsglobal_node"),
            ("pkl", "_lslsedge_index"),
            ("pkl", "_lslshead_node"),
            ("pkl", "_lslbl"),
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
        lslsglobal_node: List[List[Node]] = []
        lslshead_node: List[List[Node]] = []
        lslsedge_index: List[List[Edge]] = []
        lssent, lslbl = list(zip(*self.txt_src))
        for sent in tqdm(lssent, desc="Doing SRL"):
            lsword = self.vocab_and_emb.tokenize_before_unk(sent)
            lshead_node, lsedge_index, lsedge_type = self.sent2graph.to_graph(lsword)
            # We get indices relative to sentence beginngig, convert these to global ids
            lsglobal_node = [self.vocab_and_emb.word2id(word) for word in lsword]

            lslsglobal_node.append(lsglobal_node)
            lslshead_node.append(lshead_node)
            lslsedge_index.append(lsedge_index)

        self._lslsglobal_node = lslsglobal_node
        self._lslshead_node = lslshead_node
        self._lslsedge_index = lslsedge_index
        self._lslbl = [self.vocab_and_emb.lbl2id(lbl) for lbl in lslbl]

    def __len__(self) -> int:
        return len(self._lslsedge_index)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[List[Node], List[Edge], List[Node]], int]:
        if idx > len(self):
            raise IndexError(f"{idx} is >= {len(self)}")
        lsglobal_node = self._lslsglobal_node[idx]
        lshead_node = self._lslshead_node[idx]
        lsedge_index = self._lslsedge_index[idx]
        lbl = self._lslbl[idx]

        return (lsglobal_node, lsedge_index, lshead_node), lbl

    def __iter__(
        self,
    ) -> Iterator[Tuple[Tuple[List[Node], List[Edge], List[Node]], int]]:
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def batch_to_undirected(cls, lslsedge_index: List[List[Edge]]) -> List[List[Edge]]:
        return [to_undirected(lsedge_index) for lsedge_index in lslsedge_index]

    @staticmethod
    def collate_fn(
        batch: List[Tuple[Tuple[List[Node], List[Edge], List[Node]], int]]
    ) -> Tuple[List[Tuple[List[Node], List[Edge], List[Node]]], List[int]]:

        X, y = zip(*batch)
        return list(X), list(y)

    def draw_networkx_graph(
        self,
        lsglobal_node: List[Node],
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

        lshead_node, lslbl = zip(*lslbled_node)
        lsnode_color: List[str] = [
            "b" if node in lshead_node else "y" for node in lsnode
        ]

        # Append node label to nx "node labels"
        for head_node, lbl in lslbled_node:
            node2word[head_node] += f"|label={lbl}"

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
    dataset_dir: Path, splits: List[str] = ["train", "val"]
) -> Tuple[Dict[str, SentenceGraphDataset], VocabAndEmb]:

    txt_srcs = {
        split: CsvTextSource(
            fp=(dataset_dir / f"{split}.csv"),
            txt_col="news_title",
            lbl_col="Q3 Theme1",
            allow_unlablled=False,
        )
        for split in splits
    }

    cat_txt_src = ConcatTextSource(*txt_srcs.values())
    vocab_and_emb = VocabAndEmb(
        txt_src=cat_txt_src,
        cache_dir=dataset_dir,
        embedder=GloveWordToVec(),
        unk_thres=1,
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

    return split_datasets, vocab_and_emb


def main() -> None:
    dataset_dir = Path(  # noqa:
            "/data/paraphrase/paws/"
    )


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
