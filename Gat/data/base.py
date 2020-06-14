from __future__ import annotations

import abc
import csv
import hashlib
import logging
import typing as T
from pathlib import Path

import torch
import typing_extensions as TT
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore
from typing_extensions import Counter

from ..sent2graph.base import SentenceToGraph
from ..sent2graph.dep import DepSentenceToGraph
from ..sent2graph.srl import SRLSentenceToGraph
from ..utils.base import grouper
from ..utils.base import SentExample
from ..utils.base import SentGraph
from ..utils.base import SentgraphExample
from Gat.data import tokenizers


logger = logging.getLogger("__main__")


class TextSource(T.Iterable[SentExample], T.Sized):
    def __getitem__(self, idx: int) -> SentExample:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def __iter__(self) -> T.Iterator[SentExample]:
        for i in range(len(self)):
            yield self[i]


class FromIterableTextSource(TextSource):
    def __init__(self, iterable: T.Iterable[SentExample]) -> None:
        self._ls = list(iterable)

    def __len__(self) -> int:
        return len(self._ls)

    def __getitem__(self, idx: int) -> SentExample:
        if idx < 0 or idx > len(self):
            raise IndexError(
                f"f{self.__class__.__name__} has only {len(self)} items. {idx} was asked, which is either negative or "
                "greater than length."
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
        lstxt_col: T.List[str],
        lbl_col: str,
        allow_unlablled: bool,
        csv_reader_kwargs: T.Dict[str, T.Any] = {},
    ) -> None:

        self.fp = fp

        with fp.open() as f:
            reader = csv.reader(f, **csv_reader_kwargs)
            headers = next(reader)
            for txt_col in lstxt_col:
                if headers.count(txt_col) != 1 or headers.count(lbl_col) != 1:
                    raise Exception(
                        f"{txt_col} or {lbl_col} not found as a header in csv flie {str(fp)},"
                        " or were found more than once."
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
    def cached_attrs(self) -> T.List[str]:
        raise NotImplementedError()

    def process(self) -> None:
        raise NotImplementedError()

    def cached_exists(self) -> bool:
        return all(
            [
                self._cache_fp_for_attr(attr_name).exists()
                for attr_name in self.cached_attrs
            ]
        )

    def from_cache(self) -> None:
        for attr_name in self.cached_attrs:
            fp = self._cache_fp_for_attr(attr_name)

            with fp.open("rb") as fb:
                obj = torch.load(fb)  # type: ignore
            setattr(self, attr_name, obj)

    def _cache_fp_for_attr(self, attr_name: str) -> Path:
        return self.specific_cache_dir / f"{attr_name}.torch"

    def to_cache(self) -> None:
        for attr_name in self.cached_attrs:

            fp = self._cache_fp_for_attr(attr_name)
            obj = getattr(self, attr_name)
            with fp.open("wb") as fb:
                torch.save(obj, fb)  # type: ignore


class Vocab(abc.ABC):
    """
    Rule:
        For every method, have a batch version.
    """

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    def simplify_txt(self, txt: str) -> str:
        """This would be the place to do things like lowercasing, stripping out punctuation, ..etc"""
        return txt

    def batch_simplify_txt(self, lstxt: T.List[str]) -> T.List[str]:
        return [self.simplify_txt(txt) for txt in lstxt]

    @abc.abstractproperty
    def tokenizer(self) -> tokenizers.Tokenizer:
        pass

    @abc.abstractmethod
    def get_toks(self, lstok_id: T.List[int]) -> T.List[str]:
        pass

    def batch_get_toks(self, lslstok_id: T.List[T.List[int]]) -> T.List[T.List[str]]:
        return [self.get_toks(lstok_id) for lstok_id in lslstok_id]

    @abc.abstractmethod
    def get_tok_ids(self, lsword: T.List[str]) -> T.List[int]:
        pass

    def batch_get_tok_ids(self, lslsword: T.List[T.List[str]]) -> T.List[T.List[int]]:
        return [self.get_tok_ids(lsword) for lsword in lslsword]

    def tokenize_and_get_tok_ids(self, txt: str) -> T.List[int]:
        return self.get_tok_ids(self.tokenizer.tokenize(txt))

    def batch_tokenize_and_get_tok_ids(self, lstxt: T.List[str]) -> T.List[T.List[int]]:
        return self.batch_get_tok_ids(self.tokenizer.batch_tokenize(lstxt))

    @abc.abstractmethod
    def get_lbl_id(self, lbl: str) -> int:
        pass

    def batch_get_lbl_id(self, lslbl: T.List[str]) -> T.List[int]:
        return [self.get_lbl_id(lbl) for lbl in lslbl]

    @abc.abstractmethod
    def get_lbl(self, lbl_id: int) -> str:
        pass

    def batch_get_lbl(self, lslbl_id: T.List[int]) -> T.List[str]:
        return [self.get_lbl(lbl_id) for lbl_id in lslbl_id]

    @abc.abstractproperty
    def padding_tok_id(self) -> int:
        pass

    @abc.abstractproperty
    def cls_tok_id(self) -> int:
        pass

    @abc.abstractproperty
    def unk_tok_id(self) -> int:
        pass


class Labels(Cacheable):
    def __init__(self, cache_dir: Path, ignore_cache: bool = False) -> None:
        super().__init__(cache_dir=cache_dir, ignore_cache=ignore_cache)

        self._lbl2id: T.Dict[str, int] = {
            lbl: id_ for id_, lbl in enumerate(self._id2lbl)
        }

    @property
    def cached_attrs(self) -> T.List[str]:
        return ["_id2lbl"]

    def get_lbl_id(self, lbl: str) -> int:
        return self._lbl2id[lbl]

    def get_lbl(self, lbl_id: int) -> str:
        return self._id2lbl[lbl_id]


class BasicVocab(Vocab, Cacheable):
    """
    Supports lowercasing option, and having a minimum count(unk tokens),
    and reserving one initial token ids for special tokens to be used by the model(like [CLS]).
    """

    def __init__(
        self,
        txt_src: TextSource,
        cache_dir: Path,
        tokenizer: tokenizers.base.Tokenizer,
        lower_case: bool = True,
        unk_thres: int = 1,
        ignore_cache: bool = False,
    ) -> None:
        self._lower_case = lower_case
        self._unk_thres = unk_thres
        self._txt_src = txt_src
        self._tokenizer = tokenizer

        self._pad_id = 0
        self._cls_id = 1
        self._unk_id = 2

        Cacheable.__init__(self, cache_dir=cache_dir, ignore_cache=ignore_cache)

        self._word2id: T.Dict[str, int] = {
            word: id_ for id_, word in enumerate(self._id2word)
        }

    def simplify_txt(self, txt: str) -> str:
        """This would be the place to do things like lowercasing, stripping out punctuation, ..etc"""
        if self._lower_case:
            return txt.lower()
        return txt

    @property
    def cached_attrs(self) -> T.List[str]:
        return [
            "_id2word",
        ]

    @property
    def padding_tok_id(self) -> int:
        return 0

    @property
    def cls_tok_id(self) -> int:
        return 1

    @property
    def unk_tok_id(self) -> int:
        return 2

    def __repr__(self) -> str:
        return (
            f"BasicVocab-"
            f"tokenizer_{self.tokenizer}_lower_case_{self._lower_case}-unk_thres_{self._unk_thres}-"
            f"txt_src_{self._txt_src}"
        )

    @property
    def tokenizer(self) -> tokenizers.base.Tokenizer:
        return self._tokenizer

    def get_tok_ids(self, lsword: T.List[str]) -> T.List[int]:
        return [self._word2id.get(word, self.unk_tok_id) for word in lsword]

    def get_toks(self, lsword_id: T.List[int]) -> T.List[str]:
        return [self._id2word[word_id] for word_id in lsword_id]

    def tokenize_and_get_tok_ids(self, txt: str) -> T.List[int]:
        return self.get_tok_ids(self.tokenizer.tokenize(txt))

    def process(self) -> None:

        # Compute word2id

        word_counts: Counter[str] = Counter()
        lslbl: T.List[str] = []

        for lssent, lbl in self._txt_src:
            lslbl.append(lbl)
            for sent in lssent:
                sent = self.simplify_txt(sent)
                lsword = self.tokenizer.tokenize(sent)
                word_counts.update(lsword)

        self._id2word = [
            word for word, count in word_counts.items() if count >= self._unk_thres
        ]
        self._id2word = ["[PAD]", "[CLS]", "[UNK]"] + self._id2word
        self._id2lbl = list(sorted(set(lslbl)))
        self._lblcnt = Counter(lslbl)
        logger.info(f"Made id2word of length {len(self._id2word)}")
        logger.info(f"Made id2lbl of length {len(self._id2lbl)}")


class BertVocab(Vocab):
    _model_name = "bert-base-uncased"

    def __init__(self, tokenizer: tokenizers.bert.WrappedBertTokenizer) -> None:
        """Why pass in tokenizer when we only support tokenizers.bert.WrappedBertTokenizer?
        Because we want to make sure that it is indeed that tokenizer that is being used for this vocab.
        """

        self._tokenizer = tokenizer
        super().__init__()

        self._pad_id = 0
        self._cls_id = 1
        self._unk_id = 2

    def simplify_txt(self, txt: str) -> str:
        # We only support bert-base-uncased right now
        return txt.lower()

    @property
    def padding_tok_id(self) -> int:
        res = self._tokenizer.unwrapped_tokenizer.pad_token_id
        return res  # type: ignore

    @property
    def cls_tok_id(self) -> int:
        res = self._tokenizer.unwrapped_tokenizer.cls_token_id
        return res  # type: ignore

    @property
    def unk_tok_id(self) -> int:
        res = self._tokenizer.unwrapped_tokenizer.unk_token_id
        return res  # type: ignore

    def __repr__(self) -> str:
        return f"BertVocab-" f"model_name_{self._tokenizer.bert_model_name}"

    @property
    def tokenizer(self) -> tokenizers.bert.WrappedBertTokenizer:
        return self._tokenizer

    def get_tok_ids(self, lsword: T.List[str]) -> T.List[int]:
        lstok_id: T.List[
            int
        ] = self.tokenizer.unwrapped_tokenizer.convert_tokens_to_ids(lsword)
        return lstok_id

    def get_toks(self, lsword_id: T.List[int]) -> T.List[str]:
        lstok: T.List[str] = self.tokenizer.unwrapped_tokenizer._convert_token_to_id(
            lsword_id
        )
        return lstok

    def tokenize_and_get_tok_ids(self, txt: str) -> T.List[int]:
        return self.get_tok_ids(self.tokenizer.tokenize(txt))

    def get_lbl_id(self, lbl: str) -> int:
        return self._lbl2id[lbl]

    def get_lbl(self, lbl_id: int) -> str:
        return self._id2lbl[lbl_id]


class SliceDataset(Dataset):  # type: ignore
    def __init__(self, orig_ds: Dataset, n: int) -> None:  # type: ignore
        assert len(orig_ds) >= n
        self.orig_ds = orig_ds
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> T.Any:
        if i >= len(self):
            raise IndexError("SliceDataset ended.")
        return self.orig_ds[i]


if T.TYPE_CHECKING:
    BaseDataset = Dataset[SentgraphExample]
else:
    BaseDataset = Dataset


class SentenceGraphDataset(BaseDataset, Cacheable):
    def __init__(
        self,
        txt_src: TextSource,
        sent2graph: SentenceToGraph,
        vocab: Vocab,
        cache_dir: Path,
        ignore_cache: bool = False,
        processing_batch_size: int = 800,
    ):

        self.sent2graph = sent2graph
        self.txt_src = txt_src
        self.vocab = vocab
        self._processing_batch_size = processing_batch_size

        Cacheable.__init__(self, cache_dir=cache_dir, ignore_cache=ignore_cache)

    @property
    def cached_attrs(self) -> T.List[str]:
        return [
            "_lssentgraph_ex",
        ]

    def __repr__(self) -> str:
        return f"SentenceGraphDataset-sent2graph_{self.sent2graph}-txt_src_{self.txt_src}-vocab_{self.vocab}"

    def process(self) -> None:
        logger.info("Getting sentence graphs ...")
        batch_size = min(self._processing_batch_size, len(self.txt_src))
        # Do batched SRL prediction
        lslssent: T.List[T.List[str]]
        # Group the sent_ex's into batches
        batched_lssent_ex = grouper(self.txt_src, n=batch_size)

        lssentgraph_ex: T.List[SentgraphExample] = []
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
        self, lssent_ex: T.List[SentExample]
    ) -> T.List[SentgraphExample]:

        lslssent: T.List[T.List[str]]
        lslbl: T.List[str]
        lslssent, lslbl = map(list, zip(*lssent_ex))  # type: ignore

        # This part is complicated because we allow an arbitrary number of sentences per example,
        # AND we want to do batched prediction.
        # The trick is, for each batch do one "column" of the batch at a time.

        # Easily deadl with lblids
        lslbl_id = self.vocab.batch_get_lbl_id(lslbl)

        lsper_column_sentgraph: T.List[T.List[SentGraph]] = []
        for per_column_lssent in zip(*lslssent):

            # For turning the setnence into a graph
            # Note that we are bypassing any vocab filtering(like replacing with UNK)
            per_column_lsword = self.vocab.tokenizer.batch_tokenize(
                self.vocab.batch_simplify_txt(list(per_column_lssent))
            )
            per_column_sentgraph = self.sent2graph.batch_to_graph(per_column_lsword)

            # But we do want the UNK tokens for nodeid2wordid
            per_column_nodeid2wordid = self.vocab.batch_get_tok_ids(per_column_lsword)
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
        lslssentgraph: T.List[T.List[SentGraph]] = list(
            map(list, zip(*lsper_column_sentgraph))
        )
        res = [
            SentgraphExample(lssentgraph=lssentgraph, lbl_id=lbl_id)
            for lssentgraph, lbl_id in zip(lslssentgraph, lslbl_id)
        ]
        return res

    def _process_lssent_ex(self, sent_ex: SentExample) -> SentgraphExample:
        lssent, lbl = sent_ex
        lssentgraph: T.List[SentGraph] = []
        for sent in lssent:
            lsword = self.vocab.tokenizer.tokenize(self.vocab.simplify_txt(sent))
            lsedge, lsedge_type, lsimp_node, _ = self.sent2graph.to_graph(lsword)
            nodeid2wordid = self.vocab.tokenize_and_get_tok_ids(sent)
            sentgraph = SentGraph(
                lsedge=lsedge,
                lsedge_type=lsedge_type,
                lsimp_node=lsimp_node,
                nodeid2wordid=nodeid2wordid,
            )
            lssentgraph.append(sentgraph)
        lbl_id = self.vocab.get_lbl_id(lbl)
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
        lssent: T.List[str] = []
        for lsword_id in lslsword_id:
            assert lsword_id is not None
            lssent.append(" ".join(self.vocab.get_toks(lsword_id)))
        lbl = self.vocab.get_lbl(sent_graph_ex.lbl_id)

        return SentExample(lssent=lssent, lbl=lbl)

    def __iter__(self,) -> T.Iterator[SentgraphExample]:
        for i in range(len(self)):
            yield self[i]

    def sentgraph_to_svg(self, sentgraph: SentGraph) -> str:
        import networkx as nx  # type: ignore

        g = nx.DiGraph()

        def quote(s: str) -> str:
            return '"' + s.replace('"', '"') + '"'

        assert sentgraph.nodeid2wordid is not None
        # NetworkX format
        lsnode_word: T.List[T.Tuple[int, T.Dict[str, str]]] = [
            (node_id, {"label": quote(word)})
            for node_id, word in enumerate(self.vocab.get_toks(sentgraph.nodeid2wordid))
        ]

        # Edges in nx format
        lsedge_role: T.List[T.Tuple[int, int, T.Dict[str, str]]] = [
            (n1, n2, {"label": quote(self.sent2graph.id2edge_type[edge_id])})
            for (n1, n2), edge_id in zip(sentgraph.lsedge, sentgraph.lsedge_type)
        ]
        g.add_nodes_from(lsnode_word)
        g.add_edges_from(lsedge_role)
        p = nx.drawing.nx_pydot.to_pydot(g)
        return p.create_svg().decode()  # type: ignore

    @staticmethod
    def collate_fn(
        batch: T.List[SentgraphExample],
    ) -> T.Tuple[T.List[T.List[SentGraph]], T.List[int]]:
        X, y = zip(*batch)
        return list(X), list(y)


SENT2GRAPHS: T.Dict[str, T.Type[SentenceToGraph]] = {
    "srl": SRLSentenceToGraph,
    "dep": DepSentenceToGraph,
}


def load_splits(
    dataset_dir: Path,
    sent2graph_name: TT.Literal["srl", "dep"],
    splits: T.List[str] = ["train", "val"],
    fp_ending: str = "tsv",
    lstxt_col: T.List[str] = ["sentence1", "sentence2"],
    lbl_col: str = "label",
    delimiter: str = "\t",
    unk_thres: int = 2,
) -> T.Tuple[
    T.Dict[str, Dataset[SentgraphExample]],  # line breaker  # lb
    T.Dict[str, CsvTextSource],
    Vocab,
]:

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

    vocab = BasicVocab(
        txt_src=txt_srcs["train"],
        cache_dir=dataset_dir,
        unk_thres=unk_thres,
        tokenizer=tokenizers.Tokenizer(),  # type: ignore # TODO
    )

    cls_sent2graph = SENT2GRAPHS[sent2graph_name]
    split_datasets: T.Dict[str, Dataset[SentgraphExample]] = {
        split: SentenceGraphDataset(
            cache_dir=dataset_dir,
            txt_src=txt_src,
            sent2graph=cls_sent2graph(),
            vocab=vocab,
            processing_batch_size=1000,
        )
        for split, txt_src in txt_srcs.items()
    }

    logger.info("First 10 of each split")
    for split, dataset in split_datasets.items():
        logger.info(f"{split}")
        for i in range(min(len(dataset), 5)):
            lssentgraph, lbl_id = dataset[i]
            logger.info(f"{vocab.get_lbl(lbl_id)}")
            for _, _, _, nodeid2wordid in lssentgraph:
                assert nodeid2wordid is not None
                logger.info(f"\t{vocab.get_toks(nodeid2wordid)}")

    return split_datasets, txt_srcs, vocab
