"""Basic NLP preprocessing. Also, caching preprocessing."""
from __future__ import annotations

import abc
import csv
import hashlib
import logging
import typing as T
from pathlib import Path

import torch
import typing_extensions as TT
from Gat.data import tokenizers
from Gat.neural import layers
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore
from typing_extensions import Counter

from ..sent2graph.base import SentenceToGraph
from ..sent2graph.dep import DepSentenceToGraph
from ..sent2graph.srl import SRLSentenceToGraph
from ..utils import Graph
from ..utils import GraphExample
from ..utils import grouper
from ..utils import SentExample


logger = logging.getLogger(__name__)


class TextSource(T.Iterable[SentExample], T.Sized):
    """A source of labelled examples.

    Important is the __repr__ method. It is used to avoid duplicaiton of
    data processing.
    """

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
    """Can just use a list instead of this, but that wouldn't pass mypy.

    Also, actually turns iterable into list. *facepalm*
    """

    def __init__(self, iterable: T.Iterable[T.Tuple[T.List[str], str]]) -> None:
        """Turn iterable into list and set attribute.

        Args:
            iterable: An iterable that yields a list of sentences and a label.
                      Every yielded example must have the same number of sentences.
        """
        self._ls = list(iterable)

    def __len__(self) -> int:
        """Get length."""
        return len(self._ls)

    def __getitem__(self, idx: int) -> SentExample:
        """Get item."""
        if idx < 0 or idx > len(self):
            raise IndexError(
                f"f{self.__class__.__name__} has only {len(self)} items. {idx} was"
                " asked, which is either negative or greater than length."
            )
        return SentExample(*self._ls[idx])

    def __repr__(self) -> str:
        """Need to use hashlib here because hash() is not reproducible acrosss run."""
        return hashlib.sha1(str(self._ls).encode()).hexdigest()


class ConcatTextSource(TextSource):
    """Not sure why I need this."""

    def __init__(self, arg: TextSource, *args: TextSource):
        """Init."""
        self.lstxt_src = (arg,) + args
        self.lens = list(map(len, self.lstxt_src))

    def __getitem__(self, idx: int) -> SentExample:
        """Get i-th item."""
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
    """Supports reading a CSV with multiple columns of text and one label column."""

    def __init__(
        self,
        fp: Path,
        lstxt_col: T.List[str],
        lbl_col: str,
        csv_reader_kwargs: T.Dict[str, T.Any] = {},
    ) -> None:
        """Init.

        Args:
            lstxt_col: the column headers for the text column.
            lbl_col: the column header for the label column.
            csv_reader_kwargs:
        """
        self.fp = fp

        with fp.open() as f:
            reader = csv.reader(f, **csv_reader_kwargs)
            headers = next(reader)
            for txt_col in lstxt_col:
                if headers.count(txt_col) != 1 or headers.count(lbl_col) != 1:
                    raise Exception(
                        f"{txt_col} or {lbl_col} not found as a header"
                        " in csv flie {str(fp)}, or were found more than once."
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


class Cacheable(abc.ABC):
    """Support caching anything.

    Look at the abstract methods defined below to understand how to use this.
    """

    def __init__(self, cache_dir: Path, ignore_cache: bool) -> None:
        """Check if a cached version is available."""
        # Use the  repr to create a cache dir
        obj_repr_hash = hashlib.sha1(repr(self).encode()).hexdigest()
        self.specific_cache_dir = cache_dir / obj_repr_hash
        self.specific_cache_dir.mkdir(exist_ok=True)

        if self._cached_exists() and not (ignore_cache):
            logger.info(f"{obj_repr_hash} found cached.")
            self._from_cache()
        else:
            logger.info(f"{obj_repr_hash} not found cached. Processing ...")
            self.process()
            self.to_cache()

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Return a unique representation of the settings for the class.

        What is returned must be:
            1. The same across instances that should share the same cached attributes.
            2. Reproducible across multiple Python runs(so `hash()` doesn't work).
        """
        pass

    @abc.abstractproperty
    def _cached_attrs(self) -> T.List[str]:
        """List of attributes that will be cached/restored from cache."""
        pass

    @abc.abstractmethod
    def process(self) -> None:
        """Do the processing that will set the _cached_attrs.

        This function will not be called if a cached version is found.
        After this is called, every attribute in self._cached_attrs must be set.
        """
        pass

    def _cached_exists(self) -> bool:
        """Check if a cached version of the cached attributes exist."""
        return all(
            [
                self._cache_fp_for_attr(attr_name).exists()
                for attr_name in self._cached_attrs
            ]
        )

    def _from_cache(self) -> None:
        """Restore cached attributes."""
        for attr_name in self._cached_attrs:
            fp = self._cache_fp_for_attr(attr_name)

            with fp.open("rb") as fb:
                obj = torch.load(fb)  # type: ignore
            setattr(self, attr_name, obj)

    def _cache_fp_for_attr(self, attr_name: str) -> Path:
        """Return the cache file name for a specific attribute."""
        return self.specific_cache_dir / f"{attr_name}.torch"

    def to_cache(self) -> None:
        """Save cached attributes to cache."""
        for attr_name in self._cached_attrs:

            fp = self._cache_fp_for_attr(attr_name)
            obj = getattr(self, attr_name)
            with fp.open("wb") as fb:
                torch.save(obj, fb)  # type: ignore


class Labels:
    """A class to encapsulate turning labels into ids."""

    def __init__(self, id2lbl: T.List[str]) -> None:
        """Set self._id2lbl.

        Args:
            id2lbl: A list of unique ids. Their position in the list will be their id.
        """
        self._id2lbl = id2lbl

        self._lbl2id: T.Dict[str, int] = {
            lbl: id_ for id_, lbl in enumerate(self._id2lbl)
        }

    def get_lbl_id(self, lbl: str) -> int:
        """Get the id of a label."""
        return self._lbl2id[lbl]

    def get_lbl(self, lbl_id: int) -> str:
        """Given an id, return the label."""
        return self._id2lbl[lbl_id]

    def batch_get_lbl_id(self, lslbl: T.List[str]) -> T.List[int]:
        return [self.get_lbl_id(lbl) for lbl in lslbl]

    def batch_get_lbl(self, lslbl_id: T.List[int]) -> T.List[str]:
        return [self.get_lbl(lbl_id) for lbl_id in lslbl_id]

    @property
    def all_lbls(self) -> T.List[str]:
        return self._id2lbl


class Numerizer(abc.ABC):
    # TODO: Document

    @abc.abstractmethod
    def get_toks(self, lstok_id: T.List[int]) -> T.List[str]:
        """Get a list of tokens given a list of token ids."""
        pass

    def batch_get_toks(self, lslstok_id: T.List[T.List[int]]) -> T.List[T.List[str]]:
        """Batch version of get_toks."""
        return [self.get_toks(lstok_id) for lstok_id in lslstok_id]

    @abc.abstractmethod
    def get_tok_ids(self, lsword: T.List[str]) -> T.List[int]:
        """Get token ids for the tokens in vocab."""
        pass

    def batch_get_tok_ids(self, lslsword: T.List[T.List[str]]) -> T.List[T.List[int]]:
        """Batch version of get_tok_ids."""
        return [self.get_tok_ids(lsword) for lsword in lslsword]

    def prepare_for_embedder(
        self,
        lslstok_id: T.List[T.List[int]],
        embedder: layers.Embedder,
        device: torch.device = torch.device("cpu"),
    ) -> torch.LongTensor:
        """Pad/truncate tokens, convert them to torch tensors and move them to device.

        The padding token is `self.padding_tok_id`. The length of the sequence after
        padding/truncating will be equal to the longest sequence in `lslstok_id` if
        `seq_len` is `None`.

        Args:
            lslstok_id:
            embedder:
            device:

        Returns:
            tok_ids: (B, L, E)
        """
        seq_len = max(map(len, lslstok_id))
        if embedder.max_seq_len is not None and embedder.max_seq_len > seq_len:
            seq_len = embedder.max_seq_len

        padded_lslstok_id = [
            lstok_id[:seq_len] + [self.padding_tok_id] * max(0, seq_len - len(lstok_id))
            for lstok_id in lslstok_id
        ]
        tok_ids: torch.LongTensor = torch.tensor(  # type: ignore
            padded_lslstok_id, dtype=torch.long, device=device,
        )
        return tok_ids.rename("B", "L")  # type: ignore

    @abc.abstractproperty
    def padding_tok_id(self) -> int:
        """The padding token id."""
        pass

    @property
    def padding_tok(self) -> str:
        """The padding token id."""
        # TODO: What about all the other special tokens in Vocab? Why don't they get a
        # property that returns the token, not just the token id?
        # It's not a problem for now, but it's uniuntuitive.
        return "[PAD]"


class Vocab(Numerizer, Cacheable):
    """A class to encapsulate preprocessing of text, and mapping tokens to ids.

    Also contains a `Labels` object.
    """

    def __init__(self, cache_dir: Path, ignore_cache: bool = False) -> None:
        """Look at superclass doc."""
        Cacheable.__init__(self, cache_dir, ignore_cache)

    def simplify_txt(self, txt: str) -> str:
        """Do things like lowercasing stripping out punctuation, ..etc."""
        return txt

    def batch_simplify_txt(self, lstxt: T.List[str]) -> T.List[str]:
        """Call simplify_txt on a batch."""
        return [self.simplify_txt(txt) for txt in lstxt]

    @abc.abstractproperty
    def tokenizer(self) -> tokenizers.Tokenizer:
        """Return the tokenizer used to produce this vocabulary."""
        pass

    def tokenize_and_get_tok_ids(self, txt: str) -> T.List[int]:
        """Convinience function to call tokenize and get tok ids in one."""
        return self.get_tok_ids(self.tokenizer.tokenize(self.simplify_txt(txt)))

    def batch_tokenize_and_get_tok_ids(self, lstxt: T.List[str]) -> T.List[T.List[int]]:
        """Batch version."""
        return self.batch_get_tok_ids(self.tokenizer.batch_tokenize(lstxt))

    @abc.abstractproperty
    def cls_tok_id(self) -> int:
        """The "CLS" token id.

        Currently, we imitate BERT and create a CLS node to connect words to.
        """
        pass

    @abc.abstractproperty
    def unk_tok_id(self) -> int:
        """Token id for uknown token."""
        pass

    @abc.abstractproperty
    def labels(self) -> Labels:
        pass

    @abc.abstractmethod
    def prepare_for_embedder(
        self,
        lslstok_id: T.List[T.List[int]],
        embedder: layers.Embedder,
        device: torch.device = torch.device("cpu"),
    ) -> torch.LongTensor:
        pass

    def strip_after_embedder(self, embs: torch.Tensor) -> torch.Tensor:
        """Strip special tokens after passing through embedder.

        Currently, we use this only to remove the [CLS] token from BERT. (We don't even
        remove the [SEP] token with it).

        Args:
            embs: (B, L, E)

        Returns:
            embs: (B, L, E)
        """
        assert embs.names == ("B", "L", "E")  # type: ignore
        return embs


class BasicVocab(Vocab):
    """Vocab subclass that should work for most non-sub-word level tasks.

    Supports lowercasing, having a minimum count(unk tokens).
    """

    def __init__(
        self,
        txt_src: TextSource,
        tokenizer: tokenizers.base.Tokenizer,
        cache_dir: Path,
        lower_case: bool = True,
        unk_thres: int = 1,
        ignore_cache: bool = False,
    ) -> None:
        """Set self._word2id after doing self.process() (via Cacheable.__init__()).

        Args:
            txt_src: Used to build the vocabulary, as well as the list of labels.
            tokenizer: Used to break txt_src examples into tokens and build vocab.
            lower_case: Obvious.
            unk_thres: the minimum num of times a token has to appear to be included
                       in vocab.
            cache_dir: Look at Cacheable.__init__
            ignore_cache: Look at Cacheable.__init__

        Sets:
            self._word2id: T.Dict[str, int]
            self._labels: Labels
        """
        self._lower_case = lower_case
        self._unk_thres = unk_thres
        self._txt_src = txt_src
        self._tokenizer = tokenizer

        super().__init__(cache_dir, ignore_cache)

        self._word2id: T.Dict[str, int] = {
            word: id_ for id_, word in enumerate(self._id2word)
        }

    def simplify_txt(self, txt: str) -> str:
        """Lower case if necessary."""
        if self._lower_case:
            return txt.lower()
        return txt

    @property
    def _cached_attrs(self) -> T.List[str]:
        """Look at superclass doc."""
        return ["_id2word", "_labels"]

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
        """Look at superclass doc."""
        return (
            f"BasicVocab"
            f"-tokenizer_{self.tokenizer}"
            f"-lower_case_{self._lower_case}"
            f"-unk_thres_{self._unk_thres}"
            f"txt_src_{self._txt_src}"
        )

    @property
    def tokenizer(self) -> tokenizers.base.Tokenizer:
        return self._tokenizer

    def get_tok_ids(self, lsword: T.List[str]) -> T.List[int]:
        return [self._word2id.get(word, self.unk_tok_id) for word in lsword]

    def get_toks(self, lsword_id: T.List[int]) -> T.List[str]:
        return [self._id2word[word_id] for word_id in lsword_id]

    def process(self) -> None:
        """Look at Cacheable.process.

        Sets:
            self._id2word: List[str]
            self._labels: Labels
        """
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

        id2lbl = list(sorted(set(lslbl)))
        self._labels = Labels(id2lbl)
        logger.info(f"Made id2word of length {len(self._id2word)}")
        logger.info(f"Made id2lbl of length {len(self.labels.all_lbls)}")

    @property
    def labels(self) -> Labels:
        return self._labels


class BertVocab(Vocab):
    """Wrapper around the tokenizer from the transformers library."""

    _model_name = "bert-base-uncased"

    def __init__(
        self,
        txt_src: TextSource,
        tokenizer: tokenizers.bert.WrappedBertTokenizer,
        cache_dir: Path,
        ignore_cache: bool = False,
    ) -> None:
        """Extract unique labels."""
        self._tokenizer = tokenizer
        self._txt_src = txt_src
        super().__init__(cache_dir, ignore_cache)

        self._pad_id = 0
        self._cls_id = 1
        self._unk_id = 2

    def simplify_txt(self, txt: str) -> str:
        # We only support bert-base-uncased right now
        return txt.lower()

    @property
    def padding_tok_id(self) -> int:
        res = self._tokenizer.unwrapped_tokenizer.pad_token_id
        return res

    @property
    def cls_tok_id(self) -> int:
        res = self._tokenizer.unwrapped_tokenizer.cls_token_id
        return res

    @property
    def sep_tok_id(self) -> int:
        res = self._tokenizer.unwrapped_tokenizer.sep_token_id
        return res

    @property
    def unk_tok_id(self) -> int:
        res = self._tokenizer.unwrapped_tokenizer.unk_token_id
        return res

    def __repr__(self) -> str:
        return f"BertVocab-" f"model_name_{self._tokenizer.bert_model_name}"

    @property
    def tokenizer(self) -> tokenizers.bert.WrappedBertTokenizer:
        return self._tokenizer

    def get_tok_ids(self, lsword: T.List[str]) -> T.List[int]:
        lstok_id = self.tokenizer.unwrapped_tokenizer.convert_tokens_to_ids(lsword)
        return lstok_id

    def get_toks(self, lsword_id: T.List[int]) -> T.List[str]:
        lstok: T.List[str] = self.tokenizer.unwrapped_tokenizer.convert_ids_to_tokens(
            lsword_id
        )
        return lstok

    @property
    def _cached_attrs(self) -> T.List[str]:
        """Look at Cacheable._cached_attrs."""
        return ["_labels"]

    def process(self) -> None:
        """Look at Cacheable.process().

        Sets:
            self._labels
        """
        lslbl: T.List[str] = []
        for _, lbl in self._txt_src:
            lslbl.append(lbl)
        id2lbl = list(sorted(set(lslbl)))
        self._labels = Labels(id2lbl)

        logger.info(f"Made id2lbl of length {len(self.labels.all_lbls)}")

    @property
    def labels(self) -> Labels:
        return self._labels

    def prepare_for_embedder(
        self,
        lslstok_id: T.List[T.List[int]],
        embedder: layers.Embedder,
        device: torch.device = torch.device("cpu"),
    ) -> torch.LongTensor:
        """Pad/truncate tokens, convert them to torch tensors and move them to device.

        This adds [CLS], [SEP], and obviously [PAD] in the correct spots.
        The length of the sequence after padding/truncating will be equal to the longest
        sequence in `lslstok_id`, or embedder.max_seq_len, whichever is smaller.

        Args:
            lslstok_id:
            embedder:
            device:

        Returns:
            tok_ids: (B, L, E)
        """
        num_special_tokens = 2  # CLS and SEP
        non_special_tok_seq_len = max(map(len, lslstok_id))

        if (
            embedder.max_seq_len is not None
            and non_special_tok_seq_len > embedder.max_seq_len - num_special_tokens
        ):
            non_special_tok_seq_len = embedder.max_seq_len - num_special_tokens

        padded_lslstok_id = [
            [self.cls_tok_id]
            + lstok_id[:non_special_tok_seq_len]
            + [self.sep_tok_id]
            + [self.padding_tok_id] * max(0, non_special_tok_seq_len - len(lstok_id))
            for lstok_id in lslstok_id
        ]
        tok_ids: torch.LongTensor = torch.tensor(  # type: ignore
            padded_lslstok_id, dtype=torch.long, device=device,
        )

        print(tok_ids.size())
        return tok_ids.rename("B", "L")  # type: ignore

    def strip_after_embedder(self, embs: torch.Tensor) -> torch.Tensor:
        """Look at superclass doc."""
        assert embs.names == ("B", "L", "E")  # type: ignore
        return embs[:, 1:]


# Check here
# https://mypy.readthedocs.io/en/stable/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
# for why.
if T.TYPE_CHECKING:
    BaseDataset = Dataset[GraphExample]
else:
    BaseDataset = Dataset


class SentenceGraphDataset(BaseDataset, Cacheable):
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
        sent2graph: SentenceToGraph,
        vocab: Vocab,
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
    def _cached_attrs(self) -> T.List[str]:
        return [
            "_lssentgraph_ex",
        ]

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
            per_column_lsword = self._vocab.tokenizer.batch_tokenize(
                self._vocab.batch_simplify_txt(list(per_column_lssent))
            )
            per_column_sentgraph = self._sent2graph.batch_to_graph(per_column_lsword)

            # But we do want the UNK tokens for nodeid2wordid
            per_column_nodeid2wordid = self._vocab.batch_get_tok_ids(per_column_lsword)
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
            GraphExample(lssentgraph=lssentgraph, lbl_id=lbl_id)
            for lssentgraph, lbl_id in zip(lslssentgraph, lslbl_id)
        ]
        return res

    def _process_lssent_ex(self, sent_ex: SentExample) -> GraphExample:
        lssent, lbl = sent_ex
        lssentgraph: T.List[Graph] = []
        for sent in lssent:
            lsword = self._vocab.tokenizer.tokenize(self._vocab.simplify_txt(sent))
            lsedge, lsedge_type, lsimp_node, _ = self._sent2graph.to_graph(lsword)
            nodeid2wordid = self._vocab.tokenize_and_get_tok_ids(sent)
            sentgraph = Graph(
                lsedge=lsedge,
                lsedge_type=lsedge_type,
                lsimp_node=lsimp_node,
                nodeid2wordid=nodeid2wordid,
            )
            lssentgraph.append(sentgraph)
        lbl_id = self._vocab.labels.get_lbl_id(lbl)
        sentgraph_ex = GraphExample(lssentgraph=lssentgraph, lbl_id=lbl_id)
        return sentgraph_ex

    def __len__(self) -> int:
        return len(self._lssentgraph_ex)

    def __getitem__(self, idx: int) -> GraphExample:
        if idx > len(self):
            raise IndexError(f"{idx} is >= {len(self)}")

        return self._lssentgraph_ex[idx]

    def sentgraph_ex_to_sent_ex(self, sent_graph_ex: GraphExample) -> SentExample:
        lslsword_id = [
            sentgraph.nodeid2wordid for sentgraph in sent_graph_ex.lssentgraph
        ]
        lssent: T.List[str] = []
        for lsword_id in lslsword_id:
            assert lsword_id is not None
            lssent.append(" ".join(self._vocab.get_toks(lsword_id)))
        lbl = self._vocab.labels.get_lbl(sent_graph_ex.lbl_id)

        return SentExample(lssent=lssent, lbl=lbl)

    def __iter__(self,) -> T.Iterator[GraphExample]:
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def collate_fn(
        batch: T.List[GraphExample],
    ) -> T.Tuple[T.List[T.List[Graph]], T.List[int]]:
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
    T.Dict[str, SentenceGraphDataset], T.Dict[str, CsvTextSource], Vocab,
]:
    """Build `Vocab` and `Labels` from training data. Process all splits."""
    assert "train" in splits
    txt_srcs = {
        split: CsvTextSource(
            fp=(dataset_dir / f"{split}.{fp_ending}"),
            lstxt_col=lstxt_col,
            lbl_col=lbl_col,
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
    split_datasets: T.Dict[str, SentenceGraphDataset] = {
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
            logger.info(f"{vocab.labels.get_lbl(lbl_id)}")
            for _, _, _, nodeid2wordid in lssentgraph:
                assert nodeid2wordid is not None
                logger.info(f"\t{vocab.get_toks(nodeid2wordid)}")

    return split_datasets, txt_srcs, vocab
