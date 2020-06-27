"""Basic NLP preprocessing. Also, caching."""
from __future__ import annotations

import abc
import csv
import hashlib
import logging
import typing as T
from pathlib import Path

import torch
import typing_extensions as TT
from bpemb import BPEmb  # type: ignore
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore
from typing_extensions import Counter

from Gat.data import tokenizers
from Gat.neural import layers
from Gat.sent2graph.base import SentenceToGraph
from Gat.sent2graph.dep import DepSentenceToGraph
from Gat.sent2graph.srl import SRLSentenceToGraph
from Gat.utils import Graph
from Gat.utils import GraphExample
from Gat.utils import grouper
from Gat.utils import SentExample


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
    def get_tok_id(self, tok: str) -> int:
        pass

    @abc.abstractmethod
    def get_tok(self, tok_id: int) -> str:
        pass

    @abc.abstractmethod
    def get_lstok(self, lstok_id: T.List[int]) -> T.List[str]:
        """Get a list of tokens given a list of token ids."""
        pass

    def get_lslstok(self, lslstok_id: T.List[T.List[int]]) -> T.List[T.List[str]]:
        """Batch version of get_lstok."""
        return [self.get_lstok(lstok_id) for lstok_id in lslstok_id]

    @abc.abstractmethod
    def get_lstok_id(self, lsword: T.List[str]) -> T.List[int]:
        """Get token ids for the tokens in vocab."""
        pass

    def get_lslstok_id(self, lslsword: T.List[T.List[str]]) -> T.List[T.List[int]]:
        """Batch version of get_lstok_id."""
        return [self.get_lstok_id(lsword) for lsword in lslsword]

    def prepare_for_embedder(
        self,
        lslstok_id: T.List[T.List[int]],
        embedder: layers.Embedder,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Pad/truncate tokens, convert them to torch tensors and move them to device.

        The padding token is `self.padding_tok_id`. The length of the sequence after
        padding/truncating will be equal to the longest sequence in `lslstok_id` if
        `seq_len` is `None`.

        Args:
            lslstok_id:
            embedder:
            device:

        Returns:
            tok_ids: (B, L)
        """
        seq_len = max(map(len, lslstok_id))
        if embedder.max_seq_len is not None and embedder.max_seq_len > seq_len:
            seq_len = embedder.max_seq_len

        padded_lslstok_id = [
            lstok_id[:seq_len]
            + [self.get_tok_id(self.padding_tok)] * max(0, seq_len - len(lstok_id))
            for lstok_id in lslstok_id
        ]
        tok_ids: torch.Tensor = torch.tensor(
            padded_lslstok_id, dtype=torch.long, device=device,
        )
        # (B, L)
        return tok_ids

    def strip_after_embedder(self, embs: torch.Tensor) -> torch.Tensor:
        """Strip special tokens after passing through embedder.

        Currently, we use this only to remove the [CLS] token from BERT. (We don't even
        remove the [SEP] token with it).

        Args:
            embs: (B, L, E)

        Returns:
            embs: (B, L, E)
        """
        return embs

    @property
    def padding_tok(self) -> str:
        """The padding token id."""
        # TODO: What about all the other special tokens in Vocab? Why don't they get a
        # property that returns the token, not just the token id?
        # It's not a problem for now, but it's uniuntuitive.
        return "[PAD]"


class Vocab(Numerizer):
    """A class to encapsulate preprocessing of text, and mapping tokens to ids.

    Also contains a `Labels` object.
    """

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

    def tokenize_and_get_lstok_id(self, txt: str) -> T.List[int]:
        """Convinience function to call tokenize and get tok ids in one."""
        return self.get_lstok_id(self.tokenizer.tokenize(self.simplify_txt(txt)))

    def batch_tokenize_and_get_lstok_id(
        self, lstxt: T.List[str]
    ) -> T.List[T.List[int]]:
        """Batch version."""
        return self.get_lslstok_id(self.tokenizer.batch_tokenize(lstxt))

    @property
    def cls_tok(self) -> str:
        return "[CLS]"

    @property
    def unk_tok(self) -> str:
        return "[UNK]"

    @abc.abstractproperty
    def vocab_size(self) -> int:
        pass

    @property
    def has_pretrained_embs(self) -> bool:
        """Whether this vocabulary has an .pretrained_embs attribute that we can access to get
        embeddings.
        """
        return False

    @property
    def pretrained_embs(self) -> torch.Tensor:
        raise Exception(f"{self.__class__} doesn't support pretrained embeddings.")


class BPEVocab(Vocab):
    def __init__(
        self,
        vocab_size: T.Literal[25000],
        *,
        # Currently, one MUST load pretrained embs along
        load_pretrained_embs: T.Literal[True] = True,
        embedding_dim: T.Optional[T.Literal[300]] = 300,
        lower_case: bool = True,
    ):
        if not load_pretrained_embs:
            raise NotImplementedError(
                "load_pretrained_embs must be true for BPE right now."
            )
        if load_pretrained_embs and embedding_dim is None:
            raise ValueError(
                "must pass embedding_dim if load_pretrained_embs is specified"
            )
        self._bpemb = BPEmb(
            lang="en",
            vs=vocab_size,
            dim=embedding_dim,
            preprocess=False,
            add_pad_emb=False,
        )

        assert self._bpemb.vs == vocab_size

        self._vocab_size = vocab_size + 2  # One for padding, one for the CLS
        self._lower_case = lower_case
        self._tokenizer = tokenizers.bpe.BPETokenizer(self._bpemb)

        if load_pretrained_embs:
            self._pretrained_embs: torch.Tensor = torch.zeros(
                [self.vocab_size, embedding_dim], dtype=torch.float,
            )
            self._pretrained_embs[:vocab_size] = torch.from_numpy(self._bpemb.vectors)

        # We add PAD and CLS at the end to avoid further processing the token ids
        # we get from BPE
        self._id2word: T.List[str] = self._bpemb.words + [
            self.padding_tok,
            self.cls_tok,
        ]
        self._word2id: T.Dict[str, int] = {
            word: id_ for id_, word in enumerate(self._id2word)
        }

    def simplify_txt(self, txt: str) -> str:
        if self._lower_case:
            return txt.lower()
        return txt

    @property
    def tokenizer(self) -> tokenizers.Tokenizer:
        """Return the tokenizer used to produce this vocabulary."""
        return self._tokenizer

    def get_tok(self, tok_id: int) -> str:
        """

        Raises:
            KeyError: when a token is not in vocab. If you used self.tokenize, this iwll
            never happen.
        """
        return self._id2word[tok_id]

    def get_tok_id(self, tok: str) -> int:
        return self._word2id[tok]

    def get_lstok_id(self, lsword: T.List[str]) -> T.List[int]:
        """

        Raises:
            KeyError: when a token is not in vocab. If you used self.tokenize, this iwll
            never happen.
        """

        return [self._word2id[word] for word in lsword]

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_lstok(self, lsword_id: T.List[int]) -> T.List[str]:
        return [self._id2word[word_id] for word_id in lsword_id]

    @property
    def unk_tok(self) -> str:
        raise ValueError(
            "BPE embeddings have no UNK token, one can always tokenize down to character level."
        )

    @property
    def has_pretrained_embs(self) -> bool:
        return self._pretrained_embs is not None

    @property
    def pretrained_embs(self) -> torch.Tensor:
        if not self.has_pretrained_embs:
            raise Exception(f"{self.__class__} initialized without pretrained embs.")
        else:
            logger.warning(
                "Note that the CLS token doesn't have a pretrained"
                "representation, it's initalized to all zeros, just like the PAD token."
            )
            return self._pretrained_embs


class BasicVocab(Vocab, Cacheable):
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

    def get_tok(self, tok_id: int) -> str:
        return self._id2word[tok_id]

    def get_tok_id(self, tok: str) -> int:
        return self._word2id[tok]

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
    def vocab_size(self) -> int:
        return len(self._id2word)

    @property
    def tokenizer(self) -> tokenizers.base.Tokenizer:
        return self._tokenizer

    def get_lstok_id(self, lsword: T.List[str]) -> T.List[int]:
        return [
            self._word2id.get(word, self.get_tok_id(self.unk_tok)) for word in lsword
        ]

    def get_lstok(self, lsword_id: T.List[int]) -> T.List[str]:
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

        id2word = [
            word for word, count in word_counts.items() if count >= self._unk_thres
        ]
        self._id2word = [self.padding_tok, self.cls_tok, self.unk_tok] + id2word

        id2lbl = list(sorted(set(lslbl)))
        self._labels = Labels(id2lbl)
        logger.info(f"Made id2word of length {len(self._id2word)}")
        logger.info(f"Made id2lbl of length {len(self.labels.all_lbls)}")

    @property
    def labels(self) -> Labels:
        return self._labels


class BertVocab(Vocab):
    """Wrapper around the tokenizer from the transformers library."""

    def __init__(self) -> None:
        """Extract unique labels."""
        self._tokenizer = tokenizers.bert.WrappedBertTokenizer()
        super().__init__()

        self._pad_id = 0
        self._cls_id = 1
        self._unk_id = 2

    def simplify_txt(self, txt: str) -> str:
        # We only support bert-base-uncased right now
        return txt.lower()

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.unwrapped_tokenizer.vocab_size

    @property
    def padding_tok(self) -> str:
        return self._tokenizer.unwrapped_tokenizer.pad_token

    @property
    def cls_tok(self) -> str:
        return self._tokenizer.unwrapped_tokenizer.cls_token

    @property
    def sep_tok(self) -> str:
        return self._tokenizer.unwrapped_tokenizer.sep_token

    @property
    def unk_tok(self) -> str:
        return self._tokenizer.unwrapped_tokenizer.unk_token

    def __repr__(self) -> str:
        return f"BertVocab-" f"model_name_{self._tokenizer.bert_model_name}"

    @property
    def tokenizer(self) -> tokenizers.bert.WrappedBertTokenizer:
        return self._tokenizer

    def get_tok(self, tok_id: int) -> str:
        return self._tokenizer.unwrapped_tokenizer.convert_ids_to_tokens(tok_id)

    def get_tok_id(self, tok: str) -> int:
        return self.tokenizer.unwrapped_tokenizer.convert_tokens_to_ids(tok)

    def get_lstok_id(self, lsword: T.List[str]) -> T.List[int]:
        lstok_id = self.tokenizer.unwrapped_tokenizer.convert_tokens_to_ids(lsword)
        return lstok_id

    def get_lstok(self, lsword_id: T.List[int]) -> T.List[str]:
        lstok: T.List[str] = self.tokenizer.unwrapped_tokenizer.convert_ids_to_tokens(
            lsword_id
        )
        return lstok

    def prepare_for_embedder(
        self,
        lslstok_id: T.List[T.List[int]],
        embedder: layers.Embedder,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Pad/truncate tokens, convert them to torch tensors and move them to device.

        This adds [CLS], [SEP], and obviously [PAD] in the correct spots.
        The length of the sequence after padding/truncating will be equal to the longest
        sequence in `lslstok_id`, or embedder.max_seq_len, whichever is smaller.

        Args:
            lslstok_id:
            embedder:
            device:

        Returns:
            tok_ids: (B, L)
        """
        num_special_tokens = 2  # CLS and SEP
        non_special_tok_seq_len = max(map(len, lslstok_id))

        if (
            embedder.max_seq_len is not None
            and non_special_tok_seq_len > embedder.max_seq_len - num_special_tokens
        ):
            non_special_tok_seq_len = embedder.max_seq_len - num_special_tokens

        cls_tok_id = self.get_tok_id(self.cls_tok)
        sep_tok_id = self.get_tok_id(self.sep_tok)
        padding_tok_id = self.get_tok_id(self.padding_tok)
        padded_lslstok_id = [
            [cls_tok_id]
            + lstok_id[:non_special_tok_seq_len]
            + [sep_tok_id]
            + [padding_tok_id] * max(0, non_special_tok_seq_len - len(lstok_id))
            for lstok_id in lslstok_id
        ]
        tok_ids: torch.Tensor = torch.tensor(
            padded_lslstok_id, dtype=torch.long, device=device,
        )
        # (B, L)

        return tok_ids

    def strip_after_embedder(self, embs: torch.Tensor) -> torch.Tensor:
        """Look at superclass doc.

        Args:
            embs: (B, L, E)

        Returns:
            embs: (B, L, E)
        """
        stripped = embs[:, 1:]
        # (B, L, E)
        return stripped


_T = T.TypeVar("_T")
_S = T.TypeVar("_S")
_D = T.TypeVar("_D")


class NiceDataset(Dataset, T.Iterable[_T], abc.ABC):  # type: ignore
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
        self._total_len = total_len
        self._base_dataset = base_dataset

    def __getitem__(self, i: int) -> _T:
        if i < self._total_len:
            return self._base_dataset[i]
        else:
            raise IndexError()

    def __len__(self) -> int:
        return self._total_len


_GenericVocab = T.TypeVar("_GenericVocab", bound=Vocab, covariant=True)


class BaseSentenceToGraphDataset(NiceDataset[GraphExample], T.Generic[_GenericVocab]):
    @abc.abstractproperty
    def sent2graph(self) -> SentenceToGraph:
        pass

    @abc.abstractproperty
    def vocab(self) -> _GenericVocab:
        pass

    @abc.abstractproperty
    def id2edge_type(self) -> T.List[str]:
        pass

    @abc.abstractproperty
    def edge_type2id(self) -> T.Dict[str, int]:
        pass


class UndirectedDataset(
    TransformedDataset[GraphExample, GraphExample],
    BaseSentenceToGraphDataset[_GenericVocab],
):
    _REVERSED = "_REVERSED"

    def __init__(self, base_dataset: BaseSentenceToGraphDataset[_GenericVocab]):
        self._base_dataset = base_dataset
        self._id2edge_type = self._base_dataset.id2edge_type + [
            edge_type + self._REVERSED for edge_type in base_dataset.id2edge_type
        ]
        self._edge_type2id = {
            edge_type: i for i, edge_type in enumerate(self._id2edge_type)
        }

    @property
    def base_dataset(self) -> BaseSentenceToGraphDataset[_GenericVocab]:
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
            nodeid2wordid=g.nodeid2wordid,
        )
        return new_graph

    @property
    def sent2graph(self) -> SentenceToGraph:
        # Technically not true, since we modify the graph in this class after getting
        # it from this sent2graph
        return self._base_dataset.sent2graph

    @property
    def vocab(self) -> _GenericVocab:
        return self._base_dataset.vocab

    @property
    def id2edge_type(self) -> T.List[str]:
        return self._id2edge_type

    @property
    def edge_type2id(self) -> T.Dict[str, int]:
        return self._edge_type2id


class ConnectToClsDataset(
    TransformedDataset[GraphExample, GraphExample],
    BaseSentenceToGraphDataset[_GenericVocab],
):
    def __init__(self, base_dataset: BaseSentenceToGraphDataset[_GenericVocab]):
        cls_edge_name = "CLS_EDGE"
        assert cls_edge_name not in base_dataset.id2edge_type
        self._base_dataset = base_dataset
        self._id2edge_type = self._base_dataset.id2edge_type + [cls_edge_name]
        self._cls_edge_id = len(self._id2edge_type) - 1
        self._edge_type2id = self._base_dataset.edge_type2id.copy()
        self._edge_type2id.update({cls_edge_name: self._cls_edge_id})

    @property
    def base_dataset(self) -> BaseSentenceToGraphDataset[_GenericVocab]:
        return self._base_dataset

    def _transform(self, example: GraphExample) -> GraphExample:
        cls_tok_id = self.base_dataset.vocab.get_tok_id(self.base_dataset.vocab.cls_tok)
        new_example = GraphExample(
            [
                self._connect_imp_nodes_to_new_node(g, cls_tok_id, self._cls_edge_id)
                for g in example.lsgraph
            ],
            example.lbl_id,
        )
        return new_example

    @property
    def sent2graph(self) -> SentenceToGraph:
        # Technically not true, since we modify the graph in this class after getting
        # it from this sent2graph
        return self._base_dataset.sent2graph

    @property
    def vocab(self) -> _GenericVocab:
        return self._base_dataset.vocab

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

        assert g.nodeid2wordid is not None
        if new_node_global_id in g.nodeid2wordid:
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
            nodeid2wordid=[new_node_global_id] + g.nodeid2wordid,
        )
        return new_graph


class SentenceGraphDataset(BaseSentenceToGraphDataset[BasicVocab], Cacheable):
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
        vocab: BasicVocab,
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
    def sent2graph(self) -> SentenceToGraph:
        return self._sent2graph

    @property
    def vocab(self) -> BasicVocab:
        return self._vocab

    @property
    def id2edge_type(self) -> T.List[str]:
        return self._sent2graph.id2edge_type

    @property
    def edge_type2id(self) -> T.Dict[str, int]:
        return self._sent2graph.edge_type2id

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
            lsword = self._vocab.tokenizer.tokenize(self._vocab.simplify_txt(sent))
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


SENT2GRAPHS: T.Dict[str, T.Type[SentenceToGraph]] = {
    "srl": SRLSentenceToGraph,
    "dep": DepSentenceToGraph,
}


def load_splits(
    dataset_dir: Path,
    sent2graph_name: TT.Literal["srl", "dep"],
    splits: T.List[str] = ["train", "val"],
    fp_ending: str = "tsv",
    lstxt_col: T.List[str] = ["sentence"],
    lbl_col: str = "label",
    delimiter: str = "\t",
    unk_thres: int = 2,
) -> T.Tuple[
    T.Dict[str, BaseSentenceToGraphDataset[BasicVocab]],
    T.Dict[str, CsvTextSource],
    BasicVocab,
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
        tokenizer=tokenizers.spacy.WrappedSpacyTokenizer(),
    )

    cls_sent2graph = SENT2GRAPHS[sent2graph_name]
    split_datasets: T.Dict[str, BaseSentenceToGraphDataset[BasicVocab]] = {
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
                logger.info(f"\t{vocab.get_lstok(nodeid2wordid)}")

    return split_datasets, txt_srcs, vocab
