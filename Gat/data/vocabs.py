import abc
import io
import logging
import typing as T
from pathlib import Path

import torch
import typing_extensions as TT
import youtokentome as yttm  # type: ignore[import]

from Gat.data.cacheable import Cacheable
from Gat.data.cacheable import CachingTool
from Gat.data.cacheable import TorchCachingTool
from Gat.data.numerizer import Numerizer
from Gat.data.sources import TextSource
from Gat.data.tokenizers import Tokenizer
from Gat.data.tokenizers import WrappedBertTokenizer
from Gat.data.tokenizers import WrappedBPETokenizer
from Gat.neural import layers

logger = logging.getLogger(__name__)


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
    def _tokenizer(self) -> Tokenizer:
        pass

    def tokenize(self, txt: str) -> T.List[str]:
        """Vocab.tokenize is different from Tokenizer.tokenize because Vocab.tokenize
        has access to the special tokesn that should not be "cut across", 
        """
        return self._tokenizer.tokenize(txt, lsspecial_tok=self._lsspecial_tok)

    def batch_tokenize(self, lstxt: T.List[str]) -> T.List[T.List[str]]:
        return [self.tokenize(txt) for txt in lstxt]

    def tokenize_and_get_lstok_id(self, txt: str) -> T.List[int]:
        """Convinience function to call tokenize and get tok ids in one."""
        return self.get_lstok_id(self.tokenize(self.simplify_txt(txt)))

    def batch_tokenize_and_get_lstok_id(
        self, lstxt: T.List[str]
    ) -> T.List[T.List[int]]:
        """Batch version."""
        return self.get_lslstok_id(self.batch_tokenize(lstxt))

    @abc.abstractproperty
    def _lsspecial_tok(self) -> T.List[str]:
        "Things like CLS and PAD that should be preserved in tokenization."
        pass

    @property
    def cls_tok(self) -> str:
        return "[cls]"

    @property
    def unk_tok(self) -> str:
        return "[unk]"

    @property
    def padding_tok(self) -> str:
        return "[pad]"

    @abc.abstractproperty
    def vocab_size(self) -> T.Optional[int]:
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


class BasicVocab(Vocab, Cacheable):
    """Vocab subclass that should work for most non-sub-word level tasks.

    Supports lowercasing, having a minimum count(unk tokens).
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        txt_src: TextSource,
        unk_thres: T.Optional[int] = None,
        lower_case: bool = True,
        cache_dir: Path,
        ignore_cache: bool = False,
    ) -> None:
        """Set self._word2id after doing self.process() (via Cacheable.__init__()).

        Args:
            txt_src: Used to build the vocabulary, as well as the list of labels. Can be
                none, if we're building a "no vocabulary" vocab.
            tokenizer: Used to break txt_src examples into tokens and build vocab.
            lower_case: Obvious.
            unk_thres: the minimum num of times a token has to appear to be included
                       in vocab. If None, it means that the vocabulary is built up
                       continously with every request for a tokenization.

                       That means that the ids assigned to each token depend on the
                       order in which the token was "seen".

                       unk_thres should be None only in the case of further doing sub
                       word tokenization, when the token id assigned to the word level
                       token doesn't actually matter.

            cache_dir: Look at Cacheable.__init__
            ignore_cache: Look at Cacheable.__init__

        Sets:
            self._word2id: T.Dict[str, int]
            self._labels: Labels
        """
        self._lower_case = lower_case
        self._unk_thres = unk_thres
        self._txt_src = txt_src
        self.__tokenizer = tokenizer

        super().__init__(cache_dir, ignore_cache)

        self._word2id: T.Dict[str, int] = {
            word: id_ for id_, word in enumerate(self._id2word)
        }
        if unk_thres is not None:
            self._unk_tok_id = self._word2id[self.unk_tok]

    def simplify_txt(self, txt: str) -> str:
        """Lower case if necessary."""
        if self._lower_case:
            return txt.lower()
        return txt

    @property
    def _cached_attrs(self) -> T.Tuple[T.Tuple[str, CachingTool], ...]:
        """Look at superclass doc."""
        return (
            ("_id2word", TorchCachingTool()),
            ("_labels", TorchCachingTool()),
        )

    def get_tok(self, tok_id: int) -> str:
        return self._id2word[tok_id]

    def get_tok_id(self, tok: str) -> int:
        if self._unk_thres is not None:
            return self._word2id.get(tok, self._unk_tok_id)
        else:
            if tok == self.unk_tok:
                raise Exception(
                    "asked to translate unk_tok, but self._unk_thres is None"
                )
            else:
                if tok in self._id2word:
                    return self._word2id[tok]
                else:
                    tok_id = len(self._id2word)
                    self._id2word.append(tok)
                    self._word2id[tok] = tok_id
                    return tok_id

    def __repr__(self) -> str:
        """Look at superclass doc."""
        return (
            f"BasicVocab"
            f"-tokenizer_{self._tokenizer}"
            f"-lower_case_{self._lower_case}"
            f"-unk_thres_{self._unk_thres}"
            f"-txt_src_{self._txt_src}"
        )

    @property
    def vocab_size(self) -> T.Optional[int]:
        if self._unk_thres is not None:
            return len(self._id2word)
        else:
            return None

    @property
    def _tokenizer(self) -> Tokenizer:
        return self.__tokenizer

    @property
    def _lsspecial_tok(self) -> T.List[str]:
        if self._unk_thres is not None:
            return [self.padding_tok, self.cls_tok, self.unk_tok]
        return [self.padding_tok, self.cls_tok]

    def get_lstok(self, lsword_id: T.List[int]) -> T.List[str]:
        return [self.get_tok(word_id) for word_id in lsword_id]

    def process(self) -> None:
        """Look at Cacheable.process.

        Sets:
            self._id2word: List[str]
            self._labels: Labels
        """
        self._id2word: T.List[str] = self._lsspecial_tok[:]
        lslbl: T.List[str] = []
        if self._unk_thres is not None:
            word_counts: T.Counter[str] = T.Counter()

            for lssent, lbl in self._txt_src:
                lslbl.append(lbl)
                for sent in lssent:
                    sent = self.simplify_txt(sent)
                    lsword = self.tokenize(sent)
                    word_counts.update(lsword)

            id2word = [
                word for word, count in word_counts.items() if count >= self._unk_thres
            ]
            self._id2word.extend(id2word)
            logger.info(f"Made id2word of length {len(self._id2word)}")
        else:  # self._unk_thres == None
            _, lslbl = map(list, zip(*self._txt_src))  # type: ignore

        id2lbl = list(sorted(set(lslbl)))
        self._labels = Labels(id2lbl)
        logger.info(f"Made id2lbl of length {len(self.labels.all_lbls)}")

    @property
    def labels(self) -> Labels:
        return self._labels


class BertVocab(Vocab):
    """Wrapper around the tokenizer from the transformers library."""

    def __init__(self) -> None:
        """Extract unique labels."""
        self.__tokenizer = WrappedBertTokenizer()
        super().__init__()

        self._pad_id = 0
        self._cls_id = 1
        self._unk_id = 2

    def simplify_txt(self, txt: str) -> str:
        # We only support bert-base-uncased right now
        return txt.lower()

    @property
    def vocab_size(self) -> int:
        return self.__tokenizer.unwrapped_tokenizer.vocab_size

    @property
    def padding_tok(self) -> str:
        return self.__tokenizer.unwrapped_tokenizer.pad_token

    @property
    def cls_tok(self) -> str:
        return self.__tokenizer.unwrapped_tokenizer.cls_token

    @property
    def sep_tok(self) -> str:
        return self.__tokenizer.unwrapped_tokenizer.sep_token

    @property
    def unk_tok(self) -> str:
        return self.__tokenizer.unwrapped_tokenizer.unk_token

    @property
    def _lsspecial_tok(self) -> T.List[str]:
        return [self.unk_tok, self.cls_tok, self.padding_tok, self.sep_tok]

    def __repr__(self) -> str:
        return "BertVocab-" f"model_name_{self.__tokenizer.bert_model_name}"

    @property
    def _tokenizer(self) -> WrappedBertTokenizer:
        return self.__tokenizer

    def get_tok(self, tok_id: int) -> str:
        return self._tokenizer.unwrapped_tokenizer.convert_ids_to_tokens(tok_id)

    def get_tok_id(self, tok: str) -> int:
        return self._tokenizer.unwrapped_tokenizer.convert_tokens_to_ids(tok)

    def get_lstok_id(self, lsword: T.List[str]) -> T.List[int]:
        lstok_id = self._tokenizer.unwrapped_tokenizer.convert_tokens_to_ids(lsword)
        return lstok_id

    def get_lstok(self, lsword_id: T.List[int]) -> T.List[str]:
        lstok: T.List[str] = self._tokenizer.unwrapped_tokenizer.convert_ids_to_tokens(
            lsword_id
        )
        return lstok


class BPECachingTool(CachingTool):
    def load(self, file_: Path) -> yttm.BPE:
        return yttm.BPE(str(file_.resolve()))

    def save(self, obj: yttm.BPE, file_: Path) -> None:
        # A No-op, since the BPE library requires that one had it saved. Just make
        # sure that it exists.
        assert file_.exists()


class BPEVocab(Vocab, Cacheable):
    def __init__(
        self,
        txt_src: TextSource,
        bpe_vocab_size: int,
        lower_case: bool,
        cache_dir: Path,
        load_pretrained_embs: TT.Literal[False] = False,
        embedding_dim: T.Optional[int] = None,
        ignore_cache: bool = False,
        # Currently, one can't load pretrained embs along
    ):
        if load_pretrained_embs:
            raise NotImplementedError(
                "load_pretrained_embs must be false for BPE right now."
            )
            assert (
                embedding_dim is not None
            ), "must pass embedding_dim if load_pretrained_embs is specified"
        else:
            assert (
                embedding_dim is None
            ), "embedding_dim must not be passed if load_pretrained_embs is False."

        self._txt_src = txt_src
        self._bpe_vocab_size = bpe_vocab_size
        self._lower_case = lower_case

        self._padding_tok_id = 0
        self._unk_tok_id = 1
        self._cls_tok_id = 2

        Cacheable.__init__(self, cache_dir, ignore_cache)

        self._id2word: T.List[str] = self._bpe.vocab()  # Already contains the special
        # tokens
        self._id2word[:3] = ["[pad]", "[unk]", "[cls]"]  # override BPE's specail toks

        self.__tokenizer = WrappedBPETokenizer(self._bpe, repr(self))

        self._word2id: T.Dict[str, int] = {
            word: id_ for id_, word in enumerate(self._id2word)
        }

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"{self.__class__}(",
                f"    txt_src={self._txt_src},",
                "    bpe_vocab_size=self.bpe_vocab_size,",
                "    lower_case={self._lower_case}",
                ")",
            ]
        )

    @property
    def _cached_attrs(self) -> T.Tuple[T.Tuple[str, CachingTool], ...]:
        return (("_bpe", BPECachingTool()),)

    def process(self) -> None:

        # Prepare txt source for BPE, which requires a file interface
        # We use ._specific_cache_dir from Cacheable()
        data_file = self._specific_cache_dir / "bpe_train_file.txt"
        model_file = self._cache_fp_for_attr(
            "_bpe"
        )  # It MUST be this to play nice with Cacheable

        def sent_iterator() -> T.Iterator[str]:
            for lssent, _ in self._txt_src:
                for sent in lssent:
                    yield sent + "\n"

        with data_file.open("w") as f:
            f.writelines(sent_iterator())

        self._bpe = yttm.BPE.train(
            str(data_file),
            str(model_file),
            vocab_size=self._bpe_vocab_size,
            # coverage=0.9999,
            coverage=1,
            bos_id=self._cls_tok_id,
            pad_id=self._padding_tok_id,
            unk_id=self._unk_tok_id,
        )

    def simplify_txt(self, txt: str) -> str:
        if self._lower_case:
            return txt.lower()
        return txt

    @property
    def _tokenizer(self) -> Tokenizer:
        """Return the tokenizer used to produce this vocabulary."""
        return self.__tokenizer

    def get_tok(self, tok_id: int) -> str:
        """

        Raises:
            KeyError: when a token is not in vocab. If you used self.tokenize, this iwll
            never happen.
        """
        return self._id2word[tok_id]

    def get_tok_id(self, tok: str) -> int:
        return self._word2id.get(tok, self._unk_tok_id)

    def get_lstok_id(self, lsword: T.List[str]) -> T.List[int]:
        return [self.get_tok_id(word) for word in lsword]

    @property
    def vocab_size(self) -> int:
        return len(self._id2word)

    def get_lstok(self, lsword_id: T.List[int]) -> T.List[str]:
        return [self.get_tok(word_id) for word_id in lsword_id]

    @property
    def _lsspecial_tok(self) -> T.List[str]:
        return [self.padding_tok, self.cls_tok, self.unk_tok]

    @property
    def pretrained_embs(self) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def unk_tok(self) -> str:
        return self.get_tok(self._unk_tok_id)

    @property
    def cls_tok(self) -> str:
        return self.get_tok(self._cls_tok_id)

    @property
    def padding_tok(self) -> str:
        return self.get_tok(self._padding_tok_id)


__all__ = ["BertVocab", "BPEVocab", "BasicVocab", "Vocab"]
