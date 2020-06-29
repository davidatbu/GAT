import abc
import logging
import typing as T
from pathlib import Path

import torch
import typing_extensions as TT
from bpemb import BPEmb  # type: ignore

from Gat.data.cacheable import Cacheable
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
        if self._unk_thres is not None:
            return self._word2id[tok]
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
    def vocab_size(self) -> int:
        return len(self._id2word)

    @property
    def _tokenizer(self) -> Tokenizer:
        return self.__tokenizer

    @property
    def _lsspecial_tok(self) -> T.List[str]:
        return [self.padding_tok, self.cls_tok, self.unk_tok]

    def get_lstok(self, lsword_id: T.List[int]) -> T.List[str]:
        return [self.get_tok(word_id) for word_id in lsword_id]

    def process(self) -> None:
        """Look at Cacheable.process.

        Sets:
            self._id2word: List[str]
            self._labels: Labels
        """
        self._id2word: T.List[str] = [
            self.padding_tok,
            self.cls_tok,
        ]
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
            self._id2word.append(self.unk_tok)
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

    def prepare_for_embedder(
        self,
        lslstok_id: T.List[T.List[int]],
        embedder: layers.Embedder,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Pad/truncate tokens, convert them to torch tensors and move them to device.

        This adds [cls], [sep], and obviously [pad] in the correct spots.
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


class BPEVocab(Vocab):
    def __init__(
        self,
        vocab_size: TT.Literal[25000],
        *,
        # Currently, one MUST load pretrained embs along
        load_pretrained_embs: TT.Literal[True] = True,
        embedding_dim: T.Optional[TT.Literal[300]] = 300,
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

        self._vocab_size = vocab_size + len(self._lsspecial_tok)
        self._lower_case = lower_case
        self.__tokenizer = WrappedBPETokenizer(self._bpemb)

        if load_pretrained_embs:
            self._pretrained_embs: torch.Tensor = torch.zeros(
                [self.vocab_size, embedding_dim], dtype=torch.float,
            )
            self._pretrained_embs[:vocab_size] = torch.from_numpy(self._bpemb.vectors)

        # We add PAD and CLS at the end to avoid further processing the token ids
        # we get from BPE
        self._id2word: T.List[str] = self._bpemb.words + self._lsspecial_tok
        self._word2id: T.Dict[str, int] = {
            word: id_ for id_, word in enumerate(self._id2word)
        }

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
        if tok not in self._word2id:
            tok = self.unk_tok
        return self._word2id[tok]

    def get_lstok_id(self, lsword: T.List[str]) -> T.List[int]:
        """

        Raises:
            KeyError: when a token is not in vocab. If you used self.tokenize, this iwll
            never happen.
        """

        return [self.get_tok_id(word) for word in lsword]

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_lstok(self, lsword_id: T.List[int]) -> T.List[str]:
        return [self.get_tok(word_id) for word_id in lsword_id]

    @property
    def _lsspecial_tok(self) -> T.List[str]:
        return [self.padding_tok, self.cls_tok, self.unk_tok]

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


__all__ = ["BertVocab", "BPEVocab", "BasicVocab", "Vocab"]
