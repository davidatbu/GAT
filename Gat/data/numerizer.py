import abc
import typing as T

import torch

from Gat.neural import layers


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

    def get_lstok_id(self, lsword: T.List[str]) -> T.List[int]:
        """Get token ids for the tokens in vocab."""
        return [self.get_tok_id(word) for word in lsword]

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

        Currently, we use this only to remove the [cls] token from BERT. (We don't even
        remove the [sep] token with it).

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
        return "[pad]"
