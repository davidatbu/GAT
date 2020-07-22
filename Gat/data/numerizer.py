import abc
import typing as T

import torch  # type: ignore


class Numerizer(abc.ABC):
    # TODO: Document

    @abc.abstractmethod
    def get_tok_id(self, tok: str) -> int:
        pass

    @abc.abstractmethod
    def get_tok(self, tok_id: int) -> str:
        pass

    @abc.abstractproperty
    def all_tok(self) -> T.List[str]:
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
