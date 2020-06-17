import abc
import typing as T


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def tokenize(self, txt: str) -> T.List[str]:
        pass

    def batch_tokenize(
        self, lstxt: T.List[str], max_len: T.Optional[int] = None
    ) -> T.List[T.List[str]]:
        """Batch version."""
        return [self.tokenize(txt) for txt in lstxt]

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass
