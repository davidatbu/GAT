import abc
import typing as T


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def tokenize(self, txt: str) -> T.List[str]:
        pass

    def batch_tokenize(
        self, lstxt: T.List[str], max_len: T.Optional[int] = None
    ) -> T.List[T.List[str]]:
        """Will return a list of lists which are all of max_len. max_len will be set to maximum number of tokens of any one txt
           if it is passed as None.
        """
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass
