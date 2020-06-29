import abc
import typing as T


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def _tokenize(self, txt: str) -> T.List[str]:
        pass

    @staticmethod
    def split_on_special_toks(txt: str, lsspecial_tok: T.List[str]) -> T.List[str]:
        """Used to avoid splitting in the middle of special tokens.

        >>> Tokenizer.split_on_special_toks(
            ...     txt="[cls]Who's a good doggy?[pad]",
            ...     lsspecial_tok=["[cls]", "[pad]"]
            ... )
        [ "[cls]", "Whos' a good doggy?", "[pad]" ]
        """
        if lsspecial_tok == []:
            return [txt]

        lspart: T.List[str] = []
        tok = lsspecial_tok[0]
        while True:
            try:
                idx = txt.index(tok)
                part_before, part_after = txt[:idx], txt[idx + len(tok) :]
                if part_before:
                    lspart.append(part_before)
                lspart.append(tok)
                txt = part_after

            except ValueError:
                break

        if txt:
            lspart.append(txt)

        # Recurse with the other special tokens
        if len(lsspecial_tok) > 1:
            new_lspart = []
            for part in lspart:
                new_lspart.extend(
                    Tokenizer.split_on_special_toks(part, lsspecial_tok[1:])
                )
        else:
            new_lspart = lspart
        return new_lspart

    def tokenize(
        self, txt: str, lsspecial_tok: T.Optional[T.List[str]] = None
    ) -> T.List[str]:
        """Tokenize, making sure never to "cut across" special tokens."""
        if lsspecial_tok is None:
            return self._tokenize(txt)
        res = []
        for part in self.split_on_special_toks(txt, lsspecial_tok):
            if part in lsspecial_tok:
                res.append(part)
            else:
                res.extend(self._tokenize(part))
        return res

    def batch_tokenize(
        self, lstxt: T.List[str], max_len: T.Optional[int] = None
    ) -> T.List[T.List[str]]:
        """Batch version."""
        return [self.tokenize(txt) for txt in lstxt]

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass
