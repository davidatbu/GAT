import csv
import hashlib
import typing as T
from pathlib import Path

from Gat.utils import SentExample


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
