import csv
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import torch
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from embeddings import WordToVec
from sent2graph import SentenceToGraph


class TextSource:
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        raise NotImplementedError()

    def __len__(self, idx: int) -> int:
        raise NotImplementedError()


class CsvTextSource:
    def __init__(
        self, fp: Path, txt_col: str, lbl_col: str, allow_unlablled: bool
    ) -> None:

        with fp.open() as f:
            reader = csv.reader(f)
            headers = next(reader)
            if headers.count(txt_col) != 1 or headers.count(lbl_col) != 1:
                raise Exception(
                    f"{txt_col} or {lbl_col} not found as a header in csv flie {str(fp)}, or were found more than once."
                )
            txt_col_i = headers.index(txt_col)
            lbl_col_i = headers.index(lbl_col)

            self.rows = [(row[txt_col_i], row[lbl_col_i]) for row in reader]

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.rows[idx]

    def __len__(self, idx: int) -> int:
        return len(self.rows)


class SentenceGraphDataset:
    def __init__(
        self,
        cache_dir: Path,
        use_cache: bool,
        undirected_edges: bool,
        sent2graph: SentenceToGraph,
        embedder: Optional[WordToVec] = None,
        transform: Any = None,
        pre_transform: Any = None,
    ):
        self.embedder = embedder
        self.use_cache = use_cache
        self.sent2graph = sent2graph
        self.undirected_edges = undirected_edges
        self.cache_dir = cache_dir

        # For caching
        self.train_mask: Optional[torch.Tensor] = None
        self.val_mask: Optional[torch.Tensor] = None
        self.test_mask: Optional[torch.Tensor] = None

        self.data, self.slices = torch.load(self.processed_paths[0])  # type: ignore

    def __repr__(self) -> str:
        return "-".join(
            [
                f"{attr}_{str(getattr(self, attr))}"
                for attr in [
                    "undirected_edges",
                    "embedder",
                    "sent2graph",
                    "text_source",
                ]
            ]
        )

    @property
    def processed_file_names(self) -> List[str]:
        return [f"str(self)_data.pkl"]

    def download(self) -> None:
        raise NotImplementedError()

    def process(self) -> None:
        # Read data into huge `Data` list.
        data_list = [...]

        lssent = self.get_lssent()

        splitter = SpacyWordSplitter()
        lslsword: List[Tuple[str, ...]] = [
            tuple(token.text for token in lstoken)
            for lstoken in splitter.batch_split_words(lssent)
        ]
        unique_words: Set[str] = set()
        for lsword in lslsword:
            for word in lsword:
                unique_words.add(word)
        id2word: List[str] = list(sorted(unique_words))
        word2id: Dict[str, int] = {word: i for i, word in enumerate(id2word)}

        for lsword in lslsword:
            lshead_node, lsedge_index, lsedge_type = self.sent2graph.to_graph(lsword)
            # We get indices relative to sentence beginngig, convert these to global ids
            global_word_ids = [word2id[word] for word in lsword]
            global_lshead_node = [global_word_ids[id_] for id_ in lshead_node]
            global_lsedge_index = [
                (global_word_ids[edge_x], global_word_ids[edge_y])
                for edge_x, edge_y in lsedge_index
            ]
            torch.relu

        torch.save(data_list, self.processed_paths[0])  # type: ignore
