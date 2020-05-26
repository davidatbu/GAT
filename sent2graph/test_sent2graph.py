from pathlib import Path
from timeit import default_timer as timer
from typing import List

from data import CsvTextSource
from data import VocabAndEmb
from sent2graph import SRLSentenceToGraph
from utils import flatten


class TestSRLSentenceToGraph:
    def setUp(self) -> None:
        self.sent2graph = SRLSentenceToGraph()
        dataset_dir = Path("data/glue_data/SST-2/")
        txt_src = CsvTextSource(
            fp=dataset_dir / "dev.tsv",
            lstxt_col=["sentence"],
            lbl_col="label",
            allow_unlablled=False,
            csv_reader_kwargs={"delimiter": "\t"},
        )

        # Only for the tokenizing :)
        vocab_and_emb = VocabAndEmb(
            txt_src=txt_src, cache_dir=dataset_dir, embedder=None
        )

        batch_size = 128
        batch = [txt_src[i] for i in range(batch_size)]
        lslssent, _ = zip(*batch)
        lssent: List[str] = list(flatten(lslssent))
        self.lslsword = [vocab_and_emb.tokenize_before_unk(sent) for sent in lssent]

    def test_batched(self) -> None:
        print("Testing batched")
        start = timer()
        res = self.sent2graph.batch_to_graph(self.lslsword)
        end = timer()
        diff = end - start
        print(res)
        print(f"Took {diff} seconds.")

    def test_one_by_one(self) -> None:
        print("Testing one by one")
        start = timer()
        res = [self.sent2graph.to_graph(lsword) for lsword in self.lslsword]
        end = timer()
        diff = end - start
        print(res)
        print(f"Took {diff} seconds.")
