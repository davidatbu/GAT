from pathlib import Path

from .base import load_splits


class TestSvg:
    def setUp(self) -> None:
        datasets, txt_srcs, vocab_and_emb = load_splits(
            # Path("data/glue_data/SST-2"),
            Path("actual_data/SST-2_tiny"),
            splits=["train", "dev"],
            lstxt_col=["sentence"],
            sent2graph_name="dep",
            unk_thres=1,
        )
        self.vocab_and_emb = vocab_and_emb
        self.dataset = datasets["train"]
        self.txt_src = txt_srcs["train"]

    def test_it(self) -> None:
        with open("example.svg", "w") as f:
            sentgraph = self.dataset[2].lssentgraph[0]
            breakpoint()
            svg_str = self.dataset.sentgraph_to_svg(sentgraph)
            f.write(svg_str)
