from pathlib import Path

from .base import load_splits


class TestSvg:
    def setUp(self) -> None:
        datasets_per_split, _, vocab_and_emb = load_splits(
            # Path("data/glue_data/SST-2"),
            Path("data/SST-2_tiny"),
            splits=["train", "dev"],
            lstxt_col=["sentence"],
        )
        self.vocab_and_emb = vocab_and_emb
        self.dataset = datasets_per_split["dev"]

    def test_it(self) -> None:
        with open("example.svg", "w") as f:
            sentgraph = self.dataset[2].lssentgraph[0]
            svg_str = self.dataset.sentgraph_to_svg(sentgraph)
            f.write(svg_str)
