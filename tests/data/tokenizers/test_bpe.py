import unittest

from bpemb import BPEmb  # type: ignore

from Gat.data.tokenizers.bpe import BPETokenizer


class TestBPE(unittest.TestCase):
    def setUp(self) -> None:
        bpemb_en = BPEmb(
            lang="en", vs=25000, dim=100, preprocess=False, add_pad_emb=True
        )
        self._tokenizer = BPETokenizer(bpemb_en)

    def test_it(self) -> None:
        test_str = (
            "the diets of the wealthy were rich in sugars, which promoted"
            " periodontal disease. despite"
        )
        self.assertListEqual(
            self._tokenizer.tokenize(test_str),
            [
                "▁the",
                "▁diet",
                "s",
                "▁of",
                "▁the",
                "▁wealthy",
                "▁were",
                "▁rich",
                "▁in",
                "▁sug",
                "ars",
                ",",
                "▁which",
                "▁promoted",
                "▁period",
                "on",
                "tal",
                "▁disease",
                ".",
                "▁despite",
            ],
        )

    def tearDown(self) -> None:
        super().tearDown()


if __name__ == "__main__":
    unittest.main()
