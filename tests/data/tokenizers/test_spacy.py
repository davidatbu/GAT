import unittest

from Gat.data.tokenizers.spacy import WrappedSpacyTokenizer


class TestWrappedSpacyTokenizer(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._tokenizer = WrappedSpacyTokenizer()
        self._lsspecial_tok = ["[PAD]", "[CLS]", "[UNK]"]

    def test_tokenize(self) -> None:
        self.assertListEqual(
            self._tokenizer.tokenize(
                "[cls]who's a good doggy?[pad]", self._lsspecial_tok
            ),
            ["[cls]", "who", "'s", "a", "good", "doggy", "?", "[pad]"],
        )

    def tearDown(self) -> None:
        super().tearDown()


if __name__ == "__main__":
    unittest.main()
