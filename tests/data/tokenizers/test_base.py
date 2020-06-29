import unittest

from Gat.data.tokenizers.base import Tokenizer


class TestTokenizer(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_split_on_special_toks(self) -> None:
        lsspecial_tok = ["[pad]", "[cls]", "[unk]"]
        self.assertListEqual(
            Tokenizer.split_on_special_toks(
                "[cls]Who's a good doggy?[pad]", lsspecial_tok
            ),
            ["[cls]", "Who's a good doggy?", "[pad]"],
        )

        self.assertListEqual(
            Tokenizer.split_on_special_toks(
                "Many times, rare words like [unk] are not that rare.", lsspecial_tok
            ),
            ["Many times, rare words like ", "[unk]", " are not that rare."],
        )
        self.assertListEqual(
            Tokenizer.split_on_special_toks(
                "Many times, rare words like [unk] are not that rare.", lsspecial_tok
            ),
            ["Many times, rare words like ", "[unk]", " are not that rare."],
        )
        self.assertListEqual(
            Tokenizer.split_on_special_toks("[cls]", lsspecial_tok), ["[cls]"],
        )

    def tearDown(self) -> None:
        super().tearDown()


if __name__ == "__main__":
    unittest.main()
