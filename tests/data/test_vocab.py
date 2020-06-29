import unittest
from pathlib import Path

from Gat import data
from Gat.data.tokenizers.spacy import WrappedSpacyTokenizer
from Gat.testing_utils import TempDirMixin


class TestBasicVocab(TempDirMixin, unittest.TestCase):
    def setUp(self) -> None:
        """."""
        super().setUp()
        self._txt_src = data.FromIterableTextSource(
            [
                (["Love never fails.", "Love overcomes all things."], "yes"),
                (["Guard your heart.", "From his heart, living waters flow."], "no"),
                (["Always be on guard.", "Be watchful."], "yes"),
            ]
        )
        self._tokenizer = WrappedSpacyTokenizer()

    def tearDown(self) -> None:
        """."""

    def test_with_unk_thres(self) -> None:
        vocab = data.BasicVocab(
            txt_src=self._txt_src,
            tokenizer=self._tokenizer,
            cache_dir=Path(self._temp_dir),
            lower_case=True,
            unk_thres=2,
        )

        expected_setid2word = {
            "[cls]",
            "[pad]",
            "[unk]",
            "guard",
            "love",
            ".",
            "heart",
            "be",
        }
        set_id2word = set(vocab._id2word)
        assert set_id2word == expected_setid2word

    def test_without_unk_thres(self) -> None:
        vocab = data.BasicVocab(
            txt_src=self._txt_src,
            tokenizer=self._tokenizer,
            cache_dir=Path(self._temp_dir),
            lower_case=True,
            unk_thres=None,
        )

        expected_setid2word = {
            "[cls]",
            "[pad]",
        }
        self.assertSetEqual(set(vocab._id2word), expected_setid2word)

        vocab.get_tok_id("SOMETHINGRANDOM")
        self.assertSetEqual(set(vocab._id2word), {"[cls]", "SOMETHINGRANDOM", "[pad]",})


class TestBPEVocab(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self._vocab = data.BPEVocab(
            25000, load_pretrained_embs=True, embedding_dim=300, lower_case=True
        )

    def test_it(self) -> None:
        with self.subTest("padding_tok"):
            self.assertGreaterEqual(
                self._vocab.get_tok_id(self._vocab.padding_tok), self._vocab._bpemb.vs
            )
        with self.subTest("cls_tok"):
            self.assertGreaterEqual(
                self._vocab.get_tok_id(self._vocab.cls_tok), self._vocab._bpemb.vs
            )

    def tearDown(self) -> None:
        super().tearDown()


if __name__ == "__main__":
    unittest.main()
