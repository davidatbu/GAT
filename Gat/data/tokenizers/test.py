import nose

from ...data import tokenizers


class TestBertTokenizer:
    def setUp(self) -> None:
        self.tokenizer = tokenizers.bert.WrappedBertTokenizer()

    def test_it(self) -> None:
        res = self.tokenizer.tokenize("embeddings")
        exp = [
            "em",
            "##bed",
            "##ding",
            "##s",
        ]
        nose.tools.eq_(res, exp)


class TestSpacyTokenizer:
    def setUp(self) -> None:
        self.tokenizer = tokenizers.spacy.WrappedSpacyTokenizer()

    def test_it(self) -> None:
        res = self.tokenizer.tokenize("here's one for embeddings")
        nose.tools.eq_(
            res, ["here", "'s", "one", "for", "embeddings",],
        )
