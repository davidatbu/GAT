from ..utils import base as utils


class TestPadLslsid:
    def setUp(self) -> None:
        self.lslsid = [[3, 1, 4], [4], [5, 2, 4, 5]]
        self.padding_tok_id = 0

    def test_it(self) -> None:
        # Max length 1
        tmp = utils.pad_lslsid(
            self.lslsid, padding_tok_id=self.padding_tok_id, max_len=1
        )
        assert tmp == [[3], [4], [5]]

        # @ No max len
        tmp = utils.pad_lslsid(self.lslsid, padding_tok_id=self.padding_tok_id)
        assert tmp == [
            [3, 1, 4, 0],
            [4, 0, 0, 0],
            [5, 2, 4, 5],
        ]

        # @  max len 5
        tmp = utils.pad_lslsid(self.lslsid, padding_tok_id=self.padding_tok_id, max_len=5)
        assert tmp == [
            [3, 1, 4, 0, 0],
            [4, 0, 0, 0, 0],
            [5, 2, 4, 5, 0],
        ]
