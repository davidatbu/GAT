import pandas as pd
from nose import main

from visualize_data import simplify
from visualize_data import word_overlap


class TestVisualize:
    def setUp(self) -> None:
        df = pd.DataFrame(
            [
                ["I love you", "you love me", "0"],
                ["i love you", "i adore you", "1"],
                ["Who am I?", "that the highest", "0"],
            ],
            columns=["sent1", "sent2", "label"],
        )
        simplify(df)
        self.df = df

    def test_overlap(self) -> None:
        word_overlap(self.df, "sent1", "sent2")
        print(self.df)


if __name__ == "__main__":
    main()
