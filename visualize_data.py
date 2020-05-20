import pandas as pd
import plotly  # type: ignore


def simplify(df: pd.DataFrame) -> None:
    for col in df.columns:
        df[col] = df[col].str.lower()


def word_overlap(df: pd.DataFrame, col1: str, col2: str) -> None:

    col1_split = df[col1].str.split()
    col2_split = df[col2].str.split()

    col1_col2_split = pd.concat([col1_split, col2_split], axis=1)
    overlap = col1_col2_split.apply(lambda sr: sum(i in sr[1] for i in sr[0]), axis=1)
    df[f"_{col1}_{col2}_word_overlap"] = overlap


def length_diff(df: pd.DataFrame, col1: str, col2: str) -> None:
    length_diff = df.apply(lambda sr: abs(len(sr[col1]) - len(sr[col2])), axis=1)
    df[f"_{col1}_{col2}_length_diff"] = length_diff
